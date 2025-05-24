import os
import argparse
import glob
from PIL import Image
import numpy as np
import numbers
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import functional as TF

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from einops import rearrange


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(
            dim, hidden_features * 2, kernel_size=1, bias=bias
        )
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias
        )

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(
            q, "b (head c) h w -> b head c (h w)", head=self.num_heads
        )
        k = rearrange(
            k, "b (head c) h w -> b head c (h w)", head=self.num_heads
        )
        v = rearrange(
            v, "b (head c) h w -> b head c (h w)", head=self.num_heads
        )
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(
            out,
            "b head c (h w) -> b (head c) h w",
            head=self.num_heads,
            h=h,
            w=w,
        )
        out = self.project_out(out)
        return out


class resblock(nn.Module):
    def __init__(self, dim):
        super(resblock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                dim, dim, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PReLU(),
            nn.Conv2d(
                dim, dim, kernel_size=3, stride=1, padding=1, bias=False
            ),
        )

    def forward(self, x):
        res = self.body((x))
        res += x
        return res


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat,
                n_feat // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat,
                n_feat * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class TransformerBlock(nn.Module):
    def __init__(
        self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type
    ):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class PromptGenBlock(nn.Module):
    def __init__(
        self, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192
    ):
        super(PromptGenBlock, self).__init__()
        self.prompt_len = prompt_len
        self.prompt_param = nn.Parameter(
            torch.rand(
                1, self.prompt_len, prompt_dim, prompt_size, prompt_size
            )
        )
        self.linear_layer = nn.Linear(lin_dim, self.prompt_len)
        self.conv3x3 = nn.Conv2d(
            prompt_dim,
            prompt_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        prompt = (
            prompt_weights.view(B, self.prompt_len, 1, 1, 1)
            * self.prompt_param
        )

        prompt = torch.sum(prompt, dim=1)
        prompt = F.interpolate(
            prompt, (H, W), mode="bilinear", align_corners=False
        )
        prompt = self.conv3x3(prompt)
        return prompt


class PromptIR(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
        decoder_use_prompts=True,
        prompt_len=5,
    ):
        super(PromptIR, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.decoder_use_prompts = decoder_use_prompts

        if self.decoder_use_prompts:
            self.prompt1 = PromptGenBlock(
                prompt_dim=64,
                prompt_len=prompt_len,
                prompt_size=64,
                lin_dim=dim * 2,
            )
            self.prompt2 = PromptGenBlock(
                prompt_dim=128,
                prompt_len=prompt_len,
                prompt_size=32,
                lin_dim=dim * 4,
            )
            self.prompt3 = PromptGenBlock(
                prompt_dim=320,
                prompt_len=prompt_len,
                prompt_size=16,
                lin_dim=dim * 8,
            )

        self.chnl_reduce1 = nn.Conv2d(64, 64, kernel_size=1, bias=bias)
        self.chnl_reduce2 = nn.Conv2d(128, 128, kernel_size=1, bias=bias)
        self.chnl_reduce3 = nn.Conv2d(320, 256, kernel_size=1, bias=bias)

        self.reduce_noise_channel_1 = nn.Conv2d(
            dim + 64, dim, kernel_size=1, bias=bias
        )
        self.encoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )
        self.down1_2 = Downsample(dim)
        self.reduce_noise_channel_2 = nn.Conv2d(
            int(dim * 2**1) + 128, int(dim * 2**1), kernel_size=1, bias=bias
        )
        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )
        self.down2_3 = Downsample(int(dim * 2**1))
        self.reduce_noise_channel_3 = nn.Conv2d(
            int(dim * 2**2) + 256, int(dim * 2**2), kernel_size=1, bias=bias
        )
        self.encoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[2])
            ]
        )
        self.down3_4 = Downsample(int(dim * 2**2))
        self.latent = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**3),
                    num_heads=heads[3],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[3])
            ]
        )

        self.up4_3 = Upsample(int(dim * 2**3))
        self.reduce_chan_level3 = nn.Conv2d(
            int(dim * 2**2) + int(dim * 2**2),
            int(dim * 2**2),
            kernel_size=1,
            bias=bias,
        )

        self.noise_level3_dim_input = int(dim * 2**3) + 320
        self.noise_level3 = TransformerBlock(
            dim=self.noise_level3_dim_input,
            num_heads=heads[3],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )
        self.reduce_noise_level3 = nn.Conv2d(
            self.noise_level3_dim_input,
            int(dim * 2**3),
            kernel_size=1,
            bias=bias,
        )

        self.decoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[2])
            ]
        )

        self.up3_2 = Upsample(int(dim * 2**2))
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 2**1) + int(dim * 2**1),
            int(dim * 2**1),
            kernel_size=1,
            bias=bias,
        )

        self.noise_level2_dim_input = int(dim * 2**2) + 128
        self.noise_level2 = TransformerBlock(
            dim=self.noise_level2_dim_input,
            num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )
        self.reduce_noise_level2 = nn.Conv2d(
            self.noise_level2_dim_input,
            int(dim * 2**2),
            kernel_size=1,
            bias=bias,
        )

        self.decoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(int(dim * 2**1))

        self.noise_level1_dim_input = int(dim * 2**1) + 64
        self.noise_level1 = TransformerBlock(
            dim=self.noise_level1_dim_input,
            num_heads=heads[0],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )
        self.reduce_noise_level1 = nn.Conv2d(
            self.noise_level1_dim_input,
            int(dim * 2**1),
            kernel_size=1,
            bias=bias,
        )

        self.decoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**0) + int(dim * 2**0),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )
        self.refinement = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**0) + int(dim * 2**0),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_refinement_blocks)
            ]
        )
        self.output = nn.Conv2d(
            int(dim * 2**0) + int(dim * 2**0),
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )

    # Make sure forward uses self.decoder_use_prompts
    def forward(self, inp_img, noise_emb=None):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        current_latent = latent
        if self.decoder_use_prompts:  # Check the flag
            dec3_param = self.prompt3(current_latent)
            current_latent = torch.cat([current_latent, dec3_param], 1)
            current_latent = self.noise_level3(current_latent)
            current_latent = self.reduce_noise_level3(current_latent)

        inp_dec_level3 = self.up4_3(current_latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        current_dec_level3 = out_dec_level3
        if self.decoder_use_prompts:  # Check the flag
            dec2_param = self.prompt2(current_dec_level3)
            current_dec_level3 = torch.cat([current_dec_level3, dec2_param], 1)
            current_dec_level3 = self.noise_level2(current_dec_level3)
            current_dec_level3 = self.reduce_noise_level2(current_dec_level3)

        inp_dec_level2 = self.up3_2(current_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        current_dec_level2 = out_dec_level2
        if self.decoder_use_prompts:  # Check the flag
            dec1_param = self.prompt1(current_dec_level2)
            current_dec_level2 = torch.cat([current_dec_level2, dec1_param], 1)
            current_dec_level2 = self.noise_level1(current_dec_level2)
            current_dec_level2 = self.reduce_noise_level1(current_dec_level2)

        inp_dec_level1 = self.up2_1(current_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)

        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img
        return out_dec_level1


class HomeworkTrainDataset(Dataset):
    def __init__(self, data_dir, patch_size, image_files_list):
        super().__init__()
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.image_files = image_files_list
        self.degraded_dir = os.path.join(data_dir, "train", "degraded")
        self.clean_dir = os.path.join(data_dir, "train", "clean")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        degraded_fname, deg_type = self.image_files[idx]

        clean_fname_prefix = deg_type + "_clean-"
        clean_fname_suffix = degraded_fname.split("-")[-1]
        clean_fname = clean_fname_prefix + clean_fname_suffix

        degraded_img_path = os.path.join(self.degraded_dir, degraded_fname)
        clean_img_path = os.path.join(self.clean_dir, clean_fname)

        degraded_img = Image.open(degraded_img_path).convert("RGB")
        clean_img = Image.open(clean_img_path).convert("RGB")

        i, j, h, w = T.RandomCrop.get_params(
            degraded_img, output_size=(self.patch_size, self.patch_size)
        )
        degraded_patch = TF.to_tensor(TF.crop(degraded_img, i, j, h, w))
        clean_patch = TF.to_tensor(TF.crop(clean_img, i, j, h, w))

        return degraded_patch, clean_patch


class HomeworkValDataset(Dataset):
    def __init__(self, data_dir, patch_size, image_files_list):
        super().__init__()
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.image_files = image_files_list
        self.degraded_dir = os.path.join(data_dir, "train", "degraded")
        self.clean_dir = os.path.join(data_dir, "train", "clean")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        degraded_fname, deg_type = self.image_files[idx]

        clean_fname_prefix = deg_type + "_clean-"
        clean_fname_suffix = degraded_fname.split("-")[-1]
        clean_fname = clean_fname_prefix + clean_fname_suffix

        degraded_img_path = os.path.join(self.degraded_dir, degraded_fname)
        clean_img_path = os.path.join(self.clean_dir, clean_fname)

        degraded_img = Image.open(degraded_img_path).convert("RGB")
        clean_img = Image.open(clean_img_path).convert("RGB")

        i, j, h, w = T.RandomCrop.get_params(
            degraded_img, output_size=(self.patch_size, self.patch_size)
        )
        degraded_patch = TF.to_tensor(TF.crop(degraded_img, i, j, h, w))
        clean_patch = TF.to_tensor(TF.crop(clean_img, i, j, h, w))

        return degraded_patch, clean_patch


class HomeworkTestDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.degraded_dir = os.path.join(data_dir, "test", "degraded")
        self.image_files = sorted(
            glob.glob(os.path.join(self.degraded_dir, "*.png"))
        )
        self.transforms = T.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        degraded_img_path = self.image_files[idx]
        filename = os.path.basename(degraded_img_path)
        degraded_img = Image.open(degraded_img_path).convert("RGB")
        degraded_tensor = self.transforms(degraded_img)
        return degraded_tensor, filename


def calculate_psnr(img1, img2, max_val=1.0):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return torch.tensor(float("inf")).to(img1.device)
    return 20 * torch.log10(max_val / torch.sqrt(mse))


class RestorationModel(pl.LightningModule):
    def __init__(
        self,
        learning_rate=2e-4,
        warmup_epochs=15,
        max_epochs=150,
        promptir_dim=48,
        log_image_every_n_epochs=10,
        num_val_images_to_log=4,
        use_prompts=True,
        loss_function="l1",
        promptir_num_blocks=[4, 6, 6, 8],
        promptir_heads=[1, 2, 4, 8],
        promptir_ffn_expansion_factor=2.66,
        prompt_len=5,
    ):
        super().__init__()
        self.save_hyperparameters(
            "learning_rate",
            "warmup_epochs",
            "max_epochs",
            "promptir_dim",
            "log_image_every_n_epochs",
            "num_val_images_to_log",
            "use_prompts",
            "loss_function",
            "promptir_num_blocks",
            "promptir_heads",
            "promptir_ffn_expansion_factor",
            "prompt_len",
        )

        self.model = PromptIR(
            dim=self.hparams.promptir_dim,
            decoder_use_prompts=self.hparams.use_prompts,
            num_blocks=self.hparams.promptir_num_blocks,
            heads=self.hparams.promptir_heads,
            ffn_expansion_factor=self.hparams.promptir_ffn_expansion_factor,
            prompt_len=self.hparams.prompt_len,
        )

        if self.hparams.loss_function == "l1":
            self.loss_fn = nn.L1Loss()
        elif self.hparams.loss_function == "mse":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(
                f"Unsupported loss function: {self.hparams.loss_function}"
            )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        degraded_patch, clean_patch = batch
        restored_patch = self.model(degraded_patch)
        loss = self.loss_fn(restored_patch, clean_patch)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        train_psnr = calculate_psnr(restored_patch, clean_patch)
        self.log(
            "train_psnr",
            train_psnr,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        degraded_patch, clean_patch = batch
        restored_patch = self.model(degraded_patch)
        loss = self.loss_fn(restored_patch, clean_patch)
        psnr = calculate_psnr(restored_patch, clean_patch)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_psnr", psnr, on_epoch=True, prog_bar=True, logger=True)

        if (
            self.logger
            and self.logger.experiment
            and (self.current_epoch + 1)
            % self.hparams.log_image_every_n_epochs
            == 0
            and batch_idx < self.hparams.num_val_images_to_log
        ):
            num_images_to_log_actual = min(
                degraded_patch.size(0),
                self.hparams.num_val_images_to_log
                - batch_idx * degraded_patch.size(0),
            )

            for i in range(num_images_to_log_actual):
                if (
                    batch_idx * degraded_patch.size(0) + i
                    < self.hparams.num_val_images_to_log
                ):
                    grid = torch.stack(
                        [
                            degraded_patch[i],
                            torch.clamp(restored_patch[i], 0, 1),
                            clean_patch[i],
                        ]
                    )
                    self.logger.experiment.add_images(
                        f"{self.current_epoch+1}_batch{batch_idx}_img{i}",
                        grid,
                        self.current_epoch + 1,
                    )
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-2
        )

        if (
            self.hparams.warmup_epochs > 0
            and self.hparams.warmup_epochs < self.hparams.max_epochs
        ):
            scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                total_iters=self.hparams.warmup_epochs,
            )
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.max_epochs - self.hparams.warmup_epochs,
                eta_min=1e-6,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[scheduler_warmup, scheduler_cosine],
                milestones=[self.hparams.warmup_epochs],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6
            )

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


def train_model(args):
    all_image_files = []
    degraded_dir = os.path.join(args.data_dir, "train", "degraded")
    for degradation_type in ["rain", "snow"]:
        files = glob.glob(
            os.path.join(degraded_dir, f"{degradation_type}-*.png")
        )
        all_image_files.extend(
            [(os.path.basename(f), degradation_type) for f in files]
        )

    random.seed(args.seed)
    random.shuffle(all_image_files)

    split_idx = int(len(all_image_files) * (1 - args.val_split_ratio))
    train_files_list = all_image_files[:split_idx]
    val_files_list = all_image_files[split_idx:]

    if not train_files_list:
        raise ValueError(
            "Training dataset is empty. Check data_dir and val_split_ratio."
        )
    if not val_files_list and args.val_split_ratio > 0:
        print(
            "Warning: Validation dataset is empty"
        )

    train_dataset = HomeworkTrainDataset(
        args.data_dir,
        patch_size=args.patch_size,
        image_files_list=train_files_list,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if val_files_list:
        val_dataset = HomeworkValDataset(
            args.data_dir,
            patch_size=args.patch_size,
            image_files_list=val_files_list,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    model = RestorationModel(
        learning_rate=args.lr,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        promptir_dim=args.promptir_dim,
        log_image_every_n_epochs=args.log_image_every_n_epochs,
        num_val_images_to_log=args.num_val_images_to_log,
        # --- Pass new experimental args ---
        use_prompts=args.use_prompts,
        loss_function=args.loss_function,
        promptir_num_blocks=args.promptir_num_blocks,
        promptir_heads=args.promptir_heads,
        promptir_ffn_expansion_factor=args.promptir_ffn_expansion_factor,
        prompt_len=args.prompt_len,
    )
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    monitor_metric = "val_psnr" if val_loader else "train_psnr"
    monitor_mode = "max"

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="best_model-{epoch:02d}-{" + monitor_metric + ":.2f}",
        save_top_k=1,
        verbose=True,
        monitor=monitor_metric,
        mode=monitor_mode,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    experiment_name_parts = [
        "restoration",
        f"loss_{args.loss_function}",
        f"prompts_{'on' if args.use_prompts else 'off'}",
        f"plen_{args.prompt_len}" if args.use_prompts else "",
        f"blocks_{'_'.join(map(str, args.promptir_num_blocks))}",
        f"heads_{'_'.join(map(str, args.promptir_heads))}",
        f"ffn_{args.promptir_ffn_expansion_factor}",
    ]
    experiment_name = "-".join(filter(None, experiment_name_parts))

    logger = TensorBoardLogger("tb_logs", name=experiment_name)

    selected_devices = None
    accelerator = "cpu"
    if torch.cuda.is_available():
        if args.device:
            try:
                if "," in args.device:
                    selected_devices = [
                        int(d.strip()) for d in args.device.split(",")
                    ]
                else:
                    selected_devices = [int(args.device)]
                accelerator = "gpu"
            except ValueError:
                print(
                    f"Invalid device string: {args.device}"
                )
                if torch.cuda.device_count() > 0:
                    selected_devices = 1
                    accelerator = "gpu"
        elif torch.cuda.device_count() > 0:
            selected_devices = 1
            accelerator = "gpu"

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        accelerator=accelerator,
        devices=selected_devices if accelerator == "gpu" else 1,
        precision=16 if args.use_amp and accelerator == "gpu" else 32,
        deterministic=True,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader if val_loader else None,
    )
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    if checkpoint_callback.best_model_score is not None:
        print(
            f"({monitor_metric}): {checkpoint_callback.best_model_score:.4f}"
        )
    else:
        print(
            f"Best model score ({monitor_metric}):"
        )


def run_inference(args):
    test_dataset = HomeworkTestDataset(args.data_dir)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers
    )

    device_to_map = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if torch.cuda.is_available() and args.device:
        try:
            gpu_id_to_use = int(args.device.split(",")[0].strip())
            if 0 <= gpu_id_to_use < torch.cuda.device_count():
                device_to_map = torch.device(f"cuda:{gpu_id_to_use}")
            else:
                print(
                    f"Specified GPU ID {gpu_id_to_use} is invalid"
                )
        except ValueError:
            print(
                f"Invalid device string for inference: {args.device}"
            )

    model = RestorationModel.load_from_checkpoint(
        args.checkpoint_path,
        map_location=device_to_map,
        promptir_dim=args.promptir_dim,
    )
    model.eval()
    model.to(device_to_map)

    predictions = {}
    with torch.no_grad():
        for batch_idx, (degraded_tensor, filename_list) in enumerate(
            test_loader
        ):
            filename = filename_list[0]
            degraded_tensor = degraded_tensor.to(model.device)

            restored_tensor = model(degraded_tensor)

            restored_tensor = torch.clamp(restored_tensor, 0, 1)

            output_np = (
                restored_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
            )
            output_np = (output_np * 255).astype(np.uint8)

            predictions[filename] = output_np
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(test_loader)} images")

    np.savez(args.output_npz_path, **predictions)
    print(f"Predictions saved to {args.output_npz_path}")

    if args.visualize_inference_output_dir:
        visualize_predictions_from_npz(
            args.output_npz_path, args.visualize_inference_output_dir
        )


def visualize_predictions_from_npz(npz_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(
        "Loading predictions and saving visualizations"
    )
    data = np.load(npz_path)
    for filename in data.files:
        img_array = data[filename]
        img = Image.fromarray(img_array)
        img.save(os.path.join(output_dir, filename))
    print(f"Visualizations saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Restoration Homework")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "infer"],
        help="Operating mode: train or infer",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./hw4_realse_dataset",
        help="Root directory of the dataset",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to checkpoint for inference or resuming training",
    )
    parser.add_argument(
        "--output_npz_path",
        type=str,
        default="./pred.npz",
        help="Path to save the .npz output file for inference",
    )

    parser.add_argument(
        "--epochs", type=int, default=150, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=128,
        help="Patch size for training images",
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=15,
        help="Number of warmup epochs for learning rate scheduler",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for DataLoader",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Specify GPU devices to use',
    )
    parser.add_argument(
        "--promptir_dim",
        type=int,
        default=48,
        help="Base dimension for PromptIR model",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Use Automatic Mixed Precision (AMP) for training if GPU is used",
    )
    parser.add_argument(
        "--val_split_ratio",
        type=float,
        default=0.1,
        help="Ratio of training data to use for validation",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--log_image_every_n_epochs",
        type=int,
        default=10,
        help="Log sample validation images to TensorBoard every N epochs",
    )
    parser.add_argument(
        "--num_val_images_to_log",
        type=int,
        default=4,
        help="Number of validation images to log from a batch to TensorBoard",
    )
    parser.add_argument(
        "--visualize_inference_output_dir",
        type=str,
        default=None,
        help="Directory to save visualized PNGs from inference .npz",
    )

    # --- New arguments for experimentation ---
    parser.add_argument(
        "--use_prompts",
        action="store_true",
        default=True,
        help="Enable prompt generation blocks in PromptIR decoder",
    )
    parser.add_argument(
        "--no_prompts",
        action="store_false",
        dest="use_prompts",
        help="Disable prompt generation blocks.",
    )
    parser.add_argument(
        "--loss_function",
        type=str,
        default="l1",
        choices=["l1", "mse"],
        help="Loss function to use (l1 or mse).",
    )
    parser.add_argument(
        "--promptir_num_blocks",
        type=str,
        default="4,6,6,8",
        help='Comma-separated list of integers',
    )
    parser.add_argument(
        "--promptir_heads",
        type=str,
        default="1,2,4,8",
        help='Comma-separated list of integers',
    )
    parser.add_argument(
        "--promptir_ffn_expansion_factor",
        type=float,
        default=2.66,
        help="FFN expansion factor for Transformer blocks.",
    )
    parser.add_argument(
        "--prompt_len",
        type=int,
        default=5,
        help="Length of the prompt in PromptGenBlock.",
    )

    args = parser.parse_args()

    # Convert comma-separated string arguments to lists of ints
    try:
        args.promptir_num_blocks = [
            int(x) for x in args.promptir_num_blocks.split(",")
        ]
        if len(args.promptir_num_blocks) != 4:
            raise ValueError("promptir_num_blocks must have 4 values.")
    except ValueError as e:
        parser.error(
            f"Invalid format for --promptir_num_blocks: {e}"
        )

    try:
        args.promptir_heads = [int(x) for x in args.promptir_heads.split(",")]
        if len(args.promptir_heads) != 4:
            raise ValueError("promptir_heads must have 4 values.")
    except ValueError as e:
        parser.error(
            f"Invalid format for --promptir_heads: {e}"
        )

    pl.seed_everything(args.seed, workers=True)

    if args.mode == "train":
        train_model(args)
    elif args.mode == "infer":
        if not args.checkpoint_path:
            raise ValueError(
                "Checkpoint path must be provided for inference mode."
            )
        run_inference(args)
    else:
        print("Invalid mode. Choose 'train' or 'infer'.")
