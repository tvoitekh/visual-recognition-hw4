# NYCU Visual Recognition using Deep Learning 2025 Spring HW4

StudentID: 111550203  
Name: 提姆西

## Introduction

This project implements an image restoration system using the **PromptIR architecture**. The system is designed to tackle the challenge of removing two types of degradations, **rain and snow**, from images to reconstruct their clean counterparts. This is approached as a "blind" restoration problem, meaning the model is not explicitly informed about the type of degradation present in a given test image.

The core approach utilizes the PromptIR model, which features a **U-Net-like structure with Transformer blocks** and a novel **prompting mechanism**. Learnable "prompts"—sets of tunable parameters—encode discriminative information about various image degradations. These prompts are dynamically combined based on the input image features and injected into the decoder to guide the restoration process. This work also explores the impact of the prompting mechanism, different loss functions, and architectural depth on model performance, measured primarily by Peak Signal-to-Noise Ratio (PSNR).

## How to install and Run

1.  Clone this repository:
    ```bash
    git clone [https://github.com/tvoitekh/visual-recognition-hw4.git](https://github.com/tvoitekh/visual-recognition-hw4.git)
    cd visual-recognition-hw4
    ```

2.  Make sure you have the data directory structure (default path is `./hw4_realse_dataset/`):
    ```
    ./hw4_realse_dataset/
    ├── train/
    │   ├── degraded/
    │   │   ├── rain-1.png
    │   │   ...
    │   │   └── snow-1600.png
    │   └── clean/
    │       ├── rain_clean-1.png
    │       ...
    │       └── snow_clean-1600.png
    └── test/
        └── degraded/
            ├── 0.png
            ...
            └── 99.png
    ```

3.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

4.  Run the model (`hw4_experiment.py`):

    * **For training the Final Model (Recommended Configuration):**
        This configuration uses Prompts, MSE Loss, Batch Size 8, and trains for 150 Epochs.
        ```bash
        python hw4_experiment.py --mode train --data_dir ./hw4_realse_dataset \
            --checkpoint_dir ./checkpoints/FINAL_MODEL_PROMPTS_MSE_BATCH8 \
            --epochs 150 --batch_size 8 --patch_size 128 --lr 2e-4 \
            --promptir_dim 48 --use_prompts --loss_function mse \
            --promptir_num_blocks "4,6,6,8" --promptir_heads "1,2,4,8" \
            --promptir_ffn_expansion_factor 2.66 --prompt_len 5 \
            --device 0 --use_amp --seed 42
        ```

    * **For inference using the trained Final Model:**
        Replace `best_model-epoch=XX-val_psnr=YY.YY.ckpt` with the actual best checkpoint file name from the `FINAL_MODEL_PROMPTS_MSE_BATCH8` directory.
        ```bash
        python hw4_experiment.py --mode infer \
            --data_dir ./hw4_realse_dataset \
            --checkpoint_path ./checkpoints/FINAL_MODEL_PROMPTS_MSE_BATCH8/best_model-epoch=XX-val_psnr=YY.YY.ckpt \
            --output_npz_path ./pred.npz \
            --device 0 \
            --promptir_dim 48
        ```
        To visualize inference outputs (optional):
        ```bash
        python hw4_experiment.py --mode infer \
            --data_dir ./hw4_realse_dataset \
            --checkpoint_path ./checkpoints/FINAL_MODEL_PROMPTS_MSE_BATCH8/best_model-epoch=XX-val_psnr=YY.YY.ckpt \
            --output_npz_path ./pred.npz \
            --device 0 \
            --promptir_dim 48 \
            --visualize_inference_output_dir ./inference_visualizations
        ```

    * **Experimental Runs (as detailed in the report):**
        * Run 1: Baseline (Prompts enabled, L1 Loss, Default Depth, 50 Epochs)
            ```bash
            python hw4_experiment.py --mode train --data_dir ./hw4_realse_dataset \
                --checkpoint_dir ./checkpoints/RUN1_BASELINE_PROMPTS_L1_DEFAULT \
                --epochs 50 --batch_size 4 --patch_size 128 --lr 2e-4 \
                --promptir_dim 48 --use_prompts --loss_function l1 \
                --promptir_num_blocks "4,6,6,8" --promptir_heads "1,2,4,8" \
                --promptir_ffn_expansion_factor 2.66 --prompt_len 5 \
                --device 0 --use_amp --seed 42
            ```
        * Run 2: No Prompts (L1 Loss, Default Depth, 50 Epochs)
            ```bash
            python hw4_experiment.py --mode train --data_dir ./hw4_realse_dataset \
                --checkpoint_dir ./checkpoints/RUN2_NO_PROMPTS_L1_DEFAULT \
                --epochs 50 --batch_size 4 --patch_size 128 --lr 2e-4 \
                --promptir_dim 48 --no_prompts --loss_function l1 \
                --promptir_num_blocks "4,6,6,8" --promptir_heads "1,2,4,8" \
                --promptir_ffn_expansion_factor 2.66 --prompt_len 5 \
                --device 0 --use_amp --seed 42
            ```
        * Run 3: MSE Loss (Prompts enabled, Default Depth, 50 Epochs)
            ```bash
            python hw4_experiment.py --mode train --data_dir ./hw4_realse_dataset \
                --checkpoint_dir ./checkpoints/RUN3_PROMPTS_MSE_DEFAULT \
                --epochs 50 --batch_size 4 --patch_size 128 --lr 2e-4 \
                --promptir_dim 48 --use_prompts --loss_function mse \
                --promptir_num_blocks "4,6,6,8" --promptir_heads "1,2,4,8" \
                --promptir_ffn_expansion_factor 2.66 --prompt_len 5 \
                --device 0 --use_amp --seed 42
            ```
        * Run 4: Shallow Depth (Prompts enabled, L1 Loss, 50 Epochs)
            ```bash
            python hw4_experiment.py --mode train --data_dir ./hw4_realse_dataset \
                --checkpoint_dir ./checkpoints/RUN4_PROMPTS_L1_SHALLOW \
                --epochs 50 --batch_size 4 --patch_size 128 --lr 2e-4 \
                --promptir_dim 48 --use_prompts --loss_function l1 \
                --promptir_num_blocks "2,3,3,4" --promptir_heads "1,2,4,8" \
                --promptir_ffn_expansion_factor 2.66 --prompt_len 5 \
                --device 0 --use_amp --seed 42
            ```

## Performance snapshot

<img width="1153" alt="image" src="https://github.com/user-attachments/assets/9e06c169-4a8e-40c9-8d2a-0821be0acf44" />

The final model configuration achieves a validation **PSNR of approximately 28.32 dB** on the image restoration task.

Key performance features:
* Based on **PromptIR architecture** with a U-Net like structure and Transformer blocks.
* Utilizes a **prompting mechanism** for adaptive restoration. For the specific rain/snow task, its distinct benefit was modest compared to the robust base architecture.
* **MSE Loss function** was found to be more effective than L1 Loss for this specific dataset and degradation types.
* The default architectural depth (number of blocks: `[4,6,6,8]`) provided the best performance among the tested configurations.
* Trained for **150 epochs** with a batch size of 8, using AdamW optimizer and a cosine annealing learning rate schedule with warmup.
* **Automatic Mixed Precision (AMP)** was used for faster training on GPU.

## Code Linting

The following commands have been run as well as manual modifications performed:

```bash
autopep8 --in-place --aggressive --max-line-length 79 hw4_experiment.py
```

```bash
black --line-length 79 hw4_experiment.py
```

<img width="722" alt="image" src="https://github.com/user-attachments/assets/a47ec4fa-6b74-4196-bdc3-5ba21130e41f" />

As can be seen no warnings or errors are present. This verifies that the code had been successfully linted as required.
