2D LiDAR Point Completion — Comparative Study of Conv and Diffusion Models

This repository contains a comparative study of deep learning models for 2D LiDAR point cloud completion.
Given partial LiDAR scans represented as 360-dimensional range vectors with missing regions, the task is to reconstruct the complete scan.

The project investigates:

How far simple convolutional baselines can go for structured 1D geometric completion.

Whether and how diffusion models can be adapted to conditional completion of sparse LiDAR observations.

Which architectural choices actually matter in practice, validated by controlled ablations and statistical tests.

Research Motivation and Scope

2D LiDAR completion is a structured 1D inpainting problem with strong geometric priors (continuity of walls, smooth surfaces, sharp corners).
While diffusion models have shown strong performance in image generation and completion, their effectiveness on structured low-dimensional sensor signals (e.g., LiDAR scans) is unclear.

This project aims to answer the following research questions:

RQ1: How strong is a simple Conv U-Net baseline for 2D LiDAR completion?

RQ2: Does replacing zero-filled missing regions with random noise improve Conv baselines?

RQ3: Can conditional diffusion models outperform Conv baselines on this task?

RQ4: Which design choices (conditioning strategy, time embedding, residual blocks) are necessary for diffusion models to become competitive?

To ensure fair comparison, all models share:

The same dataset

The same input representation (partial + mask)

The same evaluation protocol and statistical tests

Models
Model	Directory	Description
Conv(zero)	baseline/	1D Conv U-Net; missing regions filled with zeros
Conv(noise)	baseline/	1D Conv U-Net; missing regions filled with Gaussian noise
Diffusion v1	diffusion/	Conditional DDPM; masked diffusion with Conv U-Net
Diffusion v2	diffusion_v2/	Dual-Encoder DDPM with FiLM time conditioning and residual blocks
Repository Structure
diffusion_completion/
├── baseline/               # Conv(zero) and Conv(noise) models
│   ├── model.py            # LiDARCompletionModel (1D Conv U-Net)
│   ├── train.py
│   ├── evaluate.py
│   ├── dataset.py
│   ├── metrics.py
│   └── visualize.py
├── diffusion/              # Diffusion v1 (Conditional DDPM)
│   ├── model.py            # ConditionalDDPMUNet (~4-5M params)
│   ├── noise_scheduler.py  # Linear beta schedule, T=100
│   ├── train.py
│   ├── sample.py
│   ├── dataset.py
│   ├── metrics.py
│   └── visualize.py
├── diffusion_v2/           # Diffusion v2 (Dual Encoder + FiLM + ResBlocks)
│   ├── model.py            # DualEncoderDDPMUNet
│   ├── train.py
│   └── sample.py
├── generate_dataset.py     # Synthetic 2D LiDAR dataset generator
├── statistical_test.py     # Paired t-test, Wilcoxon, Cohen's d
├── run_experiments.sh      # Full pipeline (train → eval → stats)
└── requirements.txt        # (in parent directory)

Dataset Generation

Synthetic 2D LiDAR scans are generated as pairs of:

partial: corrupted scan with missing regions

mask: binary mask (1 = observed, 0 = missing)

target: complete scan

python generate_dataset.py


Outputs:

output/train, output/val

output_validation/train, output_validation/val

Experimental Protocol

Each model is trained with three random seeds (42, 43, 44).
All experiments follow the same pipeline:

Train model

Evaluate on validation sets

Save per-sample predictions

Perform paired statistical tests

This ensures that observed differences are not due to random initialization.

Training
Conv(zero) / Conv(noise)
cd baseline

python train.py \
  --train_dirs ../output/train ../output_validation/train \
  --val_dirs ../output/val ../output_validation/val \
  --fill_mode zero \          # or: noise
  --experiment_name conv_zero_seed42 \
  --output_dir ./experiments \
  --epochs 100 --batch_size 64 --lr 1e-3 --seed 42 --device cuda

Diffusion v1
cd diffusion

python train.py \
  --train_dirs ../output/train ../output_validation/train \
  --val_dirs ../output/val ../output_validation/val \
  --experiment_name diffusion_seed42 \
  --output_dir ./experiments \
  --epochs 100 --batch_size 64 --lr 2e-4 --T 100 --seed 42 --device cuda

Diffusion v2 (Dual Encoder + FiLM)
cd diffusion_v2

python train.py \
  --train_dirs ../output/train ../output_validation/train \
  --val_dirs ../output/val ../output_validation/val \
  --experiment_name diffusion_v2_seed42 \
  --output_dir ./experiments \
  --epochs 100 --batch_size 64 --lr 2e-4 --T 100 \
  --base_channels 64 --num_res_blocks 2 \
  --seed 42 --device cuda

Evaluation and Statistical Testing

After evaluation, paired statistical tests are applied to all model pairs:

Paired t-test

Wilcoxon signed-rank test

Cohen’s d (effect size)

python statistical_test.py \
  --zero_fill_dirs  results/eval/conv_zero_seed42  results/eval/conv_zero_seed43  results/eval/conv_zero_seed44 \
  --noise_fill_dirs results/eval/conv_noise_seed42 results/eval/conv_noise_seed43 results/eval/conv_noise_seed44 \
  --diffusion_dirs  results/eval/diffusion_seed42  results/eval/diffusion_seed43  results/eval/diffusion_seed44 \
  --diffusion_v2_dirs results/eval/diffusion_v2_seed42 results/eval/diffusion_v2_seed43 results/eval/diffusion_v2_seed44 \
  --output_dir results/statistical_tests \
  --seeds 42 43 44

Key Findings (Summary)

Conv(zero) is a surprisingly strong baseline due to strong inductive bias for local continuity.

Conv(noise) slightly improves robustness to missing regions.

Diffusion v1 underperforms Conv baselines, showing that naïve application of diffusion is insufficient.

Diffusion v2 improves stability and conditioning but still struggles to outperform Conv baselines on sharp geometric transitions.

Architectural design (conditioning strategy, residual blocks, FiLM time embedding) is critical for diffusion models in structured sensor domains.

Diffusion v2 Architecture (Overview)
obs_encoder(partial, mask):   2ch → ch → 2ch → 4ch  (no skips to decoder)
noisy_encoder(x_t, mask):     2ch → ch → 2ch → 4ch  (skips to decoder)

Bottleneck: cat(obs_bottom, noisy_bottom) = 8ch → FiLMResBlocks → 4ch

Decoder:
  Level 2: up(4ch) + noisy_skip(4ch) = 8ch → 4ch
  Level 1: up(4ch) + noisy_skip(2ch) = 6ch → 2ch
  Level 0: up(2ch) + noisy_skip(ch)  = 3ch → ch

Output: GN → SiLU → Conv1d(ch→1) → (B, 360)

Future Directions

Structured noise design for LiDAR geometry (local / smooth noise)

Cross-attention between observed and missing regions

Diffusion variants (DDIM) for faster sampling

Hybrid Conv + Diffusion architectures

Incorporation of physical priors (e.g., ray continuity constraints)

License and Usage

This repository is intended for research and experimental purposes.
Feel free to adapt the codebase for your own LiDAR completion or sensor reconstruction tasks.
