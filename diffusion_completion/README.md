# 2D LiDAR Point Completion — 4-Model Comparison

Comparative study of four deep learning models for 2D LiDAR point cloud completion.
Missing scan points are predicted from partial observations (360-dimensional range vectors).

## Models

| Model | Directory | Description |
|-------|-----------|-------------|
| **Conv(zero)** | `baseline/` | 1D Conv U-Net; missing regions filled with zeros |
| **Conv(noise)** | `baseline/` | 1D Conv U-Net; missing regions filled with Gaussian noise |
| **Diffusion v1** | `diffusion/` | Conditional DDPM; masked diffusion with Conv U-Net |
| **Diffusion v2** | `diffusion_v2/` | Dual Encoder DDPM with FiLM time conditioning |

## Repository Structure

```
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
├── diffusion_v2/           # Diffusion v2 (Dual Encoder + FiLM)
│   ├── model.py            # DualEncoderDDPMUNet
│   ├── train.py
│   └── sample.py
├── generate_dataset.py     # Synthetic 2D LiDAR dataset generator
├── statistical_test.py     # Paired t-test, Wilcoxon, Cohen's d
├── run_experiments.sh      # Full pipeline (train → eval → stats)
└── requirements.txt        # (in parent directory)
```

## Dependencies

Install from the project root:

```bash
pip install -r ../requirements.txt
```

Key packages: `torch`, `numpy`, `scipy`, `matplotlib`.

## Dataset Generation

Generate synthetic 2D LiDAR scan data (partial + complete pairs):

```bash
python generate_dataset.py
```

Output directories: `output/` (train/val) and `output_validation/` (additional validation set).

## Training

### Conv(zero) / Conv(noise)

```bash
cd baseline

python train.py \
  --train_dirs ../output/train ../output_validation/train \
  --val_dirs ../output/val ../output_validation/val \
  --fill_mode zero \          # or: noise
  --experiment_name conv_zero_seed42 \
  --output_dir ./experiments \
  --epochs 100 --batch_size 64 --lr 1e-3 --seed 42 --device cuda
```

### Diffusion v1

```bash
cd diffusion

python train.py \
  --train_dirs ../output/train ../output_validation/train \
  --val_dirs ../output/val ../output_validation/val \
  --experiment_name diffusion_seed42 \
  --output_dir ./experiments \
  --epochs 100 --batch_size 64 --lr 2e-4 --T 100 --seed 42 --device cuda
```

### Diffusion v2 (Dual Encoder + FiLM)

```bash
cd diffusion_v2

python train.py \
  --train_dirs ../output/train ../output_validation/train \
  --val_dirs ../output/val ../output_validation/val \
  --experiment_name diffusion_v2_seed42 \
  --output_dir ./experiments \
  --epochs 100 --batch_size 64 --lr 2e-4 --T 100 \
  --base_channels 64 --num_res_blocks 2 \
  --seed 42 --device cuda
```

## Evaluation

### Conv baseline

```bash
cd baseline
python evaluate.py \
  --checkpoint ./experiments/conv_zero_seed42/checkpoints/best_model.pth \
  --data_dirs ../output/val ../output_validation/val \
  --fill_mode zero --output_dir ../results/eval/conv_zero_seed42 \
  --save_per_sample --device cuda
```

### Diffusion v1

```bash
cd diffusion
python sample.py \
  --checkpoint ./experiments/diffusion_seed42/checkpoints/best_model.pth \
  --data_dirs ../output/val ../output_validation/val \
  --output_dir ../results/eval/diffusion_seed42 \
  --T 100 --save_per_sample --device cuda
```

### Diffusion v2

```bash
cd diffusion_v2
python sample.py \
  --checkpoint ./experiments/diffusion_v2_seed42/checkpoints/best_model.pth \
  --data_dirs ../output/val ../output_validation/val \
  --output_dir ../results/eval/diffusion_v2_seed42 \
  --T 100 --base_channels 64 --num_res_blocks 2 \
  --save_per_sample --device cuda
```

## Statistical Tests

Run after evaluation to compute paired t-test, Wilcoxon signed-rank test, and Cohen's d
across all model pairs and seeds (42, 43, 44):

```bash
python statistical_test.py \
  --zero_fill_dirs  results/eval/conv_zero_seed42  results/eval/conv_zero_seed43  results/eval/conv_zero_seed44 \
  --noise_fill_dirs results/eval/conv_noise_seed42 results/eval/conv_noise_seed43 results/eval/conv_noise_seed44 \
  --diffusion_dirs  results/eval/diffusion_seed42  results/eval/diffusion_seed43  results/eval/diffusion_seed44 \
  --diffusion_v2_dirs results/eval/diffusion_v2_seed42 results/eval/diffusion_v2_seed43 results/eval/diffusion_v2_seed44 \
  --output_dir results/statistical_tests \
  --seeds 42 43 44
```

Results saved to `results/statistical_tests/statistical_results.json`.

## Full Experiment Pipeline

Run all 12 training runs (4 models × 3 seeds), 12 evaluation runs, and statistical tests:

```bash
bash run_experiments.sh --device cuda --epochs 100
```

Options:

| Option | Description |
|--------|-------------|
| `--device DEVICE` | Compute device (default: `cuda`) |
| `--epochs N` | Training epochs (default: `100`) |
| `--skip-training` | Skip Phase 1, use existing checkpoints |
| `--skip-eval` | Skip Phase 2, use existing eval results |

## Diffusion v2 Architecture

```
obs_encoder(partial, mask):   2ch → ch → 2ch → 4ch  (no skips to decoder)
noisy_encoder(x_t, mask):     2ch → ch → 2ch → 4ch  (skips to decoder)

Bottleneck: cat(obs_bottom, noisy_bottom) = 8ch → FiLMResBlocks → 4ch

Decoder:
  Level 2: up(4ch) + noisy_skip(4ch) = 8ch → 4ch
  Level 1: up(4ch) + noisy_skip(2ch) = 6ch → 2ch
  Level 0: up(2ch) + noisy_skip(ch)  = 3ch → ch

Output: GN → SiLU → Conv1d(ch→1) → (B, 360)
```

FiLMResBlock applies per-timestep scale/shift to each residual block:
`Conv → GN → FiLM(t) → SiLU → Conv → GN + skip`
