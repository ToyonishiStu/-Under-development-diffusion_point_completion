# Diffusion v2: Dual Encoder U-Net with FiLM Conditioning

## Overview

Diffusion v2 improves on the baseline DDPM by separating observation and noisy-input encoder paths and using FiLM (Feature-wise Linear Modulation) for time conditioning.

**Key differences from v1:**
- **Dual Encoder**: `obs_encoder` processes `(partial, mask)`, `noisy_encoder` processes `(x_t, mask)` independently
- **FiLMResBlock**: `Conv -> GN -> FiLM(t) -> SiLU -> Conv -> GN + skip` — scale/shift conditioning per block
- **Decoder uses only `noisy_encoder` skips** for spatial detail; `obs_encoder` contributes only via the bottleneck

## Architecture

```
obs_encoder(partial, mask):   2ch -> ch -> 2ch -> 4ch  (no skips to decoder)
noisy_encoder(x_t, mask):     2ch -> ch -> 2ch -> 4ch  (skips to decoder)

Bottleneck: cat(obs_bottom[4ch], noisy_bottom[4ch]) = 8ch -> ResBlocks -> 4ch

Decoder:
  Level 2: up(4ch) + noisy_skip(4ch) = 8ch -> 4ch
  Level 1: up(4ch) + noisy_skip(2ch) = 6ch -> 2ch
  Level 0: up(2ch) + noisy_skip(ch)  = 3ch -> ch

Output: GN -> SiLU -> Conv1d(ch->1) -> squeeze -> (B, 360)
```

## Quick Start

### 1. Verify forward pass

```bash
cd /workspaces/toyot/diffusion_completion/diffusion_v2
python -c "
import sys, os
sys.path.append(os.path.join('..', 'diffusion'))
import torch
from model import DualEncoderDDPMUNet
m = DualEncoderDDPMUNet(base_channels=64, num_res_blocks=2)
out = m(torch.randn(4,360), torch.randn(4,360), torch.ones(4,360), torch.randint(0,100,(4,)))
assert out.shape == (4, 360)
print('OK:', out.shape, 'params:', sum(p.numel() for p in m.parameters()))
"
```

### 2. Smoke test (1 epoch, CPU)

```bash
python train.py \
  --train_dirs ../output/train --val_dirs ../output/val \
  --experiment_name smoke_test --output_dir /tmp/v2_test \
  --epochs 1 --batch_size 8 --T 10 \
  --base_channels 32 --num_res_blocks 1 --device cpu
```

### 3. Inference smoke test

```bash
python sample.py \
  --checkpoint /tmp/v2_test/smoke_test/checkpoints/best_model.pth \
  --data_dirs ../output/val --output_dir /tmp/v2_eval \
  --T 10 --base_channels 32 --num_res_blocks 1 --device cpu --save_per_sample
```

### 4. Full training (single seed)

```bash
python train.py \
  --train_dirs ../output/train ../output_validation/train \
  --val_dirs ../output/val ../output_validation/val \
  --experiment_name diffusion_v2_seed42 \
  --output_dir ./experiments \
  --epochs 100 --batch_size 64 --lr 2e-4 --T 100 \
  --base_channels 64 --num_res_blocks 2 \
  --seed 42 --device cuda
```

### 5. Full pipeline (all seeds)

```bash
bash ../run_experiments.sh --device cuda --epochs 100
```

## Output Files

- `experiments/<name>/config.json` — training configuration
- `experiments/<name>/checkpoints/best_model.pth` — best checkpoint
- `experiments/<name>/training_history.json` — loss history
- `results/eval/<name>/evaluation_diffusion_v2.json` — summary metrics
- `results/eval/<name>/per_sample_diffusion_v2.json` — per-sample metrics (with `--save_per_sample`)

## Shared Modules

`train.py` and `sample.py` import from `../diffusion/` via `sys.path`:
- `noise_scheduler.py` — DDPM forward/reverse process
- `dataset.py` — `LiDAR2DCompletionDataset`, `create_dataloaders`
- `metrics.py` — `compute_metrics`, `aggregate_metrics`
- `visualize.py` — `plot_linear_comparison`, `plot_comparison`
