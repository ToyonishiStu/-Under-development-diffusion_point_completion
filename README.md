<div align="center">

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│   ██╗     ██╗██████╗  █████╗ ██████╗     ██████╗ ██████╗       │
│   ██║     ██║██╔══██╗██╔══██╗██╔══██╗   ██╔════╝██╔════╝       │
│   ██║     ██║██║  ██║███████║██████╔╝   ██║     ██║            │
│   ██║     ██║██║  ██║██╔══██║██╔══██╗   ██║     ██║            │
│   ███████╗██║██████╔╝██║  ██║██║  ██║   ╚██████╗╚██████╗       │
│   ╚══════╝╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝    ╚═════╝ ╚═════╝      │
│                                                                  │
│        2D LiDAR Point Cloud Completion · Deep Learning          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

<br>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-Research-22C55E?style=flat-square)](#)
[![Status](https://img.shields.io/badge/Status-Active-F59E0B?style=flat-square)](#)

<br>

**Comparative study of Conv and Diffusion models for structured 1D geometric completion.**  
Given partial 360° LiDAR scans with missing regions, reconstruct the complete scan.

<br>

</div>

---

## 🔬 Research Questions

This project rigorously investigates four questions:

| # | Research Question |
|---|---|
| **RQ1** | How strong is a simple Conv U-Net baseline for 2D LiDAR completion? |
| **RQ2** | Does replacing zero-filled missing regions with **random noise** improve Conv baselines? |
| **RQ3** | Can conditional diffusion models **outperform** Conv baselines on this task? |
| **RQ4** | Which design choices (conditioning strategy, time embedding, residual blocks) are **necessary** for diffusion models to become competitive? |

> All models share the same dataset, input representation (partial + mask), evaluation protocol, and statistical tests — ensuring fair comparison.

---

## 🏗️ Models

```
┌──────────────────┬───────────────────────────────────────────────────────────┐
│ Model            │ Description                                               │
├──────────────────┼───────────────────────────────────────────────────────────┤
│ Conv(zero)       │ 1D Conv U-Net · missing regions → filled with zeros       │
│ Conv(noise)      │ 1D Conv U-Net · missing regions → filled with Gaussian ε  │
│ Diffusion v1     │ Conditional DDPM · masked diffusion with Conv U-Net       │
│ Diffusion v2     │ Dual-Encoder DDPM · FiLM time conditioning + ResBlocks    │
└──────────────────┴───────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Structure

```
diffusion_completion/
├── baseline/                  ← Conv(zero) and Conv(noise)
│   ├── model.py               · LiDARCompletionModel (1D Conv U-Net)
│   ├── train.py
│   ├── evaluate.py
│   ├── dataset.py
│   ├── metrics.py
│   └── visualize.py
│
├── diffusion/                 ← Diffusion v1  (Conditional DDPM)
│   ├── model.py               · ConditionalDDPMUNet  (~4–5M params)
│   ├── noise_scheduler.py     · Linear β schedule, T=100
│   ├── train.py
│   ├── sample.py
│   ├── dataset.py
│   ├── metrics.py
│   └── visualize.py
│
├── diffusion_v2/              ← Diffusion v2  (Dual Encoder + FiLM + ResBlocks)
│   ├── model.py               · DualEncoderDDPMUNet
│   ├── train.py
│   └── sample.py
│
├── generate_dataset.py        ← Synthetic 2D LiDAR dataset generator
├── statistical_test.py        ← Paired t-test · Wilcoxon · Cohen's d
├── run_experiments.sh         ← Full pipeline  (train → eval → stats)
└── requirements.txt
```

---

## 🗃️ Dataset Generation

Synthetic 2D LiDAR scans are generated as triplets:

```
partial  →  corrupted scan with missing regions
mask     →  binary mask  (1 = observed,  0 = missing)
target   →  complete ground-truth scan
```

```bash
python generate_dataset.py
```

**Outputs:**
- `output/train`, `output/val`
- `output_validation/train`, `output_validation/val`

---

## 🚀 Training

### Conv(zero) / Conv(noise)

```bash
cd baseline

python train.py \
  --train_dirs ../output/train ../output_validation/train \
  --val_dirs   ../output/val   ../output_validation/val   \
  --fill_mode  zero \                # or: noise
  --experiment_name conv_zero_seed42 \
  --output_dir ./experiments \
  --epochs 100 --batch_size 64 --lr 1e-3 --seed 42 --device cuda
```

### Diffusion v1

```bash
cd diffusion

python train.py \
  --train_dirs ../output/train ../output_validation/train \
  --val_dirs   ../output/val   ../output_validation/val   \
  --experiment_name diffusion_seed42 \
  --output_dir ./experiments \
  --epochs 100 --batch_size 64 --lr 2e-4 --T 100 --seed 42 --device cuda
```

### Diffusion v2 — Dual Encoder + FiLM

```bash
cd diffusion_v2

python train.py \
  --train_dirs ../output/train ../output_validation/train \
  --val_dirs   ../output/val   ../output_validation/val   \
  --experiment_name diffusion_v2_seed42 \
  --output_dir ./experiments \
  --epochs 100 --batch_size 64 --lr 2e-4 --T 100 \
  --base_channels 64 --num_res_blocks 2 \
  --seed 42 --device cuda
```

> ⚠️ Each model is trained with **three random seeds** (42, 43, 44) to ensure statistical reliability.

---

## 📊 Evaluation & Statistical Testing

```bash
python statistical_test.py \
  --zero_fill_dirs      results/eval/conv_zero_seed42  results/eval/conv_zero_seed43  results/eval/conv_zero_seed44  \
  --noise_fill_dirs     results/eval/conv_noise_seed42 results/eval/conv_noise_seed43 results/eval/conv_noise_seed44 \
  --diffusion_dirs      results/eval/diffusion_seed42  results/eval/diffusion_seed43  results/eval/diffusion_seed44  \
  --diffusion_v2_dirs   results/eval/diffusion_v2_seed42 results/eval/diffusion_v2_seed43 results/eval/diffusion_v2_seed44 \
  --output_dir results/statistical_tests \
  --seeds 42 43 44
```

**Applied tests:**

| Test | Purpose |
|------|---------|
| Paired t-test | Mean difference significance |
| Wilcoxon signed-rank | Non-parametric robustness check |
| Cohen's *d* | Effect size estimation |

---

## 🧠 Diffusion v2 Architecture

```
                         ┌─────────────────────────────────────────────┐
                         │              DualEncoderDDPMUNet             │
                         └─────────────────────────────────────────────┘

  INPUT (partial, mask)                    INPUT (x_t, mask)
        │                                        │
        ▼                                        ▼
  obs_encoder                             noisy_encoder
  2ch → ch → 2ch → 4ch                   2ch → ch → 2ch → 4ch
  (no skip connections)                  (skip connections → decoder)
        │                                        │
        └──────────────┬─────────────────────────┘
                       ▼
              Bottleneck: cat(obs, noisy) = 8ch
                       │
              FiLMResBlocks  ← time embedding
                       │
                      4ch
                       │
                  ┌────▼────────────────────────────────────┐
                  │  Decoder  (with noisy_encoder skips)    │
                  │  up(4ch) + skip(4ch) → 4ch              │
                  │  up(4ch) + skip(2ch) → 2ch              │
                  │  up(2ch) + skip(ch)  → ch               │
                  └────────────────────────────────────────┘
                       │
              GN → SiLU → Conv1d(ch→1)
                       │
                       ▼
                  Output: (B, 360)
```

---

## 📈 Key Findings

> Results reflect current experimental status.

- **Conv(zero)** is a surprisingly strong baseline — the 1D inductive bias for local continuity is very effective.
- **Conv(noise)** slightly improves robustness to missing regions compared to zero-fill.
- **Diffusion v1** underperforms Conv baselines — naïve conditional DDPM is insufficient for this structured signal.
- **Diffusion v2** improves stability and conditioning quality, but still struggles on **sharp geometric transitions** (corners, doorways).

---

## 🔭 Future Directions

- [ ] **Structured noise design** — geometry-aware local/smooth noise tailored to LiDAR
- [ ] **Cross-attention** between observed and missing regions
- [ ] **Faster sampling** via DDIM or flow matching
- [ ] **Hybrid Conv + Diffusion** architectures
- [ ] **Geometric priors** — ray continuity constraints, wall regularity

---

## 📄 License

This repository is intended for **research and experimental purposes**.  
Feel free to adapt the codebase for LiDAR completion or related sensor reconstruction tasks.

---

<div align="center">

*2D LiDAR Point Cloud Completion — Bridging structured sensor completion and generative modeling.*

</div>
