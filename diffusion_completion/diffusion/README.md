# Conditional DDPM for 2D LiDAR Completion

Masked diffusion approach for 2D LiDAR point completion. Noise is only applied to missing regions while observed regions are fixed as conditioning.

## Architecture

- **Model**: 1D Conv U-Net with sinusoidal time embedding (~4-5M params)
- **Scheduler**: Linear beta schedule with T=100 steps
- **Loss**: Masked MSE on predicted noise (missing regions only)
- **Sampling**: Full reverse diffusion with observed region enforcement at each step

## Training

```bash
cd /workspaces/toyot/diffusion_completion/diffusion

python train.py \
  --train_dirs ../output/train ../output_validation/train \
  --val_dirs ../output/val ../output_validation/val \
  --epochs 100 \
  --batch_size 32 \
  --lr 2e-4 \
  --T 100 \
  --experiment_name ddpm_v1 \
  --val_sample_interval 10
```

### Resume training

```bash
python train.py \
  --train_dirs ../output/train ../output_validation/train \
  --val_dirs ../output/val ../output_validation/val \
  --epochs 200 \
  --experiment_name ddpm_v1 \
  --resume ./experiments/ddpm_v1/checkpoints/best_model.pth
```

## Inference & Evaluation

```bash
python sample.py \
  --checkpoint ./experiments/ddpm_v1/checkpoints/best_model.pth \
  --data_dirs ../output/val \
  --output_dir ./results \
  --T 100 \
  --visualize \
  --n_vis_samples 10 \
  --plot_type linear
```

## Comparison with Baseline

The `DiffusionModelWrapper` class provides the same `forward(partial, mask)` interface as the baseline `LiDARCompletionModel`. This enables direct comparison using the baseline's `ModelEvaluator` and `visualize_samples`:

```python
from diffusion import ConditionalDDPMUNet, NoiseScheduler, DiffusionModelWrapper

model = ConditionalDDPMUNet()
scheduler = NoiseScheduler(T=100, device="cuda")

# Load checkpoint
ckpt = torch.load("path/to/best_model.pth")
model.load_state_dict(ckpt["model_state_dict"])
model = model.to("cuda")

# Wrap for baseline-compatible interface
wrapper = DiffusionModelWrapper(model, scheduler, device="cuda")
pred = wrapper(partial, mask)  # Same as baseline model
```

## Key CLI Arguments

### train.py

| Argument | Description | Default |
|----------|-------------|---------|
| `--train_dirs` | Training data directories | Required |
| `--val_dirs` | Validation data directories | Required |
| `--epochs` | Number of epochs | 100 |
| `--batch_size` | Batch size | 32 |
| `--lr` | Learning rate | 2e-4 |
| `--T` | Diffusion timesteps | 100 |
| `--experiment_name` | Experiment name | Required |
| `--output_dir` | Output root directory | ./experiments |
| `--save_every` | Checkpoint save interval | 10 |
| `--val_sample_interval` | Full sampling eval interval | 10 |
| `--device` | Compute device | cuda |
| `--seed` | Random seed | 42 |
| `--resume` | Resume checkpoint path | None |

### sample.py

| Argument | Description | Default |
|----------|-------------|---------|
| `--checkpoint` | Model checkpoint path | Required |
| `--data_dirs` | Evaluation data directories | Required |
| `--output_dir` | Output directory | ./results |
| `--T` | Diffusion timesteps | 100 |
| `--batch_size` | Batch size | 32 |
| `--visualize` | Generate plots | False |
| `--n_vis_samples` | Number of visualizations | 5 |
| `--plot_type` | "linear" or "polar" | linear |
