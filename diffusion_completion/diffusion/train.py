#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Script for Conditional DDPM on 2D LiDAR Completion

Trains a ConditionalDDPMUNet to predict noise in missing regions.
Uses masked MSE loss (only missing region noise is penalized).
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import ConditionalDDPMUNet
from noise_scheduler import NoiseScheduler
from dataset import create_dataloaders
from metrics import compute_metrics


def masked_mse_loss(
    epsilon_pred: torch.Tensor,
    epsilon_true: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    MSE loss computed only on missing regions.

    Args:
        epsilon_pred: (B, 360) predicted noise
        epsilon_true: (B, 360) true noise
        mask: (B, 360) observation mask (1=observed, 0=missing)

    Returns:
        Scalar loss
    """
    missing = 1.0 - mask
    loss = ((epsilon_pred - epsilon_true) ** 2 * missing).sum() / (missing.sum() + 1e-8)
    return loss


def validate_sampling(model, scheduler, val_loader, device, max_batches=10):
    """
    Full sampling validation: run reverse diffusion and compute MAE on missing regions.

    Args:
        model: ConditionalDDPMUNet
        scheduler: NoiseScheduler
        val_loader: Validation DataLoader
        device: Torch device
        max_batches: Maximum number of batches to evaluate

    Returns:
        Average MAE on missing regions
    """
    model.eval()
    all_mae_missing = []

    with torch.no_grad():
        for batch_idx, (partial, mask, target) in enumerate(val_loader):
            if batch_idx >= max_batches:
                break

            partial = partial.to(device)
            mask = mask.to(device)
            target = target.to(device)

            # Initialize: observed = partial, missing = random noise
            x_t = mask * partial + (1.0 - mask) * torch.randn_like(partial)

            # Reverse diffusion loop
            for t_val in reversed(range(scheduler.T)):
                t_batch = torch.full(
                    (partial.shape[0],), t_val, device=device, dtype=torch.long
                )
                epsilon_pred = model(x_t, partial, mask, t_batch)
                x_t = scheduler.reverse_step(epsilon_pred, x_t, t_batch, mask, partial)

            # Final prediction
            pred = x_t.clamp(0.0, 2.0)
            pred = mask * partial + (1.0 - mask) * pred

            # Compute per-sample MAE on missing regions
            pred_np = pred.cpu().numpy()
            target_np = target.cpu().numpy()
            mask_np = mask.cpu().numpy()

            for i in range(pred_np.shape[0]):
                m = compute_metrics(pred_np[i], target_np[i], mask_np[i])
                all_mae_missing.append(m['mae_missing'])

    return float(np.mean(all_mae_missing)) if all_mae_missing else float('inf')


def train(args):
    """Main training function."""
    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Experiment directory
    experiment_dir = Path(args.output_dir) / args.experiment_name
    checkpoint_dir = Path(args.model_save_path) if args.model_save_path else experiment_dir / "checkpoints"
    log_dir = experiment_dir / "logs"

    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config['timestamp'] = datetime.now().isoformat()
    with open(experiment_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Data
    print("\n" + "=" * 70)
    print("Loading Data")
    print("=" * 70)

    train_loader, val_loader = create_dataloaders(
        train_dirs=args.train_dirs,
        val_dirs=args.val_dirs,
        batch_size=args.batch_size,
        num_workers=4,
        normalize=True,
        augment_train=True,
        n_variations=5,
        fill_mode="zero",
    )

    # Model
    print("\n" + "=" * 70)
    print("Creating Model")
    print("=" * 70)

    model = ConditionalDDPMUNet(
        input_length=360,
        in_channels=3,
        model_channels=64,
        channel_mult=(1, 2, 4),
        time_dim=128,
        num_groups=8,
    )
    model = model.to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Noise scheduler
    scheduler = NoiseScheduler(T=args.T, device=args.device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # TensorBoard
    writer = SummaryWriter(log_dir=str(log_dir))

    # History
    history = {
        "train_loss": [],
        "val_noise_loss": [],
        "val_mae_missing_sampling": [],
        "best_val_loss": float("inf"),
        "best_epoch": 0,
    }

    # Resume
    start_epoch = 1
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        if "history" in ckpt:
            history = ckpt["history"]
        print(f"Resuming from epoch {start_epoch}")

    # Training info
    print("\n" + "=" * 70)
    print("Training Configuration")
    print("=" * 70)
    print(f"Experiment: {args.experiment_name}")
    print(f"Epochs: {args.epochs} (starting from {start_epoch})")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Diffusion steps T: {args.T}")
    print(f"Device: {args.device}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Output: {experiment_dir}")

    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70 + "\n")

    for epoch in range(start_epoch, args.epochs + 1):
        # === Train ===
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for batch_idx, (partial, mask, target) in enumerate(pbar):
            partial = partial.to(args.device)
            mask = mask.to(args.device)
            target = target.to(args.device)

            # Sample random timesteps
            t = torch.randint(0, args.T, (partial.shape[0],), device=args.device)

            # Forward diffusion
            x_t, noise = scheduler.forward_diffusion(target, t, mask, partial)

            # Predict noise
            epsilon_pred = model(x_t, partial, mask, t)

            # Masked MSE loss
            loss = masked_mse_loss(epsilon_pred, noise, mask)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

            global_step = (epoch - 1) * len(train_loader) + batch_idx
            writer.add_scalar("Train/BatchLoss", loss.item(), global_step)

        avg_train_loss = total_loss / len(train_loader)

        # === Validate (noise prediction loss) ===
        model.eval()
        val_loss_total = 0.0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]")
            for partial, mask, target in pbar:
                partial = partial.to(args.device)
                mask = mask.to(args.device)
                target = target.to(args.device)

                t = torch.randint(0, args.T, (partial.shape[0],), device=args.device)
                x_t, noise = scheduler.forward_diffusion(target, t, mask, partial)
                epsilon_pred = model(x_t, partial, mask, t)
                loss = masked_mse_loss(epsilon_pred, noise, mask)
                val_loss_total += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        avg_val_loss = val_loss_total / len(val_loader)

        # === Full sampling evaluation (periodic) ===
        val_mae_sampling = None
        if epoch % args.val_sample_interval == 0 or epoch == args.epochs:
            print(f"  Running full sampling evaluation...")
            val_mae_sampling = validate_sampling(
                model, scheduler, val_loader, args.device, max_batches=10
            )
            history["val_mae_missing_sampling"].append(
                {"epoch": epoch, "mae_missing": val_mae_sampling}
            )
            writer.add_scalar("Val/MAE_Missing_Sampling", val_mae_sampling, epoch)

        # === Logging ===
        history["train_loss"].append(avg_train_loss)
        history["val_noise_loss"].append(avg_val_loss)
        writer.add_scalar("Train/EpochLoss", avg_train_loss, epoch)
        writer.add_scalar("Val/NoiseLoss", avg_val_loss, epoch)

        is_best = avg_val_loss < history["best_val_loss"]
        if is_best:
            history["best_val_loss"] = avg_val_loss
            history["best_epoch"] = epoch

        # Print summary
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Noise Loss: {avg_val_loss:.6f}")
        if val_mae_sampling is not None:
            print(f"  Val MAE Missing (sampling): {val_mae_sampling:.6f}")
        if is_best:
            print(f"  ** Best model so far! **")
        print(f"  Best Val Loss: {history['best_val_loss']:.6f} (Epoch {history['best_epoch']})")

        # === Save checkpoints ===
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": config,
        }

        if is_best:
            torch.save(checkpoint, checkpoint_dir / "best_model.pth")
            print(f"  Saved best model: {checkpoint_dir / 'best_model.pth'}")

        if epoch % args.save_every == 0:
            torch.save(checkpoint, checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pth")

    # Final save
    print("\n" + "=" * 70)
    print("Training Completed!")
    print("=" * 70)
    print(f"Best Val Loss: {history['best_val_loss']:.6f} (Epoch {history['best_epoch']})")

    with open(experiment_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    writer.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train Conditional DDPM for 2D LiDAR Point Completion"
    )

    parser.add_argument("--train_dirs", type=str, nargs="+", required=True,
                        help="Training data directories")
    parser.add_argument("--val_dirs", type=str, nargs="+", required=True,
                        help="Validation data directories")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--T", type=int, default=100,
                        help="Number of diffusion timesteps")
    parser.add_argument("--experiment_name", type=str, required=True,
                        help="Experiment name (used for output directory)")
    parser.add_argument("--output_dir", type=str, default="./experiments",
                        help="Root directory for experiment outputs")
    parser.add_argument("--model_save_path", type=str, default=None,
                        help="Checkpoint save directory (default: output_dir/experiment_name/checkpoints/)")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--val_sample_interval", type=int, default=10,
                        help="Run full sampling evaluation every N epochs")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Compute device")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint path to resume training from")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
