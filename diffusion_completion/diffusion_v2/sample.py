#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference & Evaluation Script for Diffusion v2 (Dual Encoder + FiLM)

Identical to diffusion/sample.py except:
- Uses DualEncoderDDPMUNet instead of ConditionalDDPMUNet
- Adds --base_channels and --num_res_blocks CLI arguments
- Saves results as per_sample_diffusion_v2.json and evaluation_diffusion_v2.json
"""

import os
import sys

# Allow importing shared modules from diffusion/ (append to keep diffusion_v2/ first)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'diffusion'))

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import DualEncoderDDPMUNet
from noise_scheduler import NoiseScheduler
from dataset import LiDAR2DCompletionDataset
from metrics import compute_metrics, aggregate_metrics
from visualize import plot_linear_comparison, plot_comparison


class DiffusionSampler:
    """
    Runs reverse diffusion sampling for conditional completion.

    Args:
        model: DualEncoderDDPMUNet
        scheduler: NoiseScheduler
        device: Torch device
    """

    def __init__(
        self,
        model: DualEncoderDDPMUNet,
        scheduler: NoiseScheduler,
        device: str = "cuda"
    ):
        self.model = model
        self.scheduler = scheduler
        self.device = device

    @torch.no_grad()
    def sample(self, partial: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Run full reverse diffusion to complete missing regions.

        Args:
            partial: (B, 360) observed partial data
            mask: (B, 360) observation mask (1=observed, 0=missing)

        Returns:
            completed: (B, 360) completed data
        """
        self.model.eval()

        # Initialize: observed = partial, missing = random noise
        x_t = mask * partial + (1.0 - mask) * torch.randn_like(partial)

        # Reverse diffusion loop: t = T-1 -> 0
        for t_val in reversed(range(self.scheduler.T)):
            t_batch = torch.full(
                (partial.shape[0],), t_val, device=self.device, dtype=torch.long
            )
            epsilon_pred = self.model(x_t, partial, mask, t_batch)
            x_t = self.scheduler.reverse_step(epsilon_pred, x_t, t_batch, mask, partial)

        # Final cleanup
        completed = x_t.clamp(0.0, 2.0)
        completed = mask * partial + (1.0 - mask) * completed

        return completed


class DiffusionModelWrapper(nn.Module):
    """
    Wraps DiffusionSampler with the same interface as baseline LiDARCompletionModel.
    """

    def __init__(
        self,
        model: DualEncoderDDPMUNet,
        scheduler: NoiseScheduler,
        device: str = "cuda"
    ):
        super().__init__()
        self.sampler = DiffusionSampler(model, scheduler, device)

    def forward(self, partial: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.sampler.sample(partial, mask)


def evaluate_diffusion(
    model: DualEncoderDDPMUNet,
    scheduler: NoiseScheduler,
    data_loader: DataLoader,
    device: str = "cuda"
) -> dict:
    """
    Full evaluation: run sampling on all data and compute metrics.

    Returns results in same format as baseline ModelEvaluator.evaluate().
    """
    sampler = DiffusionSampler(model, scheduler, device)
    metrics_list = []
    predictions = []

    pbar = tqdm(data_loader, desc="Evaluating (sampling)")
    for partial, mask, target in pbar:
        partial = partial.to(device)
        mask = mask.to(device)

        pred = sampler.sample(partial, mask)

        pred_np = pred.cpu().numpy()
        target_np = target.numpy()
        mask_np = mask.cpu().numpy()

        for i in range(pred_np.shape[0]):
            m = compute_metrics(pred_np[i], target_np[i], mask_np[i])
            metrics_list.append(m)
            predictions.append({
                'pred': pred_np[i],
                'target': target_np[i],
                'mask': mask_np[i],
            })

    summary = aggregate_metrics(metrics_list)

    return {
        'n_samples': len(metrics_list),
        'summary': summary,
        'per_sample': metrics_list,
        'predictions': predictions,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Inference & Evaluation for Diffusion v2 (Dual Encoder + FiLM)"
    )

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data_dirs", type=str, nargs="+", required=True,
                        help="Data directories for evaluation")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Output directory for results and visualizations")
    parser.add_argument("--T", type=int, default=100,
                        help="Number of diffusion timesteps")
    parser.add_argument("--base_channels", type=int, default=64,
                        help="Base channel count (must match training)")
    parser.add_argument("--num_res_blocks", type=int, default=2,
                        help="Number of FiLMResBlocks per level (must match training)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization plots")
    parser.add_argument("--n_vis_samples", type=int, default=5,
                        help="Number of samples to visualize")
    parser.add_argument("--plot_type", type=str, choices=["linear", "polar"],
                        default="linear",
                        help="Visualization plot type")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Compute device")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save_per_sample", action="store_true",
                        help="Save per-sample metrics to per_sample_diffusion_v2.json")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("\n" + "=" * 70)
    print("Loading Model")
    print("=" * 70)

    model = DualEncoderDDPMUNet(
        input_length=360,
        base_channels=args.base_channels,
        num_res_blocks=args.num_res_blocks,
        time_dim=128,
        num_groups=8,
    )

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device)
    model.eval()

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")

    # Noise scheduler
    scheduler = NoiseScheduler(T=args.T, device=args.device)

    # Load data
    print("\n" + "=" * 70)
    print("Loading Data")
    print("=" * 70)

    dataset = LiDAR2DCompletionDataset(
        data_dirs=args.data_dirs,
        normalize=True,
        augment=False,
        n_variations=5,
        fill_mode="zero",
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Evaluate
    print("\n" + "=" * 70)
    print("Evaluating")
    print("=" * 70)

    results = evaluate_diffusion(model, scheduler, data_loader, args.device)

    # Print results
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"\nTotal samples: {results['n_samples']}")

    summary = results['summary']
    for metric_name, values in summary.items():
        print(f"\n{metric_name}:")
        print(f"  Mean: {values['mean']:.6f}")
        print(f"  Std:  {values['std']:.6f}")
        print(f"  Min:  {values['min']:.6f}")
        print(f"  Max:  {values['max']:.6f}")

    # Save results
    save_results = {
        'n_samples': results['n_samples'],
        'summary': results['summary'],
        'checkpoint': args.checkpoint,
        'T': args.T,
        'base_channels': args.base_channels,
        'num_res_blocks': args.num_res_blocks,
        'data_dirs': args.data_dirs,
    }
    result_file = output_dir / "evaluation_diffusion_v2.json"
    with open(result_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to: {result_file}")

    if args.save_per_sample:
        per_sample_file = output_dir / "per_sample_diffusion_v2.json"
        with open(per_sample_file, 'w') as f:
            json.dump(results['per_sample'], f, indent=2)
        print(f"Per-sample metrics saved to: {per_sample_file}")

    # Visualization
    if args.visualize:
        print("\n" + "=" * 70)
        print("Generating Visualizations")
        print("=" * 70)

        fig_dir = output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        n_vis = min(args.n_vis_samples, len(results['predictions']))
        indices = np.random.choice(len(results['predictions']), n_vis, replace=False)

        for idx in indices:
            p = results['predictions'][idx]
            pred = p['pred']
            target = p['target']
            mask = p['mask']

            # Reconstruct partial from target and mask
            partial_vis = target * mask

            metrics = compute_metrics(pred, target, mask)
            save_path = str(fig_dir / f"sample_{idx:06d}.png")

            if args.plot_type == "polar":
                plot_comparison(partial_vis, target, pred, mask, metrics, save_path)
            else:
                plot_linear_comparison(partial_vis, target, pred, mask, metrics, save_path)

            print(f"  Saved: {save_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
