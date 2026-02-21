#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D LiDAR Point Completion - Visualization

可視化ツール
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from dataset import LiDAR2DCompletionDataset
from model import LiDARCompletionModel
from metrics import compute_metrics


def plot_lidar_polar(
    ax,
    distances: np.ndarray,
    mask: np.ndarray = None,
    title: str = "",
    color: str = "blue",
    alpha: float = 1.0,
    show_missing: bool = True
):
    """
    極座標でLiDARスキャンをプロット

    Args:
        ax: matplotlib polar axes
        distances: (360,) 距離データ
        mask: (360,) 観測マスク
        title: タイトル
        color: 観測点の色
        alpha: 透明度
        show_missing: 欠損領域を表示するか
    """
    n_beams = len(distances)
    angles = np.linspace(0, 2 * np.pi, n_beams, endpoint=False)

    if mask is not None and show_missing:
        # 観測点
        obs_mask = mask > 0
        ax.scatter(
            angles[obs_mask],
            distances[obs_mask],
            c=color,
            s=3,
            alpha=alpha,
            label="Observed"
        )

        # 欠損点
        miss_mask = mask == 0
        if miss_mask.sum() > 0:
            ax.scatter(
                angles[miss_mask],
                distances[miss_mask],
                c="red",
                s=3,
                alpha=alpha * 0.7,
                label="Missing/Filled",
                marker="x"
            )
    else:
        ax.scatter(angles, distances, c=color, s=3, alpha=alpha)

    ax.set_title(title)
    ax.set_ylim(0, 1.2)


def plot_comparison(
    partial: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    metrics: dict,
    save_path: str = None
):
    """
    入力・正解・予測の比較プロット

    Args:
        partial: 入力（欠損あり）
        target: 正解
        pred: 予測
        mask: 観測マスク
        metrics: 評価指標
        save_path: 保存先パス
    """
    fig = plt.figure(figsize=(15, 5))

    # 極座標プロット
    ax1 = fig.add_subplot(131, projection='polar')
    plot_lidar_polar(ax1, partial, mask, "Input (Partial)", "blue")

    ax2 = fig.add_subplot(132, projection='polar')
    plot_lidar_polar(ax2, target, None, "Target (Complete)", "green", show_missing=False)

    ax3 = fig.add_subplot(133, projection='polar')
    plot_lidar_polar(ax3, pred, mask, "Prediction", "purple")

    # メトリクス表示
    metrics_text = (
        f"MAE (all): {metrics['mae_all']:.4f}\n"
        f"MAE (missing): {metrics['mae_missing']:.4f}\n"
        f"Edge error: {metrics['edge_error']:.4f}\n"
        f"Missing rate: {metrics['missing_rate']:.1%}"
    )
    fig.text(0.02, 0.02, metrics_text, fontsize=10, family='monospace',
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_linear_comparison(
    partial: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    metrics: dict,
    save_path: str = None
):
    """
    線形プロットでの比較（角度 vs 距離）

    Args:
        partial: 入力（欠損あり）
        target: 正解
        pred: 予測
        mask: 観測マスク
        metrics: 評価指標
        save_path: 保存先パス
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    angles = np.arange(len(target))

    # 入力
    ax = axes[0]
    ax.fill_between(angles, 0, partial, where=mask > 0, alpha=0.3, color='blue', label='Observed')
    ax.plot(angles, partial, 'b-', linewidth=0.5)
    miss_mask = mask == 0
    if miss_mask.sum() > 0:
        ax.axvspan(angles[miss_mask].min(), angles[miss_mask].max(), alpha=0.2, color='red', label='Missing region')
    ax.set_ylabel('Distance (norm)')
    ax.set_title('Input (Partial)')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.2)

    # 正解
    ax = axes[1]
    ax.fill_between(angles, 0, target, alpha=0.3, color='green')
    ax.plot(angles, target, 'g-', linewidth=0.5)
    ax.set_ylabel('Distance (norm)')
    ax.set_title('Target (Complete)')
    ax.set_ylim(0, 1.2)

    # 予測
    ax = axes[2]
    ax.fill_between(angles, 0, pred, alpha=0.3, color='purple')
    ax.plot(angles, pred, 'purple', linewidth=0.5, label='Prediction')
    ax.plot(angles, target, 'g--', linewidth=0.5, alpha=0.5, label='Target')
    ax.set_ylabel('Distance (norm)')
    ax.set_xlabel('Beam index')
    ax.set_title('Prediction vs Target')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.2)

    # メトリクス
    metrics_text = (
        f"MAE (all): {metrics['mae_all']:.4f}  |  "
        f"MAE (missing): {metrics['mae_missing']:.4f}  |  "
        f"Edge error: {metrics['edge_error']:.4f}  |  "
        f"Missing rate: {metrics['missing_rate']:.1%}"
    )
    fig.suptitle(metrics_text, fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_samples(
    model: torch.nn.Module,
    dataset: LiDAR2DCompletionDataset,
    n_samples: int = 5,
    output_dir: str = "./figures",
    device: str = "cuda",
    plot_type: str = "linear"
):
    """
    複数サンプルの可視化

    Args:
        model: 学習済みモデル
        dataset: データセット
        n_samples: 可視化するサンプル数
        output_dir: 出力ディレクトリ
        device: デバイス
        plot_type: "polar" or "linear"
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    # ランダムにサンプルを選択
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    for i, idx in enumerate(indices):
        partial, mask, target = dataset[idx]

        # 予測
        with torch.no_grad():
            partial_t = partial.unsqueeze(0).to(device)
            mask_t = mask.unsqueeze(0).to(device)
            pred_t = model(partial_t, mask_t)
            pred = pred_t.squeeze(0).cpu().numpy()

        partial = partial.numpy()
        mask = mask.numpy()
        target = target.numpy()

        # メトリクス
        metrics = compute_metrics(pred, target, mask)

        # プロット
        save_path = output_dir / f"sample_{idx:06d}.png"

        if plot_type == "polar":
            plot_comparison(partial, target, pred, mask, metrics, str(save_path))
        else:
            plot_linear_comparison(partial, target, pred, mask, metrics, str(save_path))

        print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize LiDAR Point Completion Results"
    )

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data_dirs", type=str, nargs="+", required=True,
                        help="Data directories")
    parser.add_argument("--output_dir", type=str, default="./figures",
                        help="Output directory")
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Number of samples to visualize")
    parser.add_argument("--fill_mode", type=str, choices=["zero", "noise"],
                        default="zero")
    parser.add_argument("--noise_std", type=float, default=1.0)
    parser.add_argument("--plot_type", type=str, choices=["polar", "linear"],
                        default="linear")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_variations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    np.random.seed(args.seed)

    # データセット
    print("Loading dataset...")
    dataset = LiDAR2DCompletionDataset(
        data_dirs=args.data_dirs,
        normalize=True,
        augment=False,
        n_variations=args.n_variations,
        fill_mode=args.fill_mode,
        noise_std=args.noise_std
    )

    # モデル
    print("Loading model...")
    model = LiDARCompletionModel(
        input_length=360,
        use_groupnorm=True,
        use_sigmoid=True
    )

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # 可視化
    print(f"Visualizing {args.n_samples} samples...")
    visualize_samples(
        model=model,
        dataset=dataset,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        device=args.device,
        plot_type=args.plot_type
    )

    print("Done!")


if __name__ == "__main__":
    main()
