#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D LiDAR Point Completion - Visualization

可視化ツール
"""

import numpy as np
import matplotlib.pyplot as plt


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
