#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation Metrics for 2D LiDAR Point Completion

評価指標:
  - MAE (全体)
  - RMSE (全体)
  - MAE (欠損部のみ)
  - Edge Error (欠損境界付近のMAE)
"""

import numpy as np
from typing import Dict, Union
import torch


def compute_edge_error(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    edge_width: int = 5
) -> float:
    """
    欠損境界付近のMAEを計算

    Args:
        pred: (N,) 予測値
        target: (N,) 正解値
        mask: (N,) 観測マスク (1=観測, 0=欠損)
        edge_width: 境界から何点を含むか

    Returns:
        edge_error: 境界付近のMAE
    """
    n = len(mask)

    # マスクの変化点を検出（0→1 または 1→0）
    # 循環的に処理するため、最後の要素と最初の要素の差も考慮
    mask_diff = np.diff(mask, prepend=mask[-1])
    edge_indices = np.where(mask_diff != 0)[0]

    if len(edge_indices) == 0:
        return 0.0

    # 境界周辺のマスクを作成
    edge_mask = np.zeros(n, dtype=bool)
    for idx in edge_indices:
        for offset in range(-edge_width, edge_width + 1):
            edge_mask[(idx + offset) % n] = True

    # 境界付近のMAEを計算
    if edge_mask.sum() > 0:
        edge_error = np.abs(pred[edge_mask] - target[edge_mask]).mean()
    else:
        edge_error = 0.0

    return float(edge_error)


def compute_metrics(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    mask: Union[np.ndarray, torch.Tensor],
    edge_width: int = 5
) -> Dict[str, float]:
    """
    全ての評価指標を計算

    Args:
        pred: (N,) or (B, N) 予測値
        target: (N,) or (B, N) 正解値
        mask: (N,) or (B, N) 観測マスク (1=観測, 0=欠損)
        edge_width: Edge Error計算時の境界幅

    Returns:
        dict: {
            'mae_all': 全体のMAE,
            'rmse_all': 全体のRMSE,
            'mae_missing': 欠損部のみのMAE,
            'edge_error': 境界付近のMAE,
            'n_missing': 欠損点数,
            'missing_rate': 欠損率
        }
    """
    # Tensorの場合はnumpyに変換
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # バッチ次元がある場合はフラット化
    pred = pred.flatten()
    target = target.flatten()
    mask = mask.flatten()

    # 全体のMAE/RMSE
    diff = pred - target
    mae_all = np.abs(diff).mean()
    rmse_all = np.sqrt((diff ** 2).mean())

    # 欠損部のみのMAE
    missing_mask = (mask == 0)
    n_missing = missing_mask.sum()

    if n_missing > 0:
        mae_missing = np.abs(diff[missing_mask]).mean()
    else:
        mae_missing = 0.0

    # Edge Error
    edge_error = compute_edge_error(pred, target, mask, edge_width)

    # 欠損率
    missing_rate = n_missing / len(mask)

    return {
        'mae_all': float(mae_all),
        'rmse_all': float(rmse_all),
        'mae_missing': float(mae_missing),
        'edge_error': float(edge_error),
        'n_missing': int(n_missing),
        'missing_rate': float(missing_rate)
    }


def compute_batch_metrics(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    mask: Union[np.ndarray, torch.Tensor],
    edge_width: int = 5
) -> Dict[str, np.ndarray]:
    """
    バッチ内の各サンプルに対して評価指標を計算

    Args:
        pred: (B, N) 予測値
        target: (B, N) 正解値
        mask: (B, N) 観測マスク

    Returns:
        dict: 各指標のバッチ配列
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    batch_size = pred.shape[0]

    metrics_list = {
        'mae_all': [],
        'rmse_all': [],
        'mae_missing': [],
        'edge_error': [],
        'missing_rate': []
    }

    for i in range(batch_size):
        m = compute_metrics(pred[i], target[i], mask[i], edge_width)
        for key in metrics_list:
            metrics_list[key].append(m[key])

    return {k: np.array(v) for k, v in metrics_list.items()}


def aggregate_metrics(metrics_list: list) -> Dict[str, Dict[str, float]]:
    """
    複数サンプルのメトリクスを集計

    Args:
        metrics_list: compute_metricsの出力のリスト

    Returns:
        dict: 各指標の平均と標準偏差
    """
    keys = ['mae_all', 'rmse_all', 'mae_missing', 'edge_error', 'missing_rate']
    result = {}

    for key in keys:
        values = [m[key] for m in metrics_list]
        result[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }

    return result


if __name__ == "__main__":
    print("=" * 70)
    print("Metrics Test")
    print("=" * 70)

    # テストデータ
    np.random.seed(42)
    n = 360

    # 正解
    target = np.random.rand(n).astype(np.float32) * 0.8 + 0.1

    # マスク（30%欠損）
    mask = np.ones(n, dtype=np.float32)
    mask[100:180] = 0  # 連続欠損

    # 予測（少しノイズを加える）
    pred = target.copy()
    pred += np.random.randn(n).astype(np.float32) * 0.05

    # 欠損部は大きめの誤差
    pred[100:180] += np.random.randn(80).astype(np.float32) * 0.1

    # メトリクス計算
    metrics = compute_metrics(pred, target, mask)

    print("\nSingle Sample Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")

    # バッチテスト
    batch_pred = np.stack([pred, pred + 0.01], axis=0)
    batch_target = np.stack([target, target], axis=0)
    batch_mask = np.stack([mask, mask], axis=0)

    batch_metrics = compute_batch_metrics(batch_pred, batch_target, batch_mask)

    print("\nBatch Metrics:")
    for key, values in batch_metrics.items():
        print(f"  {key}: mean={values.mean():.6f}, std={values.std():.6f}")

    # Edge error詳細テスト
    print("\n--- Edge Error Test ---")
    edge_error = compute_edge_error(pred, target, mask, edge_width=5)
    print(f"Edge error (width=5): {edge_error:.6f}")

    edge_error_10 = compute_edge_error(pred, target, mask, edge_width=10)
    print(f"Edge error (width=10): {edge_error_10:.6f}")

    print("\nTest passed!")
