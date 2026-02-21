#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D LiDAR Point Completion - Evaluation Script

学習済みモデルの評価スクリプト
"""

import argparse
import os
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import stats

from dataset import LiDAR2DCompletionDataset
from model import LiDARCompletionModel
from metrics import compute_metrics, aggregate_metrics


class ModelEvaluator:
    """モデル評価クラス"""

    def __init__(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device

    def evaluate(self, save_predictions: bool = False) -> Dict:
        """
        全サンプルに対して評価を実行

        Args:
            save_predictions: 予測結果を保存するか

        Returns:
            dict: 評価結果
        """
        self.model.eval()
        metrics_list = []
        predictions = []

        with torch.no_grad():
            pbar = tqdm(self.data_loader, desc="Evaluating")
            for batch_idx, (partial, mask, target) in enumerate(pbar):
                partial = partial.to(self.device)
                mask = mask.to(self.device)
                target = target.to(self.device)

                pred = self.model(partial, mask)

                # CPUに戻してメトリクス計算
                pred_np = pred.cpu().numpy()
                target_np = target.cpu().numpy()
                mask_np = mask.cpu().numpy()

                # 各サンプルのメトリクスを計算
                batch_size = pred_np.shape[0]
                for i in range(batch_size):
                    m = compute_metrics(pred_np[i], target_np[i], mask_np[i])
                    metrics_list.append(m)

                    if save_predictions:
                        predictions.append({
                            'pred': pred_np[i],
                            'target': target_np[i],
                            'mask': mask_np[i]
                        })

        # 集計
        summary = aggregate_metrics(metrics_list)

        result = {
            'n_samples': len(metrics_list),
            'summary': summary,
            'per_sample': metrics_list
        }

        if save_predictions:
            result['predictions'] = predictions

        return result


def compare_methods(
    zero_fill_results: Dict,
    noise_fill_results: Dict,
    metric_keys: List[str] = ['mae_missing', 'edge_error']
) -> Dict:
    """
    2つの手法を統計的に比較

    Args:
        zero_fill_results: zero fill の評価結果
        noise_fill_results: noise fill の評価結果
        metric_keys: 比較する指標

    Returns:
        dict: 統計検定の結果
    """
    results = {}

    for key in metric_keys:
        zero_values = [m[key] for m in zero_fill_results['per_sample']]
        noise_values = [m[key] for m in noise_fill_results['per_sample']]

        # 同じサンプル数であることを確認
        n = min(len(zero_values), len(noise_values))
        zero_values = zero_values[:n]
        noise_values = noise_values[:n]

        # Paired t-test
        t_stat, p_value_t = stats.ttest_rel(zero_values, noise_values)

        # Wilcoxon signed-rank test
        try:
            w_stat, p_value_w = stats.wilcoxon(zero_values, noise_values)
        except ValueError:
            w_stat, p_value_w = np.nan, np.nan

        # Effect size (Cohen's d)
        diff = np.array(zero_values) - np.array(noise_values)
        if diff.std() > 0:
            cohens_d = diff.mean() / diff.std()
        else:
            cohens_d = 0.0

        results[key] = {
            'zero_fill_mean': float(np.mean(zero_values)),
            'zero_fill_std': float(np.std(zero_values)),
            'noise_fill_mean': float(np.mean(noise_values)),
            'noise_fill_std': float(np.std(noise_values)),
            't_test': {
                'statistic': float(t_stat),
                'p_value': float(p_value_t)
            },
            'wilcoxon': {
                'statistic': float(w_stat) if not np.isnan(w_stat) else None,
                'p_value': float(p_value_w) if not np.isnan(p_value_w) else None
            },
            'effect_size_cohens_d': float(cohens_d),
            'improvement': float(np.mean(zero_values) - np.mean(noise_values))
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Conv Baseline Model"
    )

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data_dirs", type=str, nargs="+", required=True,
                        help="Data directories for evaluation")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Output directory for results")
    parser.add_argument("--fill_mode", type=str, choices=["zero", "noise"],
                        default="zero")
    parser.add_argument("--noise_std", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--save_per_sample", action="store_true",
                        help="Save per-sample metrics to per_sample_{fill_mode}.json")
    parser.add_argument("--n_variations", type=int, default=5)

    args = parser.parse_args()

    # 出力ディレクトリ
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # データセット
    print("\n" + "=" * 70)
    print("Loading Data")
    print("=" * 70)

    dataset = LiDAR2DCompletionDataset(
        data_dirs=args.data_dirs,
        normalize=True,
        augment=False,
        n_variations=args.n_variations,
        fill_mode=args.fill_mode,
        noise_std=args.noise_std
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # モデル
    print("\n" + "=" * 70)
    print("Loading Model")
    print("=" * 70)

    model = LiDARCompletionModel(
        input_length=360,
        use_groupnorm=True,
        use_sigmoid=True
    )

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")

    # 評価
    print("\n" + "=" * 70)
    print("Evaluating")
    print("=" * 70)

    evaluator = ModelEvaluator(model, data_loader, args.device)
    results = evaluator.evaluate(save_predictions=args.save_predictions)

    # 結果表示
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)

    print(f"\nTotal samples: {results['n_samples']}")
    print(f"Fill mode: {args.fill_mode}")

    summary = results['summary']
    for metric_name, values in summary.items():
        print(f"\n{metric_name}:")
        print(f"  Mean: {values['mean']:.6f}")
        print(f"  Std:  {values['std']:.6f}")
        print(f"  Min:  {values['min']:.6f}")
        print(f"  Max:  {values['max']:.6f}")

    # 結果を保存
    output_file = output_dir / f"evaluation_{args.fill_mode}.json"

    # 予測を除いた結果を保存
    save_results = {
        'n_samples': results['n_samples'],
        'summary': results['summary'],
        'checkpoint': args.checkpoint,
        'fill_mode': args.fill_mode,
        'data_dirs': args.data_dirs
    }

    with open(output_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    if args.save_per_sample:
        per_sample_file = output_dir / f"per_sample_{args.fill_mode}.json"
        with open(per_sample_file, 'w') as f:
            json.dump(results['per_sample'], f, indent=2)
        print(f"Per-sample metrics saved to: {per_sample_file}")

    # 予測も保存する場合
    if args.save_predictions:
        pred_file = output_dir / f"predictions_{args.fill_mode}.npz"
        np.savez_compressed(
            pred_file,
            predictions=np.array([p['pred'] for p in results['predictions']]),
            targets=np.array([p['target'] for p in results['predictions']]),
            masks=np.array([p['mask'] for p in results['predictions']])
        )
        print(f"Predictions saved to: {pred_file}")


if __name__ == "__main__":
    main()