#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical tests comparing Conv(zero), Conv(noise), Diffusion v1, and Diffusion v2 models.

Comparisons:
  1. Conv(zero) vs Conv(noise)
  2. Conv(zero) vs Diffusion(noise)
  3. Conv(zero) vs Diffusion v2  (if --diffusion_v2_dirs provided)
  4. Diffusion v1 vs Diffusion v2  (if --diffusion_v2_dirs provided)

For each comparison, uses:
  - Paired t-test
  - Wilcoxon signed-rank test
  - Cohen's d effect size

Per-sample metrics are averaged across seeds before testing.
Averaging is valid because shuffle=False in DataLoader, so the same index
corresponds to the same sample across all seeds.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats


def load_per_sample_json(json_path: str) -> List[Dict[str, float]]:
    """Load per-sample metrics from a JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def pool_and_average_runs(
    run_dirs: List[str],
    per_sample_filename: str,
    metric_keys: List[str]
) -> Dict[str, np.ndarray]:
    """
    Load per-sample metrics from multiple runs and average across runs.

    Since shuffle=False in DataLoader, the same index corresponds to the
    same sample across different seeds. Averaging across runs removes
    training randomness before statistical testing.

    Args:
        run_dirs: List of evaluation output directories (one per seed).
        per_sample_filename: JSON filename (e.g. "per_sample_zero.json").
        metric_keys: Metric names to load and average.

    Returns:
        Dict mapping metric name -> np.ndarray of shape (n_samples,).
    """
    all_runs = []
    for run_dir in run_dirs:
        json_path = Path(run_dir) / per_sample_filename
        per_sample = load_per_sample_json(str(json_path))
        all_runs.append(per_sample)

    n_samples = len(all_runs[0])
    for i, run in enumerate(all_runs):
        if len(run) != n_samples:
            raise ValueError(
                f"Run {i} ({run_dirs[i]}) has {len(run)} samples, "
                f"expected {n_samples}"
            )

    # shape: (n_runs, n_samples) -> mean over axis=0 -> (n_samples,)
    averaged = {}
    for key in metric_keys:
        values = np.array([[sample[key] for sample in run] for run in all_runs])
        averaged[key] = values.mean(axis=0)

    return averaged


def run_paired_statistical_tests(
    method_a_metrics: Dict[str, np.ndarray],
    method_b_metrics: Dict[str, np.ndarray],
    metric_keys: List[str],
    name_a: str = "method_a",
    name_b: str = "method_b"
) -> Dict:
    """
    Run paired t-test, Wilcoxon signed-rank test, and Cohen's d for each metric.

    Positive improvement means method_a has a higher mean than method_b.
    For error metrics (MAE, edge_error), a positive improvement means method_a
    is *worse* than method_b.

    Args:
        method_a_metrics: Averaged per-sample metrics for method A.
        method_b_metrics: Averaged per-sample metrics for method B.
        metric_keys: Metrics to test.
        name_a: Name label for method A.
        name_b: Name label for method B.

    Returns:
        Dict mapping metric name -> test results.
    """
    results = {}

    for key in metric_keys:
        a_vals = method_a_metrics[key]
        b_vals = method_b_metrics[key]

        n = min(len(a_vals), len(b_vals))
        a_vals = a_vals[:n]
        b_vals = b_vals[:n]

        # Paired t-test
        t_stat, p_value_t = stats.ttest_rel(a_vals, b_vals)

        # Wilcoxon signed-rank test
        try:
            w_stat, p_value_w = stats.wilcoxon(a_vals, b_vals)
            w_stat_out = float(w_stat)
            p_value_w_out = float(p_value_w)
        except ValueError:
            w_stat_out = None
            p_value_w_out = None

        # Cohen's d (paired difference)
        diff = a_vals - b_vals
        cohens_d = float(diff.mean() / diff.std()) if diff.std() > 0 else 0.0

        results[key] = {
            f"{name_a}_mean": float(a_vals.mean()),
            f"{name_a}_std": float(a_vals.std()),
            f"{name_b}_mean": float(b_vals.mean()),
            f"{name_b}_std": float(b_vals.std()),
            "t_test": {
                "statistic": float(t_stat),
                "p_value": float(p_value_t)
            },
            "wilcoxon": {
                "statistic": w_stat_out,
                "p_value": p_value_w_out
            },
            "effect_size_cohens_d": cohens_d,
            "improvement": float(a_vals.mean() - b_vals.mean()),
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Statistical tests for LiDAR completion model comparison"
    )

    parser.add_argument("--zero_fill_dirs", type=str, nargs="+", required=True,
                        help="Eval output directories for Conv(zero) runs")
    parser.add_argument("--noise_fill_dirs", type=str, nargs="+", required=True,
                        help="Eval output directories for Conv(noise) runs")
    parser.add_argument("--diffusion_dirs", type=str, nargs="+", required=True,
                        help="Eval output directories for Diffusion v1 runs")
    parser.add_argument("--diffusion_v2_dirs", type=str, nargs="+", required=False,
                        default=None,
                        help="Eval output directories for Diffusion v2 runs (optional)")
    parser.add_argument("--output_dir", type=str,
                        default="results/statistical_tests",
                        help="Output directory for results")
    parser.add_argument("--metric_keys", type=str, nargs="+",
                        default=["mae_missing", "edge_error"],
                        help="Metrics to compare")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                        help="Seeds used (for metadata only)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_runs = len(args.zero_fill_dirs)

    print("=" * 70)
    print("Statistical Tests for LiDAR Completion Models")
    print("=" * 70)
    print(f"Runs        : {n_runs} (seeds: {args.seeds})")
    print(f"Metrics     : {args.metric_keys}")
    if args.diffusion_v2_dirs:
        print(f"Diffusion v2: included")
    print()

    # Load and average per-sample metrics across runs
    print("Loading per-sample metrics...")
    zero_metrics = pool_and_average_runs(
        args.zero_fill_dirs, "per_sample_zero.json", args.metric_keys
    )
    noise_metrics = pool_and_average_runs(
        args.noise_fill_dirs, "per_sample_noise.json", args.metric_keys
    )
    diffusion_metrics = pool_and_average_runs(
        args.diffusion_dirs, "per_sample_diffusion.json", args.metric_keys
    )

    diffusion_v2_metrics = None
    if args.diffusion_v2_dirs:
        diffusion_v2_metrics = pool_and_average_runs(
            args.diffusion_v2_dirs, "per_sample_diffusion_v2.json", args.metric_keys
        )

    n_samples = len(next(iter(zero_metrics.values())))
    print(f"Samples per model (averaged across {n_runs} runs): {n_samples}")
    print()

    # Comparison 1: Conv(zero) vs Conv(noise)
    print("Comparison 1: Conv(zero) vs Conv(noise)")
    print("-" * 50)
    comp1 = run_paired_statistical_tests(
        zero_metrics, noise_metrics, args.metric_keys,
        name_a="conv_zero", name_b="conv_noise"
    )
    for key, res in comp1.items():
        print(f"  {key}:")
        print(f"    Conv(zero) : {res['conv_zero_mean']:.6f} ± {res['conv_zero_std']:.6f}")
        print(f"    Conv(noise): {res['conv_noise_mean']:.6f} ± {res['conv_noise_std']:.6f}")
        print(f"    t-test     : p = {res['t_test']['p_value']:.4e}")
        p_w = res['wilcoxon']['p_value']
        print(f"    Wilcoxon   : p = {p_w:.4e}" if p_w is not None else "    Wilcoxon   : N/A")
        print(f"    Cohen's d  : {res['effect_size_cohens_d']:.4f}")
        print()

    # Comparison 2: Conv(zero) vs Diffusion v1
    print("Comparison 2: Conv(zero) vs Diffusion(noise)")
    print("-" * 50)
    comp2 = run_paired_statistical_tests(
        zero_metrics, diffusion_metrics, args.metric_keys,
        name_a="conv_zero", name_b="diffusion"
    )
    for key, res in comp2.items():
        print(f"  {key}:")
        print(f"    Conv(zero) : {res['conv_zero_mean']:.6f} ± {res['conv_zero_std']:.6f}")
        print(f"    Diffusion  : {res['diffusion_mean']:.6f} ± {res['diffusion_std']:.6f}")
        print(f"    t-test     : p = {res['t_test']['p_value']:.4e}")
        p_w = res['wilcoxon']['p_value']
        print(f"    Wilcoxon   : p = {p_w:.4e}" if p_w is not None else "    Wilcoxon   : N/A")
        print(f"    Cohen's d  : {res['effect_size_cohens_d']:.4f}")
        print()

    comp3 = None
    comp4 = None

    if diffusion_v2_metrics is not None:
        # Comparison 3: Conv(zero) vs Diffusion v2
        print("Comparison 3: Conv(zero) vs Diffusion v2")
        print("-" * 50)
        comp3 = run_paired_statistical_tests(
            zero_metrics, diffusion_v2_metrics, args.metric_keys,
            name_a="conv_zero", name_b="diffusion_v2"
        )
        for key, res in comp3.items():
            print(f"  {key}:")
            print(f"    Conv(zero)    : {res['conv_zero_mean']:.6f} ± {res['conv_zero_std']:.6f}")
            print(f"    Diffusion v2  : {res['diffusion_v2_mean']:.6f} ± {res['diffusion_v2_std']:.6f}")
            print(f"    t-test        : p = {res['t_test']['p_value']:.4e}")
            p_w = res['wilcoxon']['p_value']
            print(f"    Wilcoxon      : p = {p_w:.4e}" if p_w is not None else "    Wilcoxon      : N/A")
            print(f"    Cohen's d     : {res['effect_size_cohens_d']:.4f}")
            print()

        # Comparison 4: Diffusion v1 vs Diffusion v2
        print("Comparison 4: Diffusion v1 vs Diffusion v2")
        print("-" * 50)
        comp4 = run_paired_statistical_tests(
            diffusion_metrics, diffusion_v2_metrics, args.metric_keys,
            name_a="diffusion", name_b="diffusion_v2"
        )
        for key, res in comp4.items():
            print(f"  {key}:")
            print(f"    Diffusion v1  : {res['diffusion_mean']:.6f} ± {res['diffusion_std']:.6f}")
            print(f"    Diffusion v2  : {res['diffusion_v2_mean']:.6f} ± {res['diffusion_v2_std']:.6f}")
            print(f"    t-test        : p = {res['t_test']['p_value']:.4e}")
            p_w = res['wilcoxon']['p_value']
            print(f"    Wilcoxon      : p = {p_w:.4e}" if p_w is not None else "    Wilcoxon      : N/A")
            print(f"    Cohen's d     : {res['effect_size_cohens_d']:.4f}")
            print()

    # Save results
    comparisons = {
        "conv_zero_vs_conv_noise": comp1,
        "conv_zero_vs_diffusion": comp2,
    }
    if comp3 is not None:
        comparisons["conv_zero_vs_diffusion_v2"] = comp3
    if comp4 is not None:
        comparisons["diffusion_v1_vs_diffusion_v2"] = comp4

    metadata = {
        "n_runs": n_runs,
        "seeds": args.seeds,
        "n_samples": n_samples,
        "metric_keys": args.metric_keys,
        "zero_fill_dirs": args.zero_fill_dirs,
        "noise_fill_dirs": args.noise_fill_dirs,
        "diffusion_dirs": args.diffusion_dirs,
    }
    if args.diffusion_v2_dirs:
        metadata["diffusion_v2_dirs"] = args.diffusion_v2_dirs

    output = {
        "comparisons": comparisons,
        "metadata": metadata,
    }

    result_file = output_dir / "statistical_results.json"
    with open(result_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {result_file}")


if __name__ == "__main__":
    main()
