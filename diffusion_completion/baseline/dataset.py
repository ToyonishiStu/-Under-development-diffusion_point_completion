#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch Dataset for 2D LiDAR Point Completion (with fill_mode support)

fill_mode:
  - "zero": 欠損部を0で埋める（従来手法）
  - "noise": 欠損部をGaussian Noiseで埋める（新手法）
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Union, List


class LiDAR2DCompletionDataset(Dataset):
    """2D LiDAR点群補完用データセット（fill_mode対応）"""

    def __init__(
        self,
        data_dirs: Union[str, List[str]],
        normalize: bool = True,
        augment: bool = False,
        n_variations: int = 5,
        fill_mode: str = "zero",
        noise_std: float = 1.0
    ):
        """
        Args:
            data_dirs: データディレクトリ（単一のstrまたはstrのリスト）
            normalize: 距離を正規化するか
            augment: データ拡張を行うか（回転など）
            n_variations: 各npzファイルに含まれるpartialの数
            fill_mode: "zero" or "noise" - 欠損部の埋め方
            noise_std: noise fill時の標準偏差（正規化後の空間で）
        """
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        self.data_dirs = data_dirs
        self.normalize = normalize
        self.augment = augment
        self.n_variations = n_variations
        self.fill_mode = fill_mode
        self.noise_std = noise_std

        if fill_mode not in ["zero", "noise"]:
            raise ValueError(f"fill_mode must be 'zero' or 'noise', got '{fill_mode}'")

        # 全ディレクトリから.npzファイルのリストを取得
        self.file_list = []
        for data_dir in self.data_dirs:
            files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
            self.file_list.extend(files)
            print(f"  - Found {len(files)} files in {data_dir}")

        if len(self.file_list) == 0:
            raise ValueError(f"No .npz files found in {self.data_dirs}")

        total_samples = len(self.file_list) * self.n_variations
        print(f"Loaded {len(self.file_list)} files from {len(self.data_dirs)} directory(s)")
        print(f"Total samples: {total_samples} ({len(self.file_list)} files x {self.n_variations} variations)")
        print(f"Fill mode: {self.fill_mode}")

        # 正規化のための統計量を計算
        if self.normalize:
            self._compute_normalization_stats()

    def _compute_normalization_stats(self):
        """正規化のための統計量を計算"""
        print("Computing normalization statistics...")

        max_distances = []

        sample_size = min(1000, len(self.file_list))
        indices = np.random.choice(len(self.file_list), sample_size, replace=False)

        for idx in indices:
            data = np.load(self.file_list[idx])
            complete = data['complete']
            valid_distances = complete[complete > 0]
            if len(valid_distances) > 0:
                max_distances.append(np.max(valid_distances))

        self.distance_max = np.percentile(max_distances, 95)
        print(f"  Distance normalization max: {self.distance_max:.2f}m")

    def __len__(self) -> int:
        return len(self.file_list) * self.n_variations

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            partial: shape (n_beams,) - 欠損あり距離データ (fill_modeに応じて処理済み)
            mask: shape (n_beams,) - 欠損マスク (1.0=観測あり, 0.0=欠損)
            target: shape (n_beams,) - 欠損なし距離データ（ground truth）
        """
        # インデックスからファイルとパターンを決定
        file_idx = idx // self.n_variations
        variation_idx = idx % self.n_variations + 1

        # データを読み込み
        data = np.load(self.file_list[file_idx])

        partial_key = f'partial{variation_idx}'
        if partial_key not in data:
            if 'partial' in data and self.n_variations == 1:
                partial = data['partial'].astype(np.float32)
            else:
                raise KeyError(f"{partial_key} not found in {self.file_list[file_idx]}")
        else:
            partial = data[partial_key].astype(np.float32)

        target = data['complete'].astype(np.float32)

        # データ拡張（回転）- mask計算前に実行
        if self.augment:
            shift = np.random.randint(0, len(partial))
            partial = np.roll(partial, shift)
            target = np.roll(target, shift)

        # maskを生成（正規化前のpartialから計算）
        # partial > 0 の位置: 観測あり (1.0)
        # partial == 0 の位置: 欠損 (0.0)
        mask = (partial > 0).astype(np.float32)

        # 正規化
        if self.normalize:
            partial = partial / self.distance_max
            target = target / self.distance_max

        # fill_modeに応じて欠損部を処理（正規化後に適用）
        if self.fill_mode == "noise":
            # 欠損部にGaussian Noiseを埋める
            noise = np.random.randn(len(partial)).astype(np.float32) * self.noise_std
            partial = np.where(mask > 0, partial, noise)
        # else: fill_mode == "zero" - partialは既に欠損部が0

        # Tensorに変換
        partial = torch.from_numpy(partial)
        mask = torch.from_numpy(mask)
        target = torch.from_numpy(target)

        return partial, mask, target


def create_dataloaders(
    train_dirs: Union[str, List[str]],
    val_dirs: Union[str, List[str]],
    batch_size: int = 32,
    num_workers: int = 4,
    normalize: bool = True,
    augment_train: bool = True,
    n_variations: int = 5,
    fill_mode: str = "zero",
    noise_std: float = 1.0
) -> Tuple[DataLoader, DataLoader]:
    """
    データローダーを作成

    Args:
        train_dirs: 学習用データディレクトリ
        val_dirs: 検証用データディレクトリ
        batch_size: バッチサイズ
        num_workers: データローディングのワーカー数
        normalize: 距離を正規化するか
        augment_train: 学習データに拡張を適用するか
        n_variations: 各npzファイルに含まれるpartialの数
        fill_mode: "zero" or "noise"
        noise_std: noise fill時の標準偏差

    Returns:
        train_loader, val_loader
    """
    print("\n" + "=" * 60)
    print("Creating Train Dataset")
    print("=" * 60)

    train_dataset = LiDAR2DCompletionDataset(
        data_dirs=train_dirs,
        normalize=normalize,
        augment=augment_train,
        n_variations=n_variations,
        fill_mode=fill_mode,
        noise_std=noise_std
    )

    print("\n" + "=" * 60)
    print("Creating Val Dataset")
    print("=" * 60)

    val_dataset = LiDAR2DCompletionDataset(
        data_dirs=val_dirs,
        normalize=normalize,
        augment=False,
        n_variations=n_variations,
        fill_mode=fill_mode,
        noise_std=noise_std
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def test_fill_modes():
    """fill_modeの動作確認テスト"""
    import sys

    print("=" * 70)
    print("Fill Mode Test")
    print("=" * 70)

    # テスト用データディレクトリ
    test_dir = "../output/train"
    if not os.path.exists(test_dir):
        test_dir = "/workspaces/toyot/diffusion_completion/output/train"

    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")
        sys.exit(1)

    # Zero fill
    print("\n--- Zero Fill ---")
    dataset_zero = LiDAR2DCompletionDataset(
        data_dirs=test_dir,
        normalize=True,
        augment=False,
        n_variations=5,
        fill_mode="zero"
    )
    partial_z, mask_z, target_z = dataset_zero[0]

    # Noise fill
    print("\n--- Noise Fill ---")
    dataset_noise = LiDAR2DCompletionDataset(
        data_dirs=test_dir,
        normalize=True,
        augment=False,
        n_variations=5,
        fill_mode="noise",
        noise_std=1.0
    )
    partial_n, mask_n, target_n = dataset_noise[0]

    print("\n--- Comparison ---")
    print(f"Shapes: partial={partial_z.shape}, mask={mask_z.shape}, target={target_z.shape}")

    # マスクは同じであるべき
    print(f"Masks equal: {torch.allclose(mask_z, mask_n)}")

    # ターゲットは同じであるべき
    print(f"Targets equal: {torch.allclose(target_z, target_n)}")

    # 観測部は同じであるべき
    obs_mask = mask_z > 0
    print(f"Observed parts equal: {torch.allclose(partial_z[obs_mask], partial_n[obs_mask])}")

    # 欠損部は異なるべき（zero vs noise）
    miss_mask = mask_z == 0
    n_missing = miss_mask.sum().item()
    print(f"Missing points: {n_missing}")

    if n_missing > 0:
        zero_fill_values = partial_z[miss_mask]
        noise_fill_values = partial_n[miss_mask]

        print(f"Zero fill - missing values: min={zero_fill_values.min():.4f}, max={zero_fill_values.max():.4f}, mean={zero_fill_values.mean():.4f}")
        print(f"Noise fill - missing values: min={noise_fill_values.min():.4f}, max={noise_fill_values.max():.4f}, mean={noise_fill_values.mean():.4f}, std={noise_fill_values.std():.4f}")

        # Zero fillは全て0であるべき
        assert torch.all(zero_fill_values == 0), "Zero fill should have all zeros in missing regions"
        print("Zero fill: All missing values are 0")

        # Noise fillは0でない値を含むべき
        assert not torch.all(noise_fill_values == 0), "Noise fill should have non-zero values in missing regions"
        print("Noise fill: Missing values contain noise")

    print("\nTest passed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run fill_mode test")
    parser.add_argument("--train_dirs", type=str, nargs='+', help="Training data directories")
    parser.add_argument("--val_dirs", type=str, nargs='+', help="Validation data directories")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fill_mode", type=str, choices=["zero", "noise"], default="zero")
    parser.add_argument("--noise_std", type=float, default=1.0)
    args = parser.parse_args()

    if args.test:
        test_fill_modes()
    elif args.train_dirs and args.val_dirs:
        train_loader, val_loader = create_dataloaders(
            train_dirs=args.train_dirs,
            val_dirs=args.val_dirs,
            batch_size=args.batch_size,
            fill_mode=args.fill_mode,
            noise_std=args.noise_std
        )

        print("\n" + "=" * 60)
        print("DataLoader Test")
        print("=" * 60)

        for partial, mask, target in train_loader:
            print(f"\nBatch shapes:")
            print(f"  Partial: {partial.shape}")
            print(f"  Mask: {mask.shape}")
            print(f"  Target: {target.shape}")
            print(f"\nValue ranges:")
            print(f"  Partial: [{partial.min():.3f}, {partial.max():.3f}]")
            print(f"  Mask: [{mask.min():.3f}, {mask.max():.3f}]")
            print(f"  Target: [{target.min():.3f}, {target.max():.3f}]")
            break

        print(f"\nTrain batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
    else:
        print("Usage:")
        print("  python dataset.py --test")
        print("  python dataset.py --train_dirs DIR1 [DIR2 ...] --val_dirs DIR1 [DIR2 ...]")
