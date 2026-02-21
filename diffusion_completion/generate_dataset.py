#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KITTI 3D LiDAR Dataset to 2D LiDAR Point Completion Dataset Generator

このスクリプトは、KITTI 3D LiDARデータセットを加工し、
2D LiDAR点群補完タスク用の教師ありデータセットを生成します。

生成方針:
- PCN (Point Completion Network) と同一思想
- 完全点群 → 人工的欠損 → 補完
"""

import argparse
import os
from pathlib import Path
import numpy as np
from typing import Tuple, Optional
import glob
from tqdm import tqdm


class KITTIto2DLiDARConverter:
    """KITTI 3D点群を2D LiDAR点群補完データセットに変換"""
    
    def __init__(
        self,
        n_beams: int = 360,
        r_max: float = 30.0,
        z_min: float = -1.5,
        occlusion_min: float = 0.2,
        occlusion_max: float = 0.5,
        train_ratio: float = 0.9,
        n_variations: int = 5,
        seed: int = 42
    ):
        """
        Args:
            n_beams: LiDARビーム数（水平方向の分割数）
            r_max: 最大距離 [m]
            z_min: 地面除去の閾値 [m]
            occlusion_min: 最小欠損率
            occlusion_max: 最大欠損率
            train_ratio: 学習用データの割合
            n_variations: 1つの点群から生成する欠損パターンの数
            seed: 乱数シード
        """
        self.n_beams = n_beams
        self.r_max = r_max
        self.z_min = z_min
        self.occlusion_min = occlusion_min
        self.occlusion_max = occlusion_max
        self.train_ratio = train_ratio
        self.n_variations = n_variations
        self.seed = seed
        
        np.random.seed(seed)
    
    def load_kitti_bin(self, bin_path: str) -> np.ndarray:
        """
        KITTI .binファイルを読み込む
        
        Args:
            bin_path: .binファイルのパス
            
        Returns:
            points: shape (N, 4) [x, y, z, reflectance]
        """
        points = np.fromfile(bin_path, dtype=np.float32)
        points = points.reshape(-1, 4)
        return points
    
    def preprocess_3d_points(self, points: np.ndarray) -> np.ndarray:
        """
        3D点群の前処理
        
        Args:
            points: shape (N, 4) [x, y, z, reflectance]
            
        Returns:
            filtered_points: shape (M, 4) where M <= N
        """
        # XY平面距離を計算
        xy_distance = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        
        # 距離制限
        distance_mask = xy_distance <= self.r_max
        
        # 地面除去（簡易）
        ground_mask = points[:, 2] > self.z_min
        
        # 両方の条件を満たす点のみ残す
        valid_mask = distance_mask & ground_mask
        filtered_points = points[valid_mask]
        
        return filtered_points
    
    def convert_to_2d_lidar(self, points: np.ndarray) -> np.ndarray:
        """
        3D点群を2D LiDARスキャンに変換（complete生成）
        
        Args:
            points: shape (N, 4) [x, y, z, reflectance]
            
        Returns:
            complete: shape (n_beams,) 各角度ビンの最小距離
        """
        # 角度を計算 [-π, π)
        angles = np.arctan2(points[:, 1], points[:, 0])
        
        # XY平面距離を計算
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        
        # 角度を[0, 2π)に正規化
        angles_normalized = angles + np.pi
        
        # 角度ビンのインデックスを計算
        angle_bins = (angles_normalized / (2 * np.pi) * self.n_beams).astype(np.int32)
        angle_bins = np.clip(angle_bins, 0, self.n_beams - 1)
        
        # 各角度ビンの最小距離を計算
        complete = np.zeros(self.n_beams, dtype=np.float32)
        
        for i in range(self.n_beams):
            bin_mask = angle_bins == i
            if np.any(bin_mask):
                complete[i] = np.min(distances[bin_mask])
            # else: 0.0 のまま（点が存在しない）
        
        return complete
    
    def generate_occlusion(self, complete: np.ndarray) -> np.ndarray:
        """
        連続角度欠損を生成（partial生成）
        
        Args:
            complete: shape (n_beams,) 完全スキャン
            
        Returns:
            partial: shape (n_beams,) 欠損を含むスキャン
        """
        partial = complete.copy()
        
        # 欠損率をランダムに決定
        occlusion_ratio = np.random.uniform(self.occlusion_min, self.occlusion_max)
        
        # 欠損区間の長さ
        occlusion_length = int(self.n_beams * occlusion_ratio)
        
        # 欠損の開始位置をランダムに決定
        start_idx = np.random.randint(0, self.n_beams)
        
        # 連続した角度領域を欠損させる（循環的に）
        for i in range(occlusion_length):
            idx = (start_idx + i) % self.n_beams
            partial[idx] = 0.0
        
        return partial
    
    def process_single_frame(
        self, 
        bin_path: str
    ) -> Optional[Tuple[list, np.ndarray]]:
        """
        1フレームを処理（複数の欠損パターンを生成）
        
        Args:
            bin_path: .binファイルのパス
            
        Returns:
            (partial_list, complete) or None if invalid
            partial_list: list of np.ndarray, 長さ n_variations
            complete: np.ndarray
        """
        try:
            # 3D点群を読み込み
            points = self.load_kitti_bin(bin_path)
            
            # 前処理
            filtered_points = self.preprocess_3d_points(points)
            
            # 点が少なすぎる場合はスキップ
            if len(filtered_points) < 10:
                return None
            
            # 2D LiDARスキャンに変換
            complete = self.convert_to_2d_lidar(filtered_points)
            
            # 有効な点が少なすぎる場合はスキップ
            valid_points = np.sum(complete > 0)
            if valid_points < 50:  # 最低限の点数を確保
                return None
            
            # 複数の欠損パターンを生成
            partial_list = []
            for i in range(self.n_variations):
                partial = self.generate_occlusion(complete)
                partial_list.append(partial)
            
            return partial_list, complete
            
        except Exception as e:
            print(f"Error processing {bin_path}: {e}")
            return None
    
    def validate_sample(
        self, 
        partial: np.ndarray, 
        complete: np.ndarray
    ) -> bool:
        """
        生成されたサンプルの品質を検証
        
        Args:
            partial: shape (n_beams,)
            complete: shape (n_beams,)
            
        Returns:
            valid: 品質基準を満たすか
        """
        # NaN / Inf チェック
        if np.any(np.isnan(partial)) or np.any(np.isnan(complete)):
            return False
        if np.any(np.isinf(partial)) or np.any(np.isinf(complete)):
            return False
        
        # shape チェック
        if partial.shape != (self.n_beams,) or complete.shape != (self.n_beams,):
            return False
        
        # partial は complete の部分集合であることを確認
        # (partial > 0 の場所では partial == complete であるべき)
        partial_mask = partial > 0
        if not np.allclose(partial[partial_mask], complete[partial_mask], rtol=1e-5):
            return False
        
        # 欠損率チェック（0より大きい必要がある）
        occlusion_rate = np.sum(partial == 0) / self.n_beams
        if occlusion_rate < 0.1 or occlusion_rate > 0.9:  # 極端な欠損率を除外
            return False
        
        return True
    
    def generate_dataset(
        self,
        kitti_root: str,
        output_root: str,
        max_samples: Optional[int] = None,
        use_training: bool = True
    ):
        """
        データセット全体を生成（複数の欠損パターンを生成）
        
        Args:
            kitti_root: KITTIデータセットのルートディレクトリ
            output_root: 出力ディレクトリ
            max_samples: 最大サンプル数（Noneの場合は全て処理）
            use_training: Trueならvelodyne/training、Falseならvelodyne/testing
        """
        # 入力ディレクトリの確認（複数パターンに対応）
        velodyne_dir = None
        
        # パターン1: velodyne/training または velodyne/testing
        if use_training:
            candidate = os.path.join(kitti_root, "velodyne", "training")
        else:
            candidate = os.path.join(kitti_root, "velodyne", "testing")
        
        if os.path.exists(candidate):
            velodyne_dir = candidate
        else:
            # パターン2: training/velodyne または testing/velodyne（旧構造）
            if use_training:
                candidate = os.path.join(kitti_root, "training", "velodyne")
            else:
                candidate = os.path.join(kitti_root, "testing", "velodyne")
            
            if os.path.exists(candidate):
                velodyne_dir = candidate
        
        if velodyne_dir is None:
            raise ValueError(
                f"Velodyne directory not found. Tried:\n"
                f"  - {os.path.join(kitti_root, 'velodyne', 'training' if use_training else 'testing')}\n"
                f"  - {os.path.join(kitti_root, 'training' if use_training else 'testing', 'velodyne')}"
            )
        
        print(f"Using velodyne directory: {velodyne_dir}")
        
        # .binファイルのリストを取得
        bin_files = sorted(glob.glob(os.path.join(velodyne_dir, "*.bin")))
        
        if len(bin_files) == 0:
            raise ValueError(f"No .bin files found in {velodyne_dir}")
        
        print(f"Found {len(bin_files)} .bin files")
        print(f"Generating {self.n_variations} variations per frame")
        
        if max_samples is not None:
            bin_files = bin_files[:max_samples]
            print(f"Processing first {max_samples} files")
        
        # 出力ディレクトリを作成
        train_dir = os.path.join(output_root, "train")
        val_dir = os.path.join(output_root, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        # データセットを生成
        train_count = 0
        val_count = 0
        skipped_count = 0
        
        print("Generating dataset...")
        for bin_path in tqdm(bin_files):
            # フレームを処理（複数のpartialを生成）
            result = self.process_single_frame(bin_path)
            
            if result is None:
                skipped_count += 1
                continue
            
            partial_list, complete = result
            
            # 各variationを個別のサンプルとして保存
            # すべてのパターンが有効かチェック
            all_valid = True
            for partial in partial_list:
                if not self.validate_sample(partial, complete):
                    all_valid = False
                    break
            
            if not all_valid:
                skipped_count += 1
                continue
            
            # train/val に分割（フレーム単位で）
            if np.random.random() < self.train_ratio:
                output_dir = train_dir
                sample_id = f"{train_count:06d}"
                train_count += 1
            else:
                output_dir = val_dir
                sample_id = f"{val_count:06d}"
                val_count += 1
            
            # 1つの.npzファイルに complete と複数の partial を保存
            output_path = os.path.join(output_dir, f"{sample_id}.npz")
            save_dict = {
                "complete": complete
            }
            # partial1, partial2, ..., partial5 として保存
            for i, partial in enumerate(partial_list, start=1):
                save_dict[f"partial{i}"] = partial
            
            np.savez_compressed(output_path, **save_dict)
        
        print(f"\nDataset generation completed!")
        print(f"  Train samples: {train_count}")
        print(f"  Val samples: {val_count}")
        print(f"  Skipped: {skipped_count}")
        print(f"  Total processed: {train_count + val_count}")
        print(f"  Original frames: {len(bin_files)}")
        print(f"  Variations per frame: {self.n_variations}")
        print(f"  Expected samples: {len(bin_files) * self.n_variations}")
        
        # 統計情報を保存
        stats = {
            "n_beams": self.n_beams,
            "r_max": self.r_max,
            "z_min": self.z_min,
            "occlusion_min": self.occlusion_min,
            "occlusion_max": self.occlusion_max,
            "n_variations": self.n_variations,
            "train_count": train_count,
            "val_count": val_count,
            "skipped_count": skipped_count
        }
        
        
        # 統計情報を保存
        stats = {
            "n_beams": self.n_beams,
            "r_max": self.r_max,
            "z_min": self.z_min,
            "occlusion_min": self.occlusion_min,
            "occlusion_max": self.occlusion_max,
            "n_variations": self.n_variations,
            "train_count": train_count,
            "val_count": val_count,
            "skipped_count": skipped_count
        }
        
        stats_path = os.path.join(output_root, "dataset_stats.npz")
        np.savez(stats_path, **stats)
        print(f"\nStatistics saved to: {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate 2D LiDAR Point Completion Dataset from KITTI"
    )
    
    # 必須引数
    parser.add_argument(
        "--kitti_root",
        type=str,
        required=True,
        help="Path to KITTI dataset root directory"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Path to output directory"
    )
    
    # オプション引数
    parser.add_argument(
        "--n_beams",
        type=int,
        default=360,
        help="Number of LiDAR beams (default: 360)"
    )
    parser.add_argument(
        "--r_max",
        type=float,
        default=30.0,
        help="Maximum distance in meters (default: 30.0)"
    )
    parser.add_argument(
        "--z_min",
        type=float,
        default=-1.5,
        help="Minimum z threshold for ground removal (default: -1.5)"
    )
    parser.add_argument(
        "--occlusion_min",
        type=float,
        default=0.2,
        help="Minimum occlusion ratio (default: 0.2)"
    )
    parser.add_argument(
        "--occlusion_max",
        type=float,
        default=0.5,
        help="Maximum occlusion ratio (default: 0.5)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Train/val split ratio (default: 0.9)"
    )
    parser.add_argument(
        "--n_variations",
        type=int,
        default=5,
        help="Number of occlusion variations per frame (default: 5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: None = all)"
    )
    parser.add_argument(
        "--use_testing",
        action="store_true",
        help="Use velodyne/testing instead of velodyne/training"
    )
    
    args = parser.parse_args()
    
    # データセット生成器を初期化
    converter = KITTIto2DLiDARConverter(
        n_beams=args.n_beams,
        r_max=args.r_max,
        z_min=args.z_min,
        occlusion_min=args.occlusion_min,
        occlusion_max=args.occlusion_max,
        train_ratio=args.train_ratio,
        n_variations=args.n_variations,
        seed=args.seed
    )
    
    # データセットを生成
    converter.generate_dataset(
        kitti_root=args.kitti_root,
        output_root=args.output_root,
        max_samples=args.max_samples,
        use_training=not args.use_testing  # use_testingがFalseならtraining使用
    )


if __name__ == "__main__":
    main()