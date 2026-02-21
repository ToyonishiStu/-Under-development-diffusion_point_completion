#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D LiDAR Point Completion - Training Script

Conv baseline学習スクリプト（fill_mode対応）
"""

import argparse
import os
import sys
from pathlib import Path
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 同一ディレクトリからインポート
from dataset import create_dataloaders
from model import LiDARCompletionModel
from metrics import compute_batch_metrics


class Trainer:
    """学習を管理するクラス"""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device: str = "cuda",
        checkpoint_dir: str = "./checkpoints",
        log_dir: str = "./logs",
        save_every: int = 10,
        use_schedulefree: bool = False
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.save_every = save_every
        self.use_schedulefree = use_schedulefree

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_mae_missing": [],
            "best_val_loss": float("inf"),
            "best_epoch": 0
        }

    def train_epoch(self, epoch: int):
        """1エポックの学習"""
        if self.use_schedulefree:
            self.optimizer.train()

        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, (partial, mask, target) in enumerate(pbar):
            partial = partial.to(self.device)
            mask = mask.to(self.device)
            target = target.to(self.device)

            pred = self.model(partial, mask)
            loss = self.criterion(pred, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar("Train/BatchLoss", loss.item(), global_step)

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate(self, epoch: int):
        """検証"""
        if self.use_schedulefree:
            self.optimizer.eval()

        self.model.eval()
        total_loss = 0.0
        all_mae_missing = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            for partial, mask, target in pbar:
                partial = partial.to(self.device)
                mask = mask.to(self.device)
                target = target.to(self.device)

                pred = self.model(partial, mask)
                loss = self.criterion(pred, target)

                total_loss += loss.item()

                # バッチメトリクスを計算
                batch_metrics = compute_batch_metrics(pred, target, mask)
                all_mae_missing.extend(batch_metrics['mae_missing'].tolist())

                pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(self.val_loader)
        avg_mae_missing = np.mean(all_mae_missing)

        return avg_loss, avg_mae_missing

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """チェックポイントを保存"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history
        }

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"  Saved best model: {best_path}")

    def train(self, num_epochs: int, scheduler=None):
        """学習ループ"""
        print("\n" + "=" * 70)
        print("Starting Training")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print("=" * 70 + "\n")

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_mae_missing = self.validate(epoch)

            if scheduler is not None and not self.use_schedulefree:
                scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("Train/LearningRate", current_lr, epoch)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_mae_missing"].append(val_mae_missing)

            self.writer.add_scalar("Train/EpochLoss", train_loss, epoch)
            self.writer.add_scalar("Val/EpochLoss", val_loss, epoch)
            self.writer.add_scalar("Val/MAE_Missing", val_mae_missing, epoch)

            is_best = val_loss < self.history["best_val_loss"]
            if is_best:
                self.history["best_val_loss"] = val_loss
                self.history["best_epoch"] = epoch

            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  Val MAE (missing): {val_mae_missing:.6f}")
            if is_best:
                print(f"  Best model so far!")
            print(f"  Best Val Loss: {self.history['best_val_loss']:.6f} (Epoch {self.history['best_epoch']})")

            if epoch % self.save_every == 0 or is_best or epoch == num_epochs:
                self.save_checkpoint(epoch, is_best=is_best)

        print("\n" + "=" * 70)
        print("Training Completed!")
        print("=" * 70)
        print(f"Best Val Loss: {self.history['best_val_loss']:.6f} (Epoch {self.history['best_epoch']})")

        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Saved training history: {history_path}")

        self.writer.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train Conv Baseline for 2D LiDAR Point Completion"
    )

    # データセット関連
    parser.add_argument("--train_dirs", type=str, nargs="+", required=True)
    parser.add_argument("--val_dirs", type=str, nargs="+", required=True)
    parser.add_argument("--n_variations", type=int, default=5)

    # fill_mode関連
    parser.add_argument("--fill_mode", type=str, choices=["zero", "noise"], default="zero")
    parser.add_argument("--noise_std", type=float, default=1.0)

    # モデル関連
    parser.add_argument("--use_groupnorm", action="store_true", default=True)
    parser.add_argument("--use_sigmoid", action="store_true", default=True)

    # 学習関連
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)

    # 保存関連
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./experiments")
    parser.add_argument("--save_every", type=int, default=10)

    # その他
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()

    # 乱数シード設定
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 実験ディレクトリ
    experiment_dir = Path(args.output_dir) / args.experiment_name
    checkpoint_dir = experiment_dir / "checkpoints"
    log_dir = experiment_dir / "logs"

    experiment_dir.mkdir(parents=True, exist_ok=True)

    # 設定を保存
    config = vars(args)
    config_path = experiment_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config: {config_path}")

    # データローダーを作成
    print("\n" + "=" * 70)
    print("Loading Data")
    print("=" * 70)

    train_loader, val_loader = create_dataloaders(
        train_dirs=args.train_dirs,
        val_dirs=args.val_dirs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=True,
        augment_train=True,
        n_variations=args.n_variations,
        fill_mode=args.fill_mode,
        noise_std=args.noise_std
    )

    # モデルを作成
    print("\n" + "=" * 70)
    print("Creating Model")
    print("=" * 70)

    model = LiDARCompletionModel(
        input_length=360,
        use_groupnorm=args.use_groupnorm,
        use_sigmoid=args.use_sigmoid
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # 損失関数（L1 Loss）
    criterion = nn.L1Loss()

    # 最適化器
    try:
        from schedulefree import RAdamScheduleFree
        optimizer = RAdamScheduleFree(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        print("Using RAdam ScheduleFree optimizer")
        use_schedulefree = True
    except ImportError:
        print("schedulefree not installed, using Adam")
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        use_schedulefree = False

    scheduler = None
    if not use_schedulefree:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10
        )

    # チェックポイントから再開
    start_epoch = 1
    if args.resume:
        print(f"\nLoading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    # Trainer を作成
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=args.device,
        checkpoint_dir=str(checkpoint_dir),
        log_dir=str(log_dir),
        save_every=args.save_every,
        use_schedulefree=use_schedulefree
    )

    # 学習情報を表示
    print("\n" + "=" * 70)
    print("Training Configuration")
    print("=" * 70)
    print(f"Experiment: {args.experiment_name}")
    print(f"Fill mode: {args.fill_mode}")
    if args.fill_mode == "noise":
        print(f"Noise std: {args.noise_std}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Seed: {args.seed}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Device: {args.device}")
    print(f"Output dir: {experiment_dir}")

    # 学習開始
    trainer.train(num_epochs=args.epochs, scheduler=scheduler)


if __name__ == "__main__":
    main()
