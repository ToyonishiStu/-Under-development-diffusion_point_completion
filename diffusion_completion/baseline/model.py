#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D LiDAR Point Completion Model
Conv1D Encoder-Decoder with Dilated Bottleneck

このモデルは欠損を含む2D LiDAR距離列から完全な距離列を復元します。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1DBlock(nn.Module):
    """
    Conv1D + Normalization + ReLU のブロック
    
    すべての Conv1D は stride=1 で、padding を適切に設定し長さを維持
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        dilation: int = 1,
        use_groupnorm: bool = True,
        num_groups: int = 8
    ):
        """
        Args:
            in_channels: 入力チャンネル数
            out_channels: 出力チャンネル数
            kernel_size: カーネルサイズ
            dilation: dilation rate
            use_groupnorm: GroupNorm を使用するか（False の場合 BatchNorm）
            num_groups: GroupNorm のグループ数
        """
        super().__init__()
        
        # padding を計算して長さを維持
        # padding = (kernel_size - 1) * dilation // 2
        # より正確には: padding = dilation * (kernel_size - 1) // 2
        padding = dilation * (kernel_size - 1) // 2
        
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation
        )
        
        # 正規化層の選択
        if use_groupnorm:
            # GroupNorm: バッチサイズに依存しない
            self.norm = nn.GroupNorm(
                num_groups=min(num_groups, out_channels),
                num_channels=out_channels
            )
        else:
            # BatchNorm
            self.norm = nn.BatchNorm1d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        Args:
            x: (B, C_in, L)
        Returns:
            (B, C_out, L)
        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class LiDARCompletionModel(nn.Module):
    """
    2D LiDAR 距離列補完モデル
    
    構成:
        - Encoder (6層): 局所特徴抽出
        - Bottleneck (5層): Dilated Conv で広い受容野を確保
        - Decoder (6層): 距離列復元
    
    入力:
        - partial: (B, 360) 欠損あり距離列
        - mask: (B, 360) 観測マスク
    
    出力:
        - pred: (B, 360) 補完された距離列
    """
    
    def __init__(
        self,
        input_length: int = 360,
        use_groupnorm: bool = True,
        use_sigmoid: bool = False
    ):
        """
        Args:
            input_length: 入力系列長（デフォルト: 360）
            use_groupnorm: GroupNorm を使用するか
            use_sigmoid: 出力層に Sigmoid を適用するか
                        (True: 出力を [0,1] に明示的に制限)
                        (False: 損失関数と正規化に依存)
        """
        super().__init__()
        
        self.input_length = input_length
        self.use_sigmoid = use_sigmoid
        
        # ==========================================
        # Encoder (6層) - 局所特徴抽出
        # ==========================================
        # すべて kernel_size=5, dilation=1
        
        # Block 1-2: 2 → 64 → 64
        self.enc_block1 = Conv1DBlock(2, 64, kernel_size=5, dilation=1, 
                                      use_groupnorm=use_groupnorm, num_groups=8)
        self.enc_block2 = Conv1DBlock(64, 64, kernel_size=5, dilation=1,
                                      use_groupnorm=use_groupnorm, num_groups=8)
        
        # Block 3-4: 64 → 128 → 128
        self.enc_block3 = Conv1DBlock(64, 128, kernel_size=5, dilation=1,
                                      use_groupnorm=use_groupnorm, num_groups=8)
        self.enc_block4 = Conv1DBlock(128, 128, kernel_size=5, dilation=1,
                                      use_groupnorm=use_groupnorm, num_groups=16)
        
        # Block 5-6: 128 → 256 → 256
        self.enc_block5 = Conv1DBlock(128, 256, kernel_size=5, dilation=1,
                                      use_groupnorm=use_groupnorm, num_groups=16)
        self.enc_block6 = Conv1DBlock(256, 256, kernel_size=5, dilation=1,
                                      use_groupnorm=use_groupnorm, num_groups=32)
        
        # ==========================================
        # Bottleneck (5層) - Dilated Conv
        # ==========================================
        # 目的: 長距離の連続欠損（最大180 beam）を捉える
        # すべて kernel_size=5, dilation を指数的に増加
        
        # Block 7: dilation=1
        self.bottleneck1 = Conv1DBlock(256, 512, kernel_size=5, dilation=1,
                                       use_groupnorm=use_groupnorm, num_groups=32)
        # Block 8: dilation=2
        self.bottleneck2 = Conv1DBlock(512, 512, kernel_size=5, dilation=2,
                                       use_groupnorm=use_groupnorm, num_groups=32)
        # Block 9: dilation=4
        self.bottleneck3 = Conv1DBlock(512, 512, kernel_size=5, dilation=4,
                                       use_groupnorm=use_groupnorm, num_groups=32)
        # Block 10: dilation=8
        self.bottleneck4 = Conv1DBlock(512, 512, kernel_size=5, dilation=8,
                                       use_groupnorm=use_groupnorm, num_groups=32)
        # Block 11: dilation=16
        self.bottleneck5 = Conv1DBlock(512, 512, kernel_size=5, dilation=16,
                                       use_groupnorm=use_groupnorm, num_groups=32)
        
        # ==========================================
        # Decoder (6層) - 復元
        # ==========================================
        # すべて kernel_size=5, dilation=1
        
        # Block 12-13: 512 → 256 → 256
        self.dec_block1 = Conv1DBlock(512, 256, kernel_size=5, dilation=1,
                                      use_groupnorm=use_groupnorm, num_groups=32)
        self.dec_block2 = Conv1DBlock(256, 256, kernel_size=5, dilation=1,
                                      use_groupnorm=use_groupnorm, num_groups=32)
        
        # Block 14-15: 256 → 128 → 128
        self.dec_block3 = Conv1DBlock(256, 128, kernel_size=5, dilation=1,
                                      use_groupnorm=use_groupnorm, num_groups=16)
        self.dec_block4 = Conv1DBlock(128, 128, kernel_size=5, dilation=1,
                                      use_groupnorm=use_groupnorm, num_groups=16)
        
        # Block 16: 128 → 64
        self.dec_block5 = Conv1DBlock(128, 64, kernel_size=5, dilation=1,
                                      use_groupnorm=use_groupnorm, num_groups=8)
        
        # Block 17: 64 → 1 (最終出力層)
        # 最終層は Conv のみ（Norm + ReLU なし）
        padding = (5 - 1) // 2
        self.output_conv = nn.Conv1d(
            in_channels=64,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=padding
        )
        
        # Sigmoid を使用する場合の理由:
        # 出力を [0, 1] の範囲に明示的に制限することで、
        # 正規化された距離値の範囲を保証し、学習の安定性を向上
        if self.use_sigmoid:
            self.output_activation = nn.Sigmoid()
    
    def forward(self, partial, mask):
        """
        Forward pass
        
        Args:
            partial: (B, 360) 欠損あり距離列
            mask: (B, 360) 観測マスク (1=観測, 0=欠損)
        
        Returns:
            pred: (B, 360) 補完された距離列
        """
        # partial と mask をチャンネル方向に結合
        # (B, 360) → (B, 1, 360) → (B, 2, 360)
        partial = partial.unsqueeze(1)  # (B, 1, 360)
        mask = mask.unsqueeze(1)        # (B, 1, 360)
        x = torch.cat([partial, mask], dim=1)  # (B, 2, 360)
        
        # ==========================================
        # Encoder
        # ==========================================
        x = self.enc_block1(x)   # (B, 64, 360)
        x = self.enc_block2(x)   # (B, 64, 360)
        
        x = self.enc_block3(x)   # (B, 128, 360)
        x = self.enc_block4(x)   # (B, 128, 360)
        
        x = self.enc_block5(x)   # (B, 256, 360)
        x = self.enc_block6(x)   # (B, 256, 360)
        
        # ==========================================
        # Bottleneck (Dilated Conv)
        # ==========================================
        x = self.bottleneck1(x)  # (B, 512, 360)
        x = self.bottleneck2(x)  # (B, 512, 360)
        x = self.bottleneck3(x)  # (B, 512, 360)
        x = self.bottleneck4(x)  # (B, 512, 360)
        x = self.bottleneck5(x)  # (B, 512, 360)
        
        # ==========================================
        # Decoder
        # ==========================================
        x = self.dec_block1(x)   # (B, 256, 360)
        x = self.dec_block2(x)   # (B, 256, 360)
        
        x = self.dec_block3(x)   # (B, 128, 360)
        x = self.dec_block4(x)   # (B, 128, 360)
        
        x = self.dec_block5(x)   # (B, 64, 360)
        
        # 最終出力
        x = self.output_conv(x)  # (B, 1, 360)
        
        # Sigmoid 適用（オプション）
        if self.use_sigmoid:
            x = self.output_activation(x)
        
        # (B, 1, 360) → (B, 360)
        pred = x.squeeze(1)
        
        return pred


def test_model():
    """
    モデルの動作確認
    """
    print("=" * 70)
    print("Model Architecture Test")
    print("=" * 70)
    
    # モデルを作成
    model = LiDARCompletionModel(
        input_length=360,
        use_groupnorm=True,
        use_sigmoid=True
    )
    
    # パラメータ数を表示
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # ダミーデータで forward テスト
    batch_size = 4
    seq_len = 360
    
    partial = torch.randn(batch_size, seq_len)
    mask = torch.randint(0, 2, (batch_size, seq_len)).float()
    
    print(f"\nInput shapes:")
    print(f"  partial: {partial.shape}")
    print(f"  mask: {mask.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        pred = model(partial, mask)
    
    print(f"\nOutput shape:")
    print(f"  pred: {pred.shape}")
    print(f"  pred range: [{pred.min():.3f}, {pred.max():.3f}]")
    
    # 正しい shape かチェック
    assert pred.shape == (batch_size, seq_len), f"Expected shape {(batch_size, seq_len)}, got {pred.shape}"
    
    print("\n✓ Model test passed!")
    
    return model


if __name__ == "__main__":
    test_model()