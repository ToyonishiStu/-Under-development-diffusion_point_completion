#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diffusion v2: Dual Encoder U-Net with FiLM conditioning for 1D LiDAR Completion

Architecture improvements over v1:
- Dual Encoder: separates observation path (partial, mask) from noisy input path (x_t, mask)
- FiLMResBlock: GroupNorm -> FiLM(t) -> SiLU -> Conv1d residual block
- FiLM time embedding: scale/shift per block instead of additive injection

Channel flow (base_channels=64):
  Encoder_obs  (partial, mask) [2ch] -> ch -> 2ch -> 4ch  (skips not used in decoder)
  Encoder_noisy (x_t, mask)   [2ch] -> ch -> 2ch -> 4ch  (skips used in decoder)
  Bottleneck: cat(8ch) -> ResBlocks -> 4ch
  Decoder: 4ch -> 2ch -> ch (using noisy encoder skips)
  Output: GroupNorm -> SiLU -> Conv1d(ch->1)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) integer timesteps
        Returns:
            (B, dim) sinusoidal embeddings
        """
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation layer.
    Projects time embedding to per-channel scale and shift.

    Args:
        time_emb_dim: Dimension of time embedding input
        out_ch: Number of channels to modulate
    """

    def __init__(self, time_emb_dim: int, out_ch: int):
        super().__init__()
        self.proj = nn.Linear(time_emb_dim, 2 * out_ch)

    def forward(self, h: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, out_ch, L) feature map
            t_emb: (B, time_emb_dim) time embedding
        Returns:
            (B, out_ch, L) modulated feature map
        """
        params = self.proj(t_emb)          # (B, 2*out_ch)
        scale, shift = params.chunk(2, dim=-1)  # each (B, out_ch)
        scale = scale.unsqueeze(-1)        # (B, out_ch, 1)
        shift = shift.unsqueeze(-1)        # (B, out_ch, 1)
        return h * (1.0 + scale) + shift


class FiLMResBlock(nn.Module):
    """
    Residual block with FiLM conditioning.

    Structure: Conv -> GN -> FiLM(t) -> SiLU -> Conv -> GN + skip

    Args:
        in_ch: Input channels
        out_ch: Output channels
        time_emb_dim: Time embedding dimension
        kernel_size: Convolution kernel size
        num_groups: Number of groups for GroupNorm
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        time_emb_dim: int = 128,
        kernel_size: int = 5,
        num_groups: int = 8,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(min(num_groups, out_ch), out_ch)
        self.film = FiLMLayer(time_emb_dim, out_ch)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(min(num_groups, out_ch), out_ch)

        if in_ch != out_ch:
            self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_ch, L)
            t_emb: (B, time_emb_dim)
        Returns:
            (B, out_ch, L)
        """
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.film(h, t_emb)
        h = self.act1(h)

        h = self.conv2(h)
        h = self.norm2(h)

        return h + self.skip(x)


class Downsample1D(nn.Module):
    """Spatial downsampling by factor 2 using strided convolution."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    """Spatial upsampling by factor 2 using transposed convolution."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SingleEncoder(nn.Module):
    """
    3-level encoder with 2-channel input and skip connection storage.

    Levels:
      0: ch  -> ch,  skip shape (B, ch,  L/1); downsample -> L/2
      1: ch  -> 2ch, skip shape (B, 2ch, L/2); downsample -> L/4
      2: 2ch -> 4ch, skip shape (B, 4ch, L/4); downsample -> L/8

    Args:
        base_ch: Base channel count (ch)
        time_emb_dim: Time embedding dimension
        num_res_blocks: Number of FiLMResBlocks per encoder level
        num_groups: GroupNorm groups
    """

    def __init__(
        self,
        base_ch: int,
        time_emb_dim: int = 128,
        num_res_blocks: int = 2,
        num_groups: int = 8,
    ):
        super().__init__()
        self.base_ch = base_ch
        self.num_res_blocks = num_res_blocks

        # Input conv: 2ch -> base_ch
        self.input_conv = nn.Conv1d(2, base_ch, kernel_size=5, padding=2)

        channel_mults = [1, 2, 4]
        self.levels = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        in_ch = base_ch
        for mult in channel_mults:
            out_ch = base_ch * mult
            level_blocks = nn.ModuleList()
            for i in range(num_res_blocks):
                block_in = in_ch if i == 0 else out_ch
                level_blocks.append(
                    FiLMResBlock(block_in, out_ch, time_emb_dim, num_groups=num_groups)
                )
            self.levels.append(level_blocks)
            self.downsamples.append(Downsample1D(out_ch))
            in_ch = out_ch

        self.out_ch = in_ch  # 4 * base_ch

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        """
        Args:
            x: (B, 2, L) two-channel input
            t_emb: (B, time_emb_dim)
        Returns:
            bottom: (B, 4*base_ch, L/8) bottom feature map
            skips: list of 3 tensors [(B, ch, L), (B, 2ch, L/2), (B, 4ch, L/4)]
        """
        h = self.input_conv(x)  # (B, base_ch, L)

        skips = []
        for level_blocks, downsample in zip(self.levels, self.downsamples):
            for block in level_blocks:
                h = block(h, t_emb)
            skips.append(h)
            h = downsample(h)

        return h, skips


class DualEncoderDDPMUNet(nn.Module):
    """
    Dual Encoder U-Net for Conditional DDPM with FiLM conditioning.

    Two separate encoders:
      - obs_encoder: processes (partial, mask) — observation path
      - noisy_encoder: processes (x_t, mask) — noisy input path

    Only noisy_encoder skips are used in the decoder.
    Both encoder bottoms are concatenated and fed to the bottleneck.

    Args:
        input_length: Sequence length (default 360)
        base_channels: Base channel count (default 64)
        num_res_blocks: ResBlocks per encoder level (default 2)
        time_dim: Time embedding dimension (default 128)
        num_groups: GroupNorm groups (default 8)
    """

    def __init__(
        self,
        input_length: int = 360,
        base_channels: int = 64,
        num_res_blocks: int = 2,
        time_dim: int = 128,
        num_groups: int = 8,
    ):
        super().__init__()
        self.input_length = input_length
        ch = base_channels

        # Time embedding: sinusoidal -> MLP
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Dual encoders
        self.obs_encoder = SingleEncoder(ch, time_dim, num_res_blocks, num_groups)
        self.noisy_encoder = SingleEncoder(ch, time_dim, num_res_blocks, num_groups)

        # Bottleneck: cat(obs_bottom, noisy_bottom) = 8ch -> ResBlocks -> 4ch
        bottleneck_in = 8 * ch
        bottleneck_out = 4 * ch
        self.bottleneck = nn.ModuleList([
            FiLMResBlock(bottleneck_in, bottleneck_out, time_dim, num_groups=num_groups),
            FiLMResBlock(bottleneck_out, bottleneck_out, time_dim, num_groups=num_groups),
        ])

        # Decoder levels (reverse: level 2 -> 1 -> 0)
        # Level 2: up(4ch) + skip(4ch) = 8ch -> 4ch
        # Level 1: up(4ch) + skip(2ch) = 6ch -> 2ch
        # Level 0: up(2ch) + skip(ch)  = 3ch -> ch
        decoder_configs = [
            (bottleneck_out, 4 * ch, 4 * ch),   # (upsample_in, skip_ch, out_ch)
            (4 * ch,         2 * ch, 2 * ch),
            (2 * ch,         ch,     ch),
        ]

        self.upsamples = nn.ModuleList()
        self.decoder_levels = nn.ModuleList()

        for upsample_in, skip_ch, out_ch in decoder_configs:
            self.upsamples.append(Upsample1D(upsample_in))
            concat_ch = upsample_in + skip_ch
            level_blocks = nn.ModuleList([
                FiLMResBlock(concat_ch, out_ch, time_dim, num_groups=num_groups),
            ])
            for _ in range(num_res_blocks - 1):
                level_blocks.append(
                    FiLMResBlock(out_ch, out_ch, time_dim, num_groups=num_groups)
                )
            self.decoder_levels.append(level_blocks)

        # Output head
        final_ch = ch
        self.output_norm = nn.GroupNorm(min(num_groups, final_ch), final_ch)
        self.output_act = nn.SiLU()
        self.output_conv = nn.Conv1d(final_ch, 1, kernel_size=1)

    def forward(
        self,
        x_t: torch.Tensor,
        partial: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict noise epsilon given noisy input, conditioning, and timestep.

        Args:
            x_t: (B, 360) noisy data at timestep t
            partial: (B, 360) observed partial data
            mask: (B, 360) observation mask (1=observed, 0=missing)
            t: (B,) timestep indices

        Returns:
            epsilon_pred: (B, 360) predicted noise
        """
        # Time embedding
        t_emb = self.time_embed(t)  # (B, time_dim)

        # Build 2-channel inputs
        obs_input = torch.stack([partial, mask], dim=1)      # (B, 2, 360)
        noisy_input = torch.stack([x_t, mask], dim=1)        # (B, 2, 360)

        # Encode both paths
        obs_bottom, _obs_skips = self.obs_encoder(obs_input, t_emb)
        noisy_bottom, noisy_skips = self.noisy_encoder(noisy_input, t_emb)

        # Bottleneck: concatenate both bottoms
        h = torch.cat([obs_bottom, noisy_bottom], dim=1)  # (B, 8ch, L/8)
        for block in self.bottleneck:
            h = block(h, t_emb)

        # Decoder with noisy encoder skips (reverse order: level 2 -> 1 -> 0)
        for upsample, level_blocks, skip in zip(
            self.upsamples, self.decoder_levels, reversed(noisy_skips)
        ):
            h = upsample(h)

            # Pad if spatial size mismatches
            if h.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - h.shape[-1]
                h = F.pad(h, (0, diff))

            h = torch.cat([h, skip], dim=1)

            for block in level_blocks:
                h = block(h, t_emb)

        # Output
        h = self.output_norm(h)
        h = self.output_act(h)
        h = self.output_conv(h)   # (B, 1, 360)

        return h.squeeze(1)       # (B, 360)
