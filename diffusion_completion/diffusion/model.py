#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conditional DDPM U-Net for 1D LiDAR Completion

1D Conv U-Net with sinusoidal time embedding and skip connections.
Input: concatenation of [x_t, partial, mask] (3 channels).
Output: predicted noise epsilon (1 channel, squeezed to (B, 360)).
"""

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for diffusion timesteps.

    Args:
        dim: Embedding dimension
    """

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


class TimeConditionedResBlock(nn.Module):
    """
    Residual block with time embedding conditioning.

    GroupNorm -> SiLU -> Conv1d -> + time_proj(t_emb) -> GroupNorm -> SiLU -> Conv1d -> + residual

    Args:
        in_ch: Input channels
        out_ch: Output channels
        time_dim: Time embedding dimension
        kernel_size: Convolution kernel size
        num_groups: Number of groups for GroupNorm
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        time_dim: int = 128,
        kernel_size: int = 5,
        num_groups: int = 8
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.norm1 = nn.GroupNorm(min(num_groups, in_ch), in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)

        self.time_proj = nn.Linear(time_dim, out_ch)

        self.norm2 = nn.GroupNorm(min(num_groups, out_ch), out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding)

        # Skip connection (1x1 conv if channel mismatch)
        if in_ch != out_ch:
            self.skip = nn.Conv1d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_ch, L)
            t_emb: (B, time_dim)

        Returns:
            (B, out_ch, L)
        """
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        # Add time embedding (broadcast over spatial dim)
        h = h + self.time_proj(t_emb).unsqueeze(-1)

        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)

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


class ConditionalDDPMUNet(nn.Module):
    """
    1D Conv U-Net for Conditional DDPM.

    Takes concatenated [x_t, partial, mask] as input (3 channels)
    and predicts noise epsilon.

    Architecture:
        Input Conv: 3 -> model_channels
        Encoder: 3 levels with ResBlocks + Downsample
        Bottleneck: 2 ResBlocks
        Decoder: 3 levels with Upsample + cat skip + ResBlocks
        Output: GroupNorm -> SiLU -> Conv1d -> squeeze

    Args:
        input_length: Sequence length (360)
        in_channels: Number of input channels (3: x_t + partial + mask)
        model_channels: Base channel count
        channel_mult: Channel multiplier per encoder level
        time_dim: Time embedding dimension
        num_groups: GroupNorm groups
    """

    def __init__(
        self,
        input_length: int = 360,
        in_channels: int = 3,
        model_channels: int = 64,
        channel_mult: tuple = (1, 2, 4),
        time_dim: int = 128,
        num_groups: int = 8
    ):
        super().__init__()
        self.input_length = input_length
        self.model_channels = model_channels

        # Time embedding: sinusoidal -> MLP
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Input conv: in_channels -> model_channels
        self.input_conv = nn.Conv1d(in_channels, model_channels, kernel_size=5, padding=2)

        # Build encoder, bottleneck, decoder
        ch = model_channels
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        skip_channels = []

        # Encoder levels
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult

            # First ResBlock (handles channel change)
            self.encoder_blocks.append(
                TimeConditionedResBlock(ch, out_ch, time_dim, num_groups=num_groups)
            )
            ch = out_ch

            # Second ResBlock
            self.encoder_blocks.append(
                TimeConditionedResBlock(ch, ch, time_dim, num_groups=num_groups)
            )

            skip_channels.append(ch)
            self.downsamples.append(Downsample1D(ch))

        # Bottleneck
        bottleneck_ch = model_channels * channel_mult[-1] * 2  # 256 -> 512
        self.bottleneck1 = TimeConditionedResBlock(ch, bottleneck_ch, time_dim, num_groups=num_groups)
        self.bottleneck2 = TimeConditionedResBlock(bottleneck_ch, ch, time_dim, num_groups=num_groups)

        # Decoder levels (reverse order)
        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        for level in reversed(range(len(channel_mult))):
            out_ch = model_channels * channel_mult[level]
            skip_ch = skip_channels[level]

            self.upsamples.append(Upsample1D(ch))

            # First ResBlock after concat with skip (ch + skip_ch -> out_ch)
            self.decoder_blocks.append(
                TimeConditionedResBlock(ch + skip_ch, out_ch, time_dim, num_groups=num_groups)
            )

            # Second ResBlock
            self.decoder_blocks.append(
                TimeConditionedResBlock(out_ch, out_ch, time_dim, num_groups=num_groups)
            )

            ch = out_ch

        # Output: GroupNorm -> SiLU -> Conv1d(ch -> 1)
        self.output_norm = nn.GroupNorm(min(num_groups, ch), ch)
        self.output_act = nn.SiLU()
        self.output_conv = nn.Conv1d(ch, 1, kernel_size=1)

    def forward(
        self,
        x_t: torch.Tensor,
        partial: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor
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

        # Concatenate inputs: (B, 3, 360)
        x = torch.stack([x_t, partial, mask], dim=1)

        # Input conv
        h = self.input_conv(x)  # (B, model_channels, 360)

        # Encoder
        skips = []
        block_idx = 0
        for level in range(len(self.downsamples)):
            h = self.encoder_blocks[block_idx](h, t_emb)
            block_idx += 1
            h = self.encoder_blocks[block_idx](h, t_emb)
            block_idx += 1
            skips.append(h)
            h = self.downsamples[level](h)

        # Bottleneck
        h = self.bottleneck1(h, t_emb)
        h = self.bottleneck2(h, t_emb)

        # Decoder
        block_idx = 0
        for level in range(len(self.upsamples)):
            h = self.upsamples[level](h)
            skip = skips.pop()

            # Handle size mismatch from downsampling/upsampling
            if h.shape[-1] != skip.shape[-1]:
                h = nn.functional.pad(h, (0, skip.shape[-1] - h.shape[-1]))

            h = torch.cat([h, skip], dim=1)
            h = self.decoder_blocks[block_idx](h, t_emb)
            block_idx += 1
            h = self.decoder_blocks[block_idx](h, t_emb)
            block_idx += 1

        # Output
        h = self.output_norm(h)
        h = self.output_act(h)
        h = self.output_conv(h)  # (B, 1, 360)

        return h.squeeze(1)  # (B, 360)
