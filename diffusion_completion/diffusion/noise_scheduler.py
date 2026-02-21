#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Noise Scheduler for Conditional DDPM

Linear beta schedule with forward/reverse diffusion processes.
Implements masked diffusion: noise is only applied to missing regions,
while observed regions are kept fixed from the partial input.
"""

import torch


class NoiseScheduler:
    """
    DDPM Noise Scheduler with linear beta schedule.

    Handles forward diffusion (adding noise) and reverse diffusion (denoising),
    with masking support for conditional completion tasks.

    Args:
        T: Number of diffusion timesteps
        beta_start: Starting beta value
        beta_end: Ending beta value
        device: Torch device for tensor allocation
    """

    def __init__(
        self,
        T: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cuda"
    ):
        self.T = T
        self.device = device

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        # Precomputed values for forward diffusion
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

        # Precomputed values for reverse diffusion
        self.alpha_bars_prev = torch.cat(
            [torch.tensor([1.0], device=device), self.alpha_bars[:-1]]
        )

        # Posterior variance: beta_tilde_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_bars_prev) / (1.0 - self.alpha_bars)
        )

        # Posterior mean coefficients
        # mu_tilde_t = coeff_x0 * x_0 + coeff_xt * x_t
        self.posterior_mean_coeff_x0 = (
            self.betas * torch.sqrt(self.alpha_bars_prev) / (1.0 - self.alpha_bars)
        )
        self.posterior_mean_coeff_xt = (
            torch.sqrt(self.alphas) * (1.0 - self.alpha_bars_prev) / (1.0 - self.alpha_bars)
        )

    def _extract(self, tensor: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Extract values from a 1D tensor at timestep indices and reshape for broadcasting.

        Args:
            tensor: (T,) precomputed schedule values
            t: (B,) timestep indices

        Returns:
            (B, 1) values for broadcasting over (B, 360)
        """
        return tensor[t].unsqueeze(-1)

    def forward_diffusion(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor,
        partial: torch.Tensor
    ) -> tuple:
        """
        Forward diffusion process with masking.

        Missing regions (mask==0): x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon
        Observed regions (mask==1): x_t = partial

        Args:
            x_0: (B, 360) clean target data
            t: (B,) timestep indices [0, T-1]
            mask: (B, 360) observation mask (1=observed, 0=missing)
            partial: (B, 360) observed partial data

        Returns:
            x_t: (B, 360) noisy data
            noise: (B, 360) sampled noise (for loss computation)
        """
        noise = torch.randn_like(x_0)

        sqrt_alpha_bar = self._extract(self.sqrt_alpha_bars, t)
        sqrt_one_minus_alpha_bar = self._extract(self.sqrt_one_minus_alpha_bars, t)

        # Forward diffusion on all positions
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

        # Enforce observed regions
        x_t = mask * partial + (1.0 - mask) * x_t

        return x_t, noise

    def reverse_step(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor,
        partial: torch.Tensor
    ) -> torch.Tensor:
        """
        Single reverse diffusion step: x_t -> x_{t-1}

        Args:
            model_output: (B, 360) predicted noise epsilon
            x_t: (B, 360) current noisy data
            t: (B,) current timestep indices
            mask: (B, 360) observation mask
            partial: (B, 360) observed partial data

        Returns:
            x_prev: (B, 360) denoised data at t-1
        """
        # Predict x_0 from noise prediction
        sqrt_alpha_bar = self._extract(self.sqrt_alpha_bars, t)
        sqrt_one_minus_alpha_bar = self._extract(self.sqrt_one_minus_alpha_bars, t)

        x_0_pred = (x_t - sqrt_one_minus_alpha_bar * model_output) / sqrt_alpha_bar
        x_0_pred = x_0_pred.clamp(0.0, 2.0)

        # Compute posterior mean
        coeff_x0 = self._extract(self.posterior_mean_coeff_x0, t)
        coeff_xt = self._extract(self.posterior_mean_coeff_xt, t)
        posterior_mean = coeff_x0 * x_0_pred + coeff_xt * x_t

        # Add noise (except at t=0)
        posterior_var = self._extract(self.posterior_variance, t)
        noise = torch.randn_like(x_t)

        # t == 0 means no noise added
        nonzero_mask = (t > 0).float().unsqueeze(-1)
        x_prev = posterior_mean + nonzero_mask * torch.sqrt(posterior_var) * noise

        # Enforce observed regions
        x_prev = mask * partial + (1.0 - mask) * x_prev

        return x_prev

    def to(self, device: str) -> "NoiseScheduler":
        """Move all tensors to the specified device."""
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self.sqrt_alpha_bars = self.sqrt_alpha_bars.to(device)
        self.sqrt_one_minus_alpha_bars = self.sqrt_one_minus_alpha_bars.to(device)
        self.alpha_bars_prev = self.alpha_bars_prev.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_mean_coeff_x0 = self.posterior_mean_coeff_x0.to(device)
        self.posterior_mean_coeff_xt = self.posterior_mean_coeff_xt.to(device)
        return self
