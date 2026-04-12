"""Regularizer for conservative pixel-wise tau values."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0 or y.numel() == 0:
        return x.new_tensor(0.0)
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = x_centered.pow(2).mean().sqrt() * y_centered.pow(2).mean().sqrt()
    if float(denom.detach().item()) < 1.0e-6:
        return x.new_tensor(0.0)
    return (x_centered * y_centered).mean() / denom


class TauRegularizer(nn.Module):
    def __init__(
        self,
        lambda_tau_mean: float = 0.001,
        lambda_tau_diff: float = 0.01,
        lambda_tau_rank: float = 0.001,
        lambda_raw_tau_center: float = 0.001,
        lambda_raw_tau_bound: float = 0.0005,
        raw_tau_bound: float = 4.0,
        difficulty_bins: int = 5,
    ) -> None:
        super().__init__()
        self.lambda_tau_mean = float(lambda_tau_mean)
        self.lambda_tau_diff = float(lambda_tau_diff)
        self.lambda_tau_rank = float(lambda_tau_rank)
        self.lambda_raw_tau_center = float(lambda_raw_tau_center)
        self.lambda_raw_tau_bound = float(lambda_raw_tau_bound)
        self.raw_tau_bound = float(raw_tau_bound)
        self.difficulty_bins = int(max(difficulty_bins, 1))

    def forward(
        self,
        adaptive_tau: torch.Tensor,
        raw_tau: torch.Tensor,
        target_tau: torch.Tensor,
        difficulty: torch.Tensor,
    ) -> Dict[str, object]:
        if adaptive_tau.numel() == 0 or difficulty.numel() == 0:
            zero = adaptive_tau.new_tensor(0.0)
            return {
                "total": zero,
                "tau_mean_loss": zero,
                "tau_diff_loss": zero,
                "tau_rank_loss": zero,
                "raw_tau_center_loss": zero,
                "raw_tau_bound_loss": zero,
                "corr_tau_difficulty": 0.0,
                "corr_raw_tau_difficulty": 0.0,
                "tau_by_difficulty_bin": [],
            }

        tau_scalar = adaptive_tau.mean(dim=1)
        raw_tau_scalar = raw_tau.mean(dim=1)

        target_tau = target_tau.to(device=tau_scalar.device, dtype=tau_scalar.dtype)
        difficulty = difficulty.to(device=tau_scalar.device, dtype=tau_scalar.dtype)

        tau_mean_loss = (tau_scalar.mean() - target_tau.mean()).pow(2)
        tau_diff_loss = F.smooth_l1_loss(tau_scalar, target_tau, reduction="mean")
        corr_tau = _pearson_corr(tau_scalar, difficulty)
        corr_raw_tau = _pearson_corr(raw_tau_scalar, difficulty)
        tau_rank_loss = 1.0 - corr_tau.clamp(min=-1.0, max=1.0)
        raw_tau_center_loss = raw_tau_scalar.mean().pow(2)
        raw_tau_bound_loss = F.relu(raw_tau_scalar.abs() - self.raw_tau_bound).pow(2).mean()

        total = (
            self.lambda_tau_mean * tau_mean_loss
            + self.lambda_tau_diff * tau_diff_loss
            + self.lambda_tau_rank * tau_rank_loss
            + self.lambda_raw_tau_center * raw_tau_center_loss
            + self.lambda_raw_tau_bound * raw_tau_bound_loss
        )

        tau_by_difficulty_bin = []
        if difficulty.numel() > 0:
            edges = torch.linspace(0.0, 1.0, steps=self.difficulty_bins + 1, device=difficulty.device, dtype=difficulty.dtype)
            for bin_idx in range(self.difficulty_bins):
                left = edges[bin_idx]
                right = edges[bin_idx + 1]
                if bin_idx == self.difficulty_bins - 1:
                    mask = (difficulty >= left) & (difficulty <= right)
                else:
                    mask = (difficulty >= left) & (difficulty < right)
                tau_mean = float(tau_scalar[mask].mean().item()) if mask.any() else 0.0
                tau_by_difficulty_bin.append(
                    {
                        "bin": int(bin_idx),
                        "range": [float(left.item()), float(right.item())],
                        "count": int(mask.sum().item()),
                        "tau_mean": tau_mean,
                    }
                )

        return {
            "total": total,
            "tau_mean_loss": tau_mean_loss,
            "tau_diff_loss": tau_diff_loss,
            "tau_rank_loss": tau_rank_loss,
            "raw_tau_center_loss": raw_tau_center_loss,
            "raw_tau_bound_loss": raw_tau_bound_loss,
            "corr_tau_difficulty": float(corr_tau.detach().item()),
            "corr_raw_tau_difficulty": float(corr_raw_tau.detach().item()),
            "tau_by_difficulty_bin": tau_by_difficulty_bin,
        }

