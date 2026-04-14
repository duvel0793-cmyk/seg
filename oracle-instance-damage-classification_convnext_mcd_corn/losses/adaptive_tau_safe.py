from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_safe_tau(logits: torch.Tensor, tau: torch.Tensor, tau_min: float, tau_max: float) -> torch.Tensor:
    tau = tau.clamp(float(tau_min), float(tau_max))
    return logits / tau.unsqueeze(1)


def build_soft_class_targets(labels: torch.Tensor, tau: torch.Tensor, num_classes: int) -> torch.Tensor:
    labels = labels.long().view(-1)
    tau = tau.view(-1, 1).clamp_min(1e-4)
    class_positions = torch.arange(num_classes, device=labels.device, dtype=torch.float32).view(1, -1)
    centers = labels.float().view(-1, 1)
    logits = -((class_positions - centers).abs() / tau)
    return torch.softmax(logits, dim=1)


def soft_cross_entropy_from_probabilities(probabilities: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    probabilities = probabilities.clamp_min(1e-8)
    return -(soft_targets * probabilities.log()).sum(dim=1).mean()


class AdaptiveTauSafeRegularizer(nn.Module):
    def __init__(self, tau_min: float = 0.85, tau_max: float = 1.15, tau_center: float = 1.0) -> None:
        super().__init__()
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.tau_center = float(tau_center)

    def forward(self, tau: torch.Tensor) -> dict[str, Any]:
        tau = tau.float()
        center_penalty = (tau - self.tau_center).pow(2).mean()
        spread_penalty = tau.std(unbiased=False).pow(2) if tau.numel() > 1 else tau.new_zeros(())
        lower_penalty = F.relu(self.tau_min - tau).pow(2).mean()
        upper_penalty = F.relu(tau - self.tau_max).pow(2).mean()
        total = center_penalty + (0.1 * spread_penalty) + lower_penalty + upper_penalty
        return {
            "loss_tau_reg": total,
            "loss_tau_center": center_penalty,
            "loss_tau_spread": spread_penalty,
            "loss_tau_bounds": lower_penalty + upper_penalty,
            "tau_mean": tau.mean(),
            "tau_std": tau.std(unbiased=False) if tau.numel() > 1 else tau.new_zeros(()),
        }

