"""Pixel-wise conservative safe tau head for ordinal logits."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SafeTau(nn.Module):
    """Predict bounded pixel-wise tau offsets with staged activation."""

    def __init__(
        self,
        num_ordinal_logits: int,
        feature_channels: int,
        enable_tau: bool = True,
        tau_mode: str = "pixel_corn_safe_v2",
        tau_init: float = 0.0,
        tau_min: float = -0.05,
        tau_max: float = 0.20,
        tau_target: float = 0.03,
        tau_warmup_epochs: int = 0,
        corn_soft_start_epoch: int = 0,
        per_boundary: bool = True,
        hidden_channels: int = 64,
        detach_features: bool = True,
        detach_logits: bool = True,
    ) -> None:
        super().__init__()
        self.enable_tau = bool(enable_tau)
        self.tau_mode = str(tau_mode)
        self.num_ordinal_logits = int(num_ordinal_logits)
        self.tau_min = float(min(tau_min, tau_max))
        self.tau_max = float(max(tau_min, tau_max))
        self.tau_target = float(tau_target)
        self.tau_init = float(tau_init)
        self.tau_warmup_epochs = int(max(tau_warmup_epochs, 0))
        self.corn_soft_start_epoch = int(max(corn_soft_start_epoch, 0))
        self.per_boundary = bool(per_boundary)
        self.hidden_channels = int(hidden_channels)
        self.detach_features = bool(detach_features)
        self.detach_logits = bool(detach_logits)

        tau_channels = self.num_ordinal_logits if self.per_boundary else 1
        in_channels = int(feature_channels) + self.num_ordinal_logits
        self.feature_proj = nn.Sequential(
            nn.Conv2d(int(feature_channels), self.hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.hidden_channels),
            nn.GELU(),
        )
        self.tau_head = nn.Sequential(
            nn.Conv2d(self.hidden_channels + self.num_ordinal_logits, self.hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_channels),
            nn.GELU(),
            nn.Conv2d(self.hidden_channels, tau_channels, kernel_size=1),
        )
        init_tau = min(max(self.tau_init, self.tau_min + 1.0e-4), self.tau_max - 1.0e-4)
        init_ratio = (init_tau - self.tau_min) / max(self.tau_max - self.tau_min, 1.0e-6)
        init_ratio = min(max(init_ratio, 1.0e-4), 1.0 - 1.0e-4)
        nn.init.constant_(self.tau_head[-1].bias, math.log(init_ratio / (1.0 - init_ratio)))
        self.register_buffer(
            "fixed_tau_template",
            torch.full((1, tau_channels, 1, 1), float(self.tau_init if self.tau_warmup_epochs > 0 else self.tau_target)),
            persistent=False,
        )

        if in_channels <= 0:
            raise ValueError("SafeTau received invalid channel configuration.")

    def _phase(self, epoch: int) -> str:
        if not self.enable_tau or self.tau_mode in {"disabled", "none"}:
            return "disabled"
        if epoch < self.tau_warmup_epochs:
            return "fixed_warmup"
        if epoch < self.corn_soft_start_epoch:
            return "adaptive_hard"
        return "adaptive_soft"

    def _bounded_tau(self, raw_tau: torch.Tensor) -> torch.Tensor:
        tau = self.tau_min + torch.sigmoid(raw_tau) * (self.tau_max - self.tau_min)
        if tau.shape[1] == 1 and self.num_ordinal_logits > 1:
            tau = tau.expand(-1, self.num_ordinal_logits, -1, -1)
        return tau

    def forward(
        self,
        raw_logits: torch.Tensor,
        reference_feature: torch.Tensor,
        epoch: int = 0,
    ) -> Dict[str, object]:
        phase = self._phase(epoch)
        if not self.enable_tau or self.tau_mode in {"disabled", "none"}:
            zero_tau = raw_logits.new_zeros(raw_logits.shape)
            return {
                "ordinal_logits": raw_logits,
                "tau_values": zero_tau,
                "adaptive_tau": zero_tau,
                "raw_tau": zero_tau,
                "fixed_tau": zero_tau,
                "tau_phase": phase,
                "corn_soft_enabled": False,
                "tau_stats": {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                },
            }

        feature = reference_feature.detach() if self.detach_features else reference_feature
        logits_for_tau = raw_logits.detach() if self.detach_logits else raw_logits
        feature = F.interpolate(feature, size=raw_logits.shape[-2:], mode="bilinear", align_corners=False)
        feature = self.feature_proj(feature)
        raw_tau = self.tau_head(torch.cat([feature, logits_for_tau], dim=1))
        adaptive_tau = self._bounded_tau(raw_tau)
        fixed_tau = self.fixed_tau_template.to(device=raw_logits.device, dtype=raw_logits.dtype).expand_as(adaptive_tau)

        tau_values = fixed_tau if phase == "fixed_warmup" else adaptive_tau
        ordinal_logits = raw_logits + tau_values
        tau_detached = tau_values.detach()
        tau_stats = {
            "mean": float(tau_detached.mean().item()),
            "std": float(tau_detached.std(unbiased=False).item()),
            "min": float(tau_detached.min().item()),
            "max": float(tau_detached.max().item()),
        }

        return {
            "ordinal_logits": ordinal_logits,
            "tau_values": tau_values,
            "adaptive_tau": adaptive_tau,
            "raw_tau": raw_tau,
            "fixed_tau": fixed_tau,
            "tau_phase": phase,
            "corn_soft_enabled": phase == "adaptive_soft",
            "tau_stats": tau_stats,
        }
