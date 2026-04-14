from __future__ import annotations

import math

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, in_features: int, hidden_features: int = 512, num_classes: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class OrdinalCORNHead(nn.Module):
    def __init__(self, in_features: int, hidden_features: int = 512, num_classes: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("OrdinalCORNHead expects num_classes >= 2.")
        self.num_classes = int(num_classes)
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, self.num_classes - 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def _compute_tau_fraction(tau_value: float, tau_min: float, tau_max: float) -> float:
    tau_range = max(float(tau_max) - float(tau_min), 1e-6)
    fraction = (float(tau_value) - float(tau_min)) / tau_range
    return min(max(fraction, 1e-6), 1.0 - 1e-6)


class AdaptiveTauSafeHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = 256,
        dropout: float = 0.1,
        tau_min: float = 0.85,
        tau_max: float = 1.15,
        tau_init: float = 1.0,
        tau_logit_scale: float = 2.0,
    ) -> None:
        super().__init__()
        if tau_max <= tau_min:
            raise ValueError("tau_max must be greater than tau_min.")
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.tau_init = float(min(max(tau_init, tau_min), tau_max))
        self.tau_logit_scale = float(tau_logit_scale)
        self.raw_tau_center = self._inverse_sigmoid(_compute_tau_fraction(1.0, self.tau_min, self.tau_max))
        self.norm = nn.LayerNorm(in_features)
        self.proj = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_features, 1)
        self.reset_parameters()

    @staticmethod
    def _inverse_sigmoid(value: float) -> float:
        value = min(max(float(value), 1e-6), 1.0 - 1e-6)
        return math.log(value / (1.0 - value))

    @staticmethod
    def _inverse_tanh(value: float) -> float:
        value = min(max(float(value), -0.999999), 0.999999)
        return 0.5 * math.log((1.0 + value) / (1.0 - value))

    def _compute_initial_delta_bias(self, tau_value: float) -> float:
        target = self._inverse_sigmoid(_compute_tau_fraction(tau_value, self.tau_min, self.tau_max))
        normalized = (target - self.raw_tau_center) / max(self.tau_logit_scale, 1e-6)
        return self._inverse_tanh(normalized)

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="linear")
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.output.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.output.bias, self._compute_initial_delta_bias(self.tau_init))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.norm(x)
        hidden = self.proj(hidden)
        hidden = self.act(hidden)
        hidden = self.dropout(hidden)
        raw_delta_tau = self.output(hidden).squeeze(1)
        raw_tau = raw_delta_tau.new_tensor(self.raw_tau_center) + (
            raw_delta_tau.new_tensor(self.tau_logit_scale) * torch.tanh(raw_delta_tau)
        )
        tau = self.tau_min + ((self.tau_max - self.tau_min) * torch.sigmoid(raw_tau))
        return {
            "raw_tau": raw_tau,
            "raw_delta_tau": raw_delta_tau,
            "tau": tau.clamp(self.tau_min, self.tau_max),
        }

