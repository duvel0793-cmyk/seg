from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn


class InstanceClassifierHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = 512,
        num_classes: int = 4,
        dropout: float = 0.2,
    ) -> None:
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


def _resolve_tau_cfg_value(cfg: Mapping[str, Any] | object, key: str) -> float:
    if isinstance(cfg, Mapping):
        return float(cfg[key])
    return float(getattr(cfg, key))


def compute_sample_tau_from_delta(raw_delta_tau: torch.Tensor, cfg: Mapping[str, Any] | object) -> torch.Tensor:
    tau_min = _resolve_tau_cfg_value(cfg, "tau_min")
    tau_max = _resolve_tau_cfg_value(cfg, "tau_max")
    tau_base = _resolve_tau_cfg_value(cfg, "tau_base")
    delta_scale = _resolve_tau_cfg_value(cfg, "delta_scale")

    delta_tau = raw_delta_tau.new_tensor(delta_scale) * torch.tanh(raw_delta_tau)
    tau = raw_delta_tau.new_tensor(tau_base) + delta_tau
    return tau.clamp(min=tau_min, max=tau_max)


class OrdinalAmbiguityHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = 256,
        dropout: float = 0.1,
        tau_min: float = 0.10,
        tau_max: float = 0.60,
        tau_base: float = 0.22,
        delta_scale: float = 0.12,
        tau_init: float = 0.27,
        tau_parameterization: str = "sigmoid",
    ) -> None:
        super().__init__()
        if tau_min <= 0.0:
            raise ValueError("tau_min must be > 0.")
        if tau_max <= tau_min:
            raise ValueError("tau_max must be larger than tau_min.")
        if delta_scale <= 0.0:
            raise ValueError("delta_scale must be > 0.")
        if tau_parameterization not in {"sigmoid", "residual"}:
            raise ValueError(f"Unsupported tau_parameterization='{tau_parameterization}'.")

        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.tau_base = float(min(max(tau_base, tau_min), tau_max))
        self.delta_scale = float(delta_scale)
        self.tau_init = float(min(max(tau_init, tau_min), tau_max))
        self.tau_parameterization = str(tau_parameterization)
        self.norm = nn.LayerNorm(in_features)
        self.proj = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_features, 1)
        self.reset_tau_parameters(self.tau_base if self.tau_parameterization == "residual" else self.tau_init)

    @staticmethod
    def _inverse_sigmoid(value: float) -> float:
        value = min(max(float(value), 1e-6), 1.0 - 1e-6)
        return math.log(value / (1.0 - value))

    def compute_initial_bias(self, tau_value: float) -> float:
        tau_value = min(max(float(tau_value), self.tau_min), self.tau_max)
        fraction = (tau_value - self.tau_min) / max(self.tau_max - self.tau_min, 1e-6)
        return self._inverse_sigmoid(fraction)

    def reset_tau_parameters(self, tau_value: float | None = None) -> None:
        target_tau = self.tau_init if tau_value is None else float(tau_value)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="linear")
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.output.weight, mean=0.0, std=1e-3)
        if self.tau_parameterization == "residual":
            nn.init.zeros_(self.output.bias)
        else:
            nn.init.constant_(self.output.bias, self.compute_initial_bias(target_tau))

    def map_raw_tau(self, raw_tau: torch.Tensor) -> torch.Tensor:
        if self.tau_parameterization == "residual":
            return compute_sample_tau_from_delta(raw_tau, self)
        tau_range = self.tau_max - self.tau_min
        return self.tau_min + tau_range * torch.sigmoid(raw_tau)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        hidden = self.norm(x)
        hidden = self.proj(hidden)
        hidden = self.act(hidden)
        hidden = self.dropout(hidden)
        raw_value = self.output(hidden).squeeze(1)
        raw_delta_tau = raw_value if self.tau_parameterization == "residual" else None
        raw_tau = raw_value
        return {
            "raw_tau": raw_tau,
            "raw_delta_tau": raw_delta_tau,
            "tau": self.map_raw_tau(raw_tau),
        }


class OrdinalCORNHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = 512,
        num_classes: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if int(num_classes) < 2:
            raise ValueError("OrdinalCORNHead expects at least 2 ordered classes.")
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
