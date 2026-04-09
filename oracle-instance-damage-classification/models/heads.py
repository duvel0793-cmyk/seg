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


def compute_tau_fraction(
    tau_value: float,
    tau_min: float,
    tau_max: float,
) -> float:
    tau_range = max(float(tau_max) - float(tau_min), 1e-6)
    fraction = (float(tau_value) - float(tau_min)) / tau_range
    return min(max(fraction, 1e-6), 1.0 - 1e-6)


def compute_raw_tau_center(
    tau_target: float,
    tau_min: float,
    tau_max: float,
) -> float:
    return OrdinalAmbiguityHead._inverse_sigmoid(compute_tau_fraction(tau_target, tau_min, tau_max))


def compute_sample_tau_from_delta(raw_delta_tau: torch.Tensor, cfg: Mapping[str, Any] | object) -> torch.Tensor:
    tau_min = _resolve_tau_cfg_value(cfg, "tau_min")
    tau_max = _resolve_tau_cfg_value(cfg, "tau_max")
    tau_base = _resolve_tau_cfg_value(cfg, "tau_base")
    delta_scale = _resolve_tau_cfg_value(cfg, "delta_scale")

    delta_tau = raw_delta_tau.new_tensor(delta_scale) * torch.tanh(raw_delta_tau)
    tau = raw_delta_tau.new_tensor(tau_base) + delta_tau
    return tau.clamp(min=tau_min, max=tau_max)


def validate_residual_tau_geometry(
    tau_min: float,
    tau_max: float,
    tau_base: float,
    delta_scale: float,
) -> None:
    safe_delta_scale = min(float(tau_base) - float(tau_min), float(tau_max) - float(tau_base))
    if float(delta_scale) > safe_delta_scale + 1e-12:
        raise ValueError(
            "当前 residual 几何范围会触碰边界，容易造成 tau clamp 塌缩，请减小 delta_scale 或改用 sigmoid "
            f"(delta_scale={float(delta_scale):.4f}, safe_max={max(float(safe_delta_scale), 0.0):.4f}, "
            f"tau_base={float(tau_base):.4f}, tau_min={float(tau_min):.4f}, tau_max={float(tau_max):.4f})."
        )


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
        tau_target: float = 0.22,
        tau_logit_scale: float = 2.0,
        tau_parameterization: str = "sigmoid",
    ) -> None:
        super().__init__()
        if tau_min <= 0.0:
            raise ValueError("tau_min must be > 0.")
        if tau_max <= tau_min:
            raise ValueError("tau_max must be larger than tau_min.")
        if delta_scale <= 0.0:
            raise ValueError("delta_scale must be > 0.")
        if tau_logit_scale <= 0.0:
            raise ValueError("tau_logit_scale must be > 0.")
        if tau_parameterization not in {"sigmoid", "residual", "bounded_sigmoid"}:
            raise ValueError(f"Unsupported tau_parameterization='{tau_parameterization}'.")

        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.tau_base = float(min(max(tau_base, tau_min), tau_max))
        self.delta_scale = float(delta_scale)
        self.tau_init = float(min(max(tau_init, tau_min), tau_max))
        self.tau_target = float(min(max(tau_target, tau_min), tau_max))
        self.tau_logit_scale = float(tau_logit_scale)
        self.raw_tau_center = compute_raw_tau_center(self.tau_target, self.tau_min, self.tau_max)
        self.tau_parameterization = str(tau_parameterization)
        if self.tau_parameterization == "residual":
            validate_residual_tau_geometry(
                tau_min=self.tau_min,
                tau_max=self.tau_max,
                tau_base=self.tau_base,
                delta_scale=self.delta_scale,
            )
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
        fraction = compute_tau_fraction(tau_value, self.tau_min, self.tau_max)
        return self._inverse_sigmoid(fraction)

    @staticmethod
    def _inverse_tanh(value: float) -> float:
        value = min(max(float(value), -0.999999), 0.999999)
        return 0.5 * math.log((1.0 + value) / (1.0 - value))

    def compute_initial_delta_bias(self, tau_value: float) -> float:
        target_raw_tau = self.compute_initial_bias(tau_value)
        normalized_offset = (target_raw_tau - self.raw_tau_center) / max(self.tau_logit_scale, 1e-6)
        return self._inverse_tanh(normalized_offset)

    def reset_tau_parameters(self, tau_value: float | None = None) -> None:
        target_tau = self.tau_init if tau_value is None else float(tau_value)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="linear")
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.output.weight, mean=0.0, std=1e-3)
        if self.tau_parameterization == "residual":
            nn.init.zeros_(self.output.bias)
        elif self.tau_parameterization == "bounded_sigmoid":
            nn.init.constant_(self.output.bias, self.compute_initial_delta_bias(target_tau))
        else:
            nn.init.constant_(self.output.bias, self.compute_initial_bias(target_tau))

    def map_raw_tau(self, raw_tau: torch.Tensor) -> torch.Tensor:
        if self.tau_parameterization == "residual":
            return compute_sample_tau_from_delta(raw_tau, self)
        tau_range = self.tau_max - self.tau_min
        return self.tau_min + tau_range * torch.sigmoid(raw_tau)

    def resolve_raw_tau_outputs(self, raw_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.tau_parameterization == "bounded_sigmoid":
            raw_delta_tau = raw_value
            raw_tau = raw_value.new_tensor(self.raw_tau_center) + (
                raw_value.new_tensor(self.tau_logit_scale) * torch.tanh(raw_delta_tau)
            )
            return raw_tau, raw_delta_tau
        if self.tau_parameterization == "residual":
            return raw_value, raw_value
        return raw_value, None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        hidden = self.norm(x)
        hidden = self.proj(hidden)
        hidden = self.act(hidden)
        hidden = self.dropout(hidden)
        raw_value = self.output(hidden).squeeze(1)
        raw_tau, raw_delta_tau = self.resolve_raw_tau_outputs(raw_value)
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
