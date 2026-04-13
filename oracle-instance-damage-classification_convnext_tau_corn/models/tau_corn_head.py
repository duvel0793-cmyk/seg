"""Stable learnable tau + CORN classification head."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics.corn_decode import VALID_DECODE_MODES, corn_class_probabilities, decode_corn_logits


def inverse_softplus(value: float) -> float:
    """Inverse of softplus for positive initialization values."""
    if value <= 0:
        raise ValueError("Softplus inverse expects a positive value.")
    return math.log(math.exp(value) - 1.0)


class TauCORNHead(nn.Module):
    """Ordinal head that applies a learnable positive tau to CORN logits."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 4,
        hidden_dim: int = 256,
        tau_mode: str = "per_threshold",
        tau_init: float = 1.0,
        tau_min: float = 0.5,
        tau_max: float = 2.0,
        dropout: float = 0.1,
        decode_mode: str = "threshold_count",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("CORN requires at least two classes.")
        if tau_mode not in {"shared", "per_threshold"}:
            raise ValueError(f"Unsupported tau_mode: {tau_mode}")
        if decode_mode not in VALID_DECODE_MODES:
            raise ValueError(f"Unsupported decode_mode: {decode_mode}")

        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        self.tau_mode = tau_mode
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.decode_mode = decode_mode
        self.eps = eps

        self.base_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_thresholds),
        )

        tau_shape = (1,) if tau_mode == "shared" else (self.num_thresholds,)
        tau_init_raw = inverse_softplus(max(tau_init - eps, eps))
        self.tau_raw = nn.Parameter(torch.full(tau_shape, tau_init_raw))

    def get_tau(self) -> torch.Tensor:
        """Return clamped positive tau values."""
        return self.get_tau_info()["tau_clamped"]

    def get_tau_info(self) -> Dict[str, torch.Tensor | float]:
        """Return raw, positive, and clamped tau values plus clamp bounds."""
        tau_positive = F.softplus(self.tau_raw) + self.eps
        tau_clamped = tau_positive.clamp(min=self.tau_min, max=self.tau_max)
        return {
            "tau_raw": self.tau_raw,
            "tau_positive_before_clamp": tau_positive,
            "tau_clamped": tau_clamped,
            "tau_min": float(self.tau_min),
            "tau_max": float(self.tau_max),
            "tau_mode": self.tau_mode,
        }

    def regularization_terms(self) -> Dict[str, torch.Tensor]:
        """Regularization terms that keep tau close to 1 and smooth across thresholds."""
        tau = self.get_tau()
        tau_center_reg = ((tau - 1.0) ** 2).mean()
        if tau.numel() > 1:
            tau_diff_reg = ((tau[1:] - tau[:-1]) ** 2).mean()
        else:
            tau_diff_reg = tau.new_zeros(())
        return {
            "tau_center_reg": tau_center_reg,
            "tau_diff_reg": tau_diff_reg,
        }

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """Decode ordinal logits into class labels."""
        return decode_corn_logits(logits, mode=self.decode_mode)

    def forward(self, fused_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate base logits, tau-scaled logits, probabilities, and predictions."""
        base_logits = self.base_head(fused_features)
        tau_info = self.get_tau_info()
        tau = tau_info["tau_clamped"]
        if tau.numel() == 1:
            scaled_logits = base_logits / tau.view(1, 1)
        else:
            scaled_logits = base_logits / tau.view(1, -1)

        pred_labels = self.decode(scaled_logits)
        class_probs = corn_class_probabilities(scaled_logits)
        regularizers = self.regularization_terms()
        return {
            "base_logits": base_logits,
            "tau": tau,
            "tau_raw": tau_info["tau_raw"],
            "tau_positive": tau_info["tau_positive_before_clamp"],
            "tau_info": tau_info,
            "logits": scaled_logits,
            "pred_labels": pred_labels,
            "class_probs": class_probs,
            **regularizers,
        }
