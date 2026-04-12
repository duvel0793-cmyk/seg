from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn


class WeightedFocalCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        class_weights: Optional[Sequence[float]] = None,
        gamma: float = 2.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.class_weights = None
        self.gamma = gamma
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W] logits, got shape {tuple(logits.shape)}")
        if target.ndim != 3:
            raise ValueError(f"Expected [B, H, W] target, got shape {tuple(target.shape)}")
        if logits.shape[-2:] != target.shape[-2:]:
            raise ValueError(
                f"Logits/target spatial mismatch: {tuple(logits.shape[-2:])} vs {tuple(target.shape[-2:])}"
            )

        log_probabilities = F.log_softmax(logits, dim=1)
        target = target.long()
        target_log_probabilities = log_probabilities.gather(1, target.unsqueeze(1)).squeeze(1)
        probabilities = target_log_probabilities.exp().clamp(min=self.eps, max=1.0)

        loss = -(1.0 - probabilities).pow(self.gamma) * target_log_probabilities

        if self.class_weights is not None:
            pixel_class_weights = self.class_weights.to(logits.device)[target]
            loss = loss * pixel_class_weights

        if weight_map is not None:
            if weight_map.ndim == 4:
                weight_map = weight_map.squeeze(1)
            if weight_map.shape != target.shape:
                raise ValueError(
                    f"Weight map shape mismatch: {tuple(weight_map.shape)} vs {tuple(target.shape)}"
                )
            loss = loss * weight_map.to(device=logits.device, dtype=loss.dtype)

        return loss.mean()


def build_prior_weight_map(prior_mask: torch.Tensor, alpha: float) -> torch.Tensor:
    if prior_mask.ndim == 3:
        prior_mask = prior_mask.unsqueeze(1)
    if prior_mask.ndim != 4:
        raise ValueError(f"Prior mask must be [B, 1, H, W] or [B, H, W], got {tuple(prior_mask.shape)}")
    return 1.0 + (alpha * prior_mask.float())
