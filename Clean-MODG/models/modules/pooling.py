"""Pooling modules for feature maps and instance masks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAvgPooling(nn.Module):
    def forward(self, feature: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if feature.ndim == 2:
            return feature
        if feature.ndim != 4:
            raise ValueError(f"GlobalAvgPooling expects 2D or 4D features, got {feature.ndim}D.")
        return feature.mean(dim=(2, 3))


class MaskGuidedPooling(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.global_pool = GlobalAvgPooling()

    def forward(self, feature: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if feature.ndim == 2:
            return feature
        if feature.ndim != 4:
            raise ValueError(f"MaskGuidedPooling expects 2D or 4D features, got {feature.ndim}D.")
        if mask is None:
            return self.global_pool(feature)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = mask.float()
        mask = F.interpolate(mask, size=feature.shape[-2:], mode="nearest")
        denom = mask.sum(dim=(2, 3), keepdim=True)
        pooled = (feature * mask).sum(dim=(2, 3), keepdim=True) / denom.clamp_min(self.eps)
        pooled = pooled.squeeze(-1).squeeze(-1)
        global_pooled = self.global_pool(feature)
        empty_mask = (denom.squeeze(-1).squeeze(-1).squeeze(-1) <= self.eps).view(-1, 1)
        pooled = torch.where(empty_mask, global_pooled, pooled)
        return pooled
