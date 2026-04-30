"""Explicit change feature builders."""

from __future__ import annotations

import torch
import torch.nn as nn


class ExplicitChangeBuilder(nn.Module):
    """Build explicit pre/post change representations.

    For feature maps: concat(pre, post, abs(post-pre), post-pre) -> 1x1 conv -> BN -> GELU
    For vectors: concat(pre, post, abs(post-pre), post-pre) -> linear -> LN -> GELU
    """

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        in_dim = feature_dim * 4
        self.map_proj = nn.Sequential(
            nn.Conv2d(in_dim, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),
        )
        self.vec_proj = nn.Sequential(
            nn.Linear(in_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

    def forward(self, pre_feat: torch.Tensor, post_feat: torch.Tensor) -> torch.Tensor:
        if pre_feat.shape != post_feat.shape:
            raise ValueError(
                f"Pre/Post feature shape mismatch: {tuple(pre_feat.shape)} vs {tuple(post_feat.shape)}"
            )
        delta = post_feat - pre_feat
        pieces = [pre_feat, post_feat, torch.abs(delta), delta]
        if pre_feat.ndim == 4:
            return self.map_proj(torch.cat(pieces, dim=1))
        if pre_feat.ndim == 2:
            return self.vec_proj(torch.cat(pieces, dim=1))
        raise ValueError(f"ExplicitChangeBuilder only supports 2D or 4D tensors, got ndim={pre_feat.ndim}.")
