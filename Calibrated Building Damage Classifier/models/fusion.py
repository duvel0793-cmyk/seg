from __future__ import annotations

import torch
import torch.nn as nn


class PrePostFusion(nn.Module):
    """Single-path fusion over aligned pre, post, diff, and product cues."""

    def __init__(self, channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=max(out_channels // 8, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, aligned_pre: torch.Tensor, post_feature: torch.Tensor) -> torch.Tensor:
        diff = post_feature - aligned_pre
        prod = post_feature * aligned_pre
        fused = torch.cat([aligned_pre, post_feature, diff, prod], dim=1)
        return self.block(fused)
