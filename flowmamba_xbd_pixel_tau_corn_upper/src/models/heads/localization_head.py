"""Localization head for building / background prediction."""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalizationHead(nn.Module):
    """Fuse multi-scale features and predict building localization logits."""

    def __init__(self, feature_channels: Sequence[int], decoder_channels: int, out_channels: int = 2) -> None:
        super().__init__()
        self.proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, decoder_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(decoder_channels),
                    nn.GELU(),
                )
                for in_channels in feature_channels
            ]
        )
        self.refine = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
            nn.Conv2d(decoder_channels, out_channels, kernel_size=1),
        )

    def forward(self, features: List[torch.Tensor], output_size) -> torch.Tensor:
        if not features:
            raise ValueError("LocalizationHead expects at least one feature map.")
        target_size = features[0].shape[-2:]
        fused = 0
        for feat, proj in zip(features, self.proj):
            fused = fused + F.interpolate(proj(feat), size=target_size, mode="bilinear", align_corners=False)
        logits = self.refine(fused)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
