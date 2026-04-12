"""Pixel-level ordinal CORN head."""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.ordinal_utils import corn_logits_to_label


class PixelCORNHead(nn.Module):
    """Predict K-1 ordinal logits for 4 xBD damage classes."""

    def __init__(self, feature_channels: Sequence[int], decoder_channels: int, num_damage_classes: int = 4) -> None:
        super().__init__()
        self.num_damage_classes = num_damage_classes
        self.num_ordinal_logits = num_damage_classes - 1
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
            nn.Conv2d(decoder_channels, self.num_ordinal_logits, kernel_size=1),
        )

    def forward(self, features: List[torch.Tensor], output_size) -> torch.Tensor:
        if not features:
            raise ValueError("PixelCORNHead expects at least one feature map.")
        target_size = features[0].shape[-2:]
        fused = 0
        for feat, proj in zip(features, self.proj):
            fused = fused + F.interpolate(proj(feat), size=target_size, mode="bilinear", align_corners=False)
        logits = self.refine(fused)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)

    @staticmethod
    def decode(logits: torch.Tensor) -> torch.Tensor:
        return corn_logits_to_label(logits)
