from __future__ import annotations

import torch
import torch.nn as nn


class ResidualGateAlignment(nn.Module):
    """
    Lightweight pre/post alignment block.

    It predicts a residual correction for the pre-disaster feature from
    concatenated pre/post evidence, reducing pseudo-change caused by mild
    misregistration, viewpoint shift, and illumination drift.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 2, 64)
        self.context = nn.Sequential(
            nn.Conv2d(channels * 3, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=max(hidden // 8, 1), bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )
        self.correction = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.gate = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, pre_feature: torch.Tensor, post_feature: torch.Tensor) -> torch.Tensor:
        diff = post_feature - pre_feature
        context = self.context(torch.cat([pre_feature, post_feature, diff], dim=1))
        correction = self.correction(context)
        gate = torch.sigmoid(self.gate(context))
        return pre_feature + (gate * correction)
