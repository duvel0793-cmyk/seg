from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn_groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(_gn_groups(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FPN(nn.Module):
    def __init__(self, in_channels: dict[str, int], out_channels: int) -> None:
        super().__init__()
        self.lateral_c2 = nn.Conv2d(in_channels["c2"], out_channels, kernel_size=1, bias=False)
        self.lateral_c3 = nn.Conv2d(in_channels["c3"], out_channels, kernel_size=1, bias=False)
        self.lateral_c4 = nn.Conv2d(in_channels["c4"], out_channels, kernel_size=1, bias=False)
        self.lateral_c5 = nn.Conv2d(in_channels["c5"], out_channels, kernel_size=1, bias=False)
        self.out_p2 = ConvNormAct(out_channels, out_channels, kernel_size=3, padding=1)
        self.out_p3 = ConvNormAct(out_channels, out_channels, kernel_size=3, padding=1)
        self.out_p4 = ConvNormAct(out_channels, out_channels, kernel_size=3, padding=1)
        self.out_p5 = ConvNormAct(out_channels, out_channels, kernel_size=3, padding=1)

    @staticmethod
    def _upsample_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=y.shape[-2:], mode="nearest") + y

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        c2 = self.lateral_c2(features["c2"])
        c3 = self.lateral_c3(features["c3"])
        c4 = self.lateral_c4(features["c4"])
        c5 = self.lateral_c5(features["c5"])

        p5 = c5
        p4 = self._upsample_add(p5, c4)
        p3 = self._upsample_add(p4, c3)
        p2 = self._upsample_add(p3, c2)
        return {
            "p2": self.out_p2(p2),
            "p3": self.out_p3(p3),
            "p4": self.out_p4(p4),
            "p5": self.out_p5(p5),
        }
