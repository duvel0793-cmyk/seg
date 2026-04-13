from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttentionGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        gate = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * gate


class LightweightDirectionalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                groups=out_channels,
                bias=False,
            ),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _AlignProject(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BidirectionalFusionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.align_pre = _AlignProject(in_channels, out_channels)
        self.align_post = _AlignProject(in_channels, out_channels)
        self.directional = LightweightDirectionalBlock(out_channels * 2, out_channels)
        self.diff_proj = _AlignProject(out_channels, out_channels)
        self.mul_proj = _AlignProject(out_channels, out_channels)
        self.out_norm = nn.BatchNorm2d(out_channels)
        self.out_act = nn.GELU()

    def forward(self, feat_pre: torch.Tensor, feat_post: torch.Tensor) -> torch.Tensor:
        pre = self.align_pre(feat_pre)
        post = self.align_post(feat_post)
        ab = torch.cat([pre, post], dim=1)
        ba = torch.cat([post, pre], dim=1)
        abs_diff = (pre - post).abs()
        mul_corr = pre * post

        z = self.directional(ab) + self.directional(ba)
        z = z + self.diff_proj(abs_diff) + self.mul_proj(mul_corr)
        z = self.out_act(self.out_norm(z))
        return z
