from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttentionGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
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


class SimpleBidirectionalFusionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_attention: bool = True) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()
        self.attention = ChannelAttentionGate(out_channels) if use_attention else nn.Identity()

    def forward(self, feat_pre: torch.Tensor, feat_post: torch.Tensor) -> torch.Tensor:
        fusion = torch.cat([feat_pre, feat_post, feat_post - feat_pre, (feat_post - feat_pre).abs()], dim=1)
        fused = self.proj(fusion)
        fused = fused + self.skip(0.5 * (feat_pre + feat_post))
        return self.attention(fused)

