from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalChannelGate(nn.Module):
    def __init__(self, channels: int, num_streams: int, reduction_ratio: int = 4) -> None:
        super().__init__()
        hidden = max(channels // max(int(reduction_ratio), 1), 16)
        self.num_streams = int(num_streams)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels * 4, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels * self.num_streams, kernel_size=1, bias=True),
        )

    def forward(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        abs_diff: torch.Tensor,
        sum_feat: torch.Tensor,
    ) -> list[torch.Tensor]:
        pooled = torch.cat(
            [
                F.adaptive_avg_pool2d(pre, 1),
                F.adaptive_avg_pool2d(post, 1),
                F.adaptive_avg_pool2d(abs_diff, 1),
                F.adaptive_avg_pool2d(sum_feat, 1),
            ],
            dim=1,
        )
        gates = torch.sigmoid(self.mlp(pooled))
        return list(gates.chunk(self.num_streams, dim=1))


class TemporalDifferenceGate(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
        )

    def forward(self, abs_diff: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate(abs_diff))


class _ResidualRefinement(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class BDATemporalFusionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 4,
        use_abs_diff: bool = True,
        use_sum_feat: bool = True,
        use_prod_feat: bool = False,
        temporal_gate: bool = True,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.use_abs_diff = bool(use_abs_diff)
        self.use_sum_feat = bool(use_sum_feat)
        self.use_prod_feat = bool(use_prod_feat)
        self.temporal_gate_enabled = bool(temporal_gate)

        num_streams = 2 + int(self.use_abs_diff) + int(self.use_sum_feat) + int(self.use_prod_feat)
        self.stream_names = ["pre", "post"]
        if self.use_abs_diff:
            self.stream_names.append("abs_diff")
        if self.use_sum_feat:
            self.stream_names.append("sum")
        if self.use_prod_feat:
            self.stream_names.append("prod")

        self.channel_gate = (
            TemporalChannelGate(channels=channels, num_streams=num_streams, reduction_ratio=reduction_ratio)
            if self.temporal_gate_enabled
            else None
        )
        self.difference_gate = TemporalDifferenceGate(channels=channels)
        self.reduce = nn.Sequential(
            nn.Conv2d(channels * num_streams, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.spatial_mixer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.change_hint_proj = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        self.refine = _ResidualRefinement(channels=channels)

    def forward(self, f_pre: torch.Tensor, f_post: torch.Tensor) -> dict[str, Any]:
        abs_diff = (f_post - f_pre).abs()
        sum_feat = f_post + f_pre
        prod_feat = f_post * f_pre

        streams = [f_pre, f_post]
        if self.use_abs_diff:
            streams.append(abs_diff)
        if self.use_sum_feat:
            streams.append(sum_feat)
        if self.use_prod_feat:
            streams.append(prod_feat)

        spatial_change_gate = self.difference_gate(abs_diff)
        if self.channel_gate is not None:
            channel_gates = self.channel_gate(f_pre, f_post, abs_diff, sum_feat)
            gated_streams = []
            for name, stream, gate in zip(self.stream_names, streams, channel_gates):
                current = stream * gate
                if name == "abs_diff":
                    current = current * spatial_change_gate
                gated_streams.append(current)
            streams = gated_streams
        else:
            streams = [
                stream * spatial_change_gate if name == "abs_diff" else stream
                for name, stream in zip(self.stream_names, streams)
            ]

        fused = self.reduce(torch.cat(streams, dim=1))
        fused = self.spatial_mixer(fused)
        residual = 0.5 * (f_pre + f_post)
        fused = self.refine(fused + residual)
        change_hint = torch.sigmoid(self.change_hint_proj(abs_diff)) * spatial_change_gate
        return {
            "fused": fused,
            "change_hint": change_hint,
            "abs_diff": abs_diff,
            "sum_feat": sum_feat,
        }
