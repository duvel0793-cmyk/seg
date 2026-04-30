from __future__ import annotations

import torch
import torch.nn as nn


def _resolve_groups(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class EvidenceHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int | None = None) -> None:
        super().__init__()
        hidden = int(hidden_channels or max(in_channels // 2, 64))
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_resolve_groups(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_resolve_groups(hidden), hidden),
            nn.GELU(),
        )
        self.evidence_out = nn.Conv2d(hidden, 4, kernel_size=1)
        self.severity_out = nn.Conv2d(hidden, 1, kernel_size=1)

    def forward(self, fused_feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.block(fused_feature)
        evidence_logits = self.evidence_out(hidden)
        severity_map = self.severity_out(hidden)
        return evidence_logits, severity_map

