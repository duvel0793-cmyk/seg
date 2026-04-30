"""Optional lightweight local-window attention for context maps."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalWindowAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, window_size: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = int(window_size)
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 4:
            return feature
        bsz, channels, height, width = feature.shape
        ws = self.window_size
        pad_h = (ws - height % ws) % ws
        pad_w = (ws - width % ws) % ws
        x = F.pad(feature, (0, pad_w, 0, pad_h))
        _, _, padded_h, padded_w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        windows = (
            x.view(bsz, padded_h // ws, ws, padded_w // ws, ws, channels)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, ws * ws, channels)
        )
        attn_input = self.norm(windows)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        windows = windows + self.proj(attn_output)
        x = (
            windows.view(bsz, padded_h // ws, padded_w // ws, ws, ws, channels)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(bsz, padded_h, padded_w, channels)
        )
        x = x[:, :height, :width, :].permute(0, 3, 1, 2).contiguous()
        return x
