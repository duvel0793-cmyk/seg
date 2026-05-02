from __future__ import annotations

import math

import torch
import torch.nn as nn


def resolve_group_count(channels: int, max_groups: int = 32) -> int:
    channels = int(channels)
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(num_channels), eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"LayerNorm2d expects a 4D tensor, got shape={tuple(x.shape)}.")
        normalized = x.permute(0, 2, 3, 1)
        normalized = self.norm(normalized)
        return normalized.permute(0, 3, 1, 2).contiguous()


def build_channel_norm_2d(num_channels: int, *, kind: str = "group", max_groups: int = 32) -> nn.Module:
    kind = str(kind).lower()
    if kind == "layer":
        return LayerNorm2d(num_channels)
    if kind == "group":
        return nn.GroupNorm(resolve_group_count(num_channels, max_groups=max_groups), num_channels)
    raise ValueError(f"Unsupported 2D norm kind '{kind}'.")


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        init_value: float = 1.0e-3,
        max_scale: float | None = None,
        ndim: int = 3,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.ndim = int(ndim)
        self.max_scale = None if max_scale is None else float(max_scale)
        init_value = float(init_value)
        if self.max_scale is None:
            self.scale = nn.Parameter(torch.full((self.dim,), init_value, dtype=torch.float32))
            self.raw_scale = None
        else:
            ratio = min(max(init_value / max(self.max_scale, 1e-6), 1.0e-4), 1.0 - 1.0e-4)
            raw_value = math.log(ratio / (1.0 - ratio))
            self.raw_scale = nn.Parameter(torch.full((self.dim,), raw_value, dtype=torch.float32))
            self.register_parameter("scale", None)

    def current_scale(self) -> torch.Tensor:
        if self.max_scale is None:
            assert self.scale is not None
            return self.scale
        assert self.raw_scale is not None
        return self.max_scale * torch.sigmoid(self.raw_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.current_scale().to(device=x.device, dtype=x.dtype)
        if self.ndim == 3:
            scale = scale.view(1, 1, -1)
        elif self.ndim == 4:
            scale = scale.view(1, -1, 1, 1)
        else:
            raise ValueError(f"Unsupported LayerScale ndim={self.ndim}.")
        return x * scale
