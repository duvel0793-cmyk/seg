from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_last") -> None:
        super().__init__()
        if data_format not in {"channels_last", "channels_first"}:
            raise ValueError(f"Unsupported data_format='{data_format}'.")
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = (normalized_shape,)
        self.eps = float(eps)
        self.data_format = data_format

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, drop_path_rate: float = 0.0, layer_scale_init_value: float = 1e-6) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim)) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return residual + self.drop_path(x)


class ConvNeXt(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        depths: Sequence[int] = (3, 3, 9, 3),
        dims: Sequence[int] = (96, 192, 384, 768),
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()
        if len(depths) != 4 or len(dims) != 4:
            raise ValueError("ConvNeXt expects four stages.")

        self.depths = list(depths)
        self.dims = list(dims)

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            )
        )
        for idx in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm(dims[idx], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[idx], dims[idx + 1], kernel_size=2, stride=2),
                )
            )

        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cursor = 0
        self.stages = nn.ModuleList()
        for stage_idx in range(4):
            blocks = []
            for block_idx in range(depths[stage_idx]):
                blocks.append(
                    ConvNeXtBlock(
                        dim=dims[stage_idx],
                        drop_path_rate=dp_rates[cursor + block_idx],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                )
            self.stages.append(nn.Sequential(*blocks))
            cursor += depths[stage_idx]

        self.output_norm = LayerNorm(dims[-1], eps=1e-6, data_format="channels_first")
        self.feature_channels = OrderedDict(
            [
                ("c2", dims[0]),
                ("c3", dims[1]),
                ("c4", dims[2]),
                ("c5", dims[3]),
            ]
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

