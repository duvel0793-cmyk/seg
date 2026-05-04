from __future__ import annotations

import torch
import torch.nn as nn


class PixelDecoder(nn.Module):
    def __init__(self, in_channels: int, pixel_channels: int) -> None:
        super().__init__()
        self.pixel_proj = nn.Sequential(
            nn.Conv2d(in_channels, pixel_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8 if pixel_channels % 8 == 0 else 1, pixel_channels),
            nn.GELU(),
            nn.Conv2d(pixel_channels, pixel_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8 if pixel_channels % 8 == 0 else 1, pixel_channels),
            nn.GELU(),
        )

    def forward(self, pyramid: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        pixel_feature = self.pixel_proj(pyramid["p2"])
        memory_levels = [pyramid["p3"], pyramid["p4"], pyramid["p5"]]
        return {"pixel_feature": pixel_feature, "memory_levels": memory_levels}
