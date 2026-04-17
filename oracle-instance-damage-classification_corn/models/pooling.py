from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMultiScalePooling(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def pool_single(self, feature: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scaled_mask = F.interpolate(mask, size=feature.shape[-2:], mode="nearest")
        scaled_mask = (scaled_mask > 0.5).float()
        gap = feature.mean(dim=(2, 3))
        gmp = feature.amax(dim=(2, 3))

        mask_area = scaled_mask.sum(dim=(2, 3))
        masked_avg = (feature * scaled_mask).sum(dim=(2, 3)) / mask_area.clamp_min(self.eps)

        mask_bool = scaled_mask.expand(-1, feature.shape[1], -1, -1) > 0.5
        masked_feature = feature.masked_fill(~mask_bool, torch.finfo(feature.dtype).min)
        masked_max = masked_feature.amax(dim=(2, 3))
        masked_max = torch.where(torch.isfinite(masked_max), masked_max, gmp)

        has_mask = (mask_area > 0).expand(-1, feature.shape[1])
        masked_avg = torch.where(has_mask, masked_avg, gap)
        masked_max = torch.where(has_mask, masked_max, gmp)
        return torch.cat([masked_avg, masked_max], dim=1)

    def forward(
        self,
        features: Iterable[torch.Tensor] | OrderedDict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(features, OrderedDict):
            features = list(features.values())
        pooled = [self.pool_single(feature, mask) for feature in features]
        return torch.cat(pooled, dim=1)
