from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMultiScalePooling(nn.Module):
    def __init__(self, eps: float = 1e-6, use_masked_max: bool = True) -> None:
        super().__init__()
        self.eps = float(eps)
        self.use_masked_max = bool(use_masked_max)

    def pool_single(self, feature: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        resized_mask = F.interpolate(mask.float(), size=feature.shape[-2:], mode="nearest")
        binary_mask = (resized_mask > 0.5).float()
        gap = feature.mean(dim=(2, 3))
        mask_area = binary_mask.sum(dim=(2, 3))
        masked_avg = (feature * binary_mask).sum(dim=(2, 3)) / mask_area.clamp_min(self.eps)
        has_mask = (mask_area > 0).expand(-1, feature.shape[1])
        masked_avg = torch.where(has_mask, masked_avg, gap)
        pooled = [masked_avg]

        if self.use_masked_max:
            gmp = feature.amax(dim=(2, 3))
            mask_bool = binary_mask.expand(-1, feature.shape[1], -1, -1) > 0.5
            masked_feature = feature.masked_fill(~mask_bool, torch.finfo(feature.dtype).min)
            masked_max = masked_feature.amax(dim=(2, 3))
            masked_max = torch.where(torch.isfinite(masked_max), masked_max, gmp)
            masked_max = torch.where(has_mask, masked_max, gmp)
            pooled.append(masked_max)
        return torch.cat(pooled, dim=1)

    def forward(self, features: Iterable[torch.Tensor] | OrderedDict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        feature_list = list(features.values()) if isinstance(features, OrderedDict) else list(features)
        pooled = [self.pool_single(feature, mask) for feature in feature_list]
        return torch.cat(pooled, dim=1)

