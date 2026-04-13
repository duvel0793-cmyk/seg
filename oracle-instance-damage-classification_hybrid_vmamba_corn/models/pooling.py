from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedAttentionPool2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 4, 16)
        self.score = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
        )

    def forward(self, feature: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        scaled_mask = F.interpolate(mask, size=feature.shape[-2:], mode="nearest")
        mask_bool = scaled_mask > 0.5
        score = self.score(feature)
        flat_score = score.flatten(2)
        flat_mask = mask_bool.flatten(2)
        global_weights = torch.softmax(flat_score, dim=-1)
        masked_score = flat_score.masked_fill(~flat_mask, torch.finfo(score.dtype).min)
        masked_weights = torch.softmax(masked_score, dim=-1)
        valid_mask = flat_mask.any(dim=-1, keepdim=True)
        weights = torch.where(valid_mask, masked_weights, global_weights).view_as(score)
        pooled = (feature * weights).sum(dim=(2, 3))

        mask_area = scaled_mask.sum(dim=(2, 3))
        global_avg = feature.mean(dim=(2, 3))
        has_valid_area = (mask_area > eps).expand(-1, feature.shape[1])
        return torch.where(has_valid_area, pooled, global_avg)


class MaskedMultiScalePooling(nn.Module):
    def __init__(
        self,
        feature_channels: Mapping[str, int] | None = None,
        pool_modes: Iterable[str] = ("avg", "max"),
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.pool_modes = tuple(str(mode) for mode in pool_modes)
        unsupported = sorted(set(self.pool_modes) - {"avg", "max", "attention"})
        if unsupported:
            raise ValueError(f"Unsupported pool_modes={unsupported}.")

        self.feature_names = list(feature_channels.keys()) if feature_channels is not None else None
        self.attention_pools = nn.ModuleDict()
        if "attention" in self.pool_modes:
            if feature_channels is None:
                raise ValueError("feature_channels is required when attention pooling is enabled.")
            self.attention_pools = nn.ModuleDict(
                {
                    name: MaskedAttentionPool2d(channels)
                    for name, channels in feature_channels.items()
                }
            )

    @property
    def output_multiplier(self) -> int:
        return len(self.pool_modes)

    @staticmethod
    def compute_output_dim(
        feature_channels: Mapping[str, int],
        pool_modes: Iterable[str] = ("avg", "max"),
    ) -> int:
        multiplier = len(tuple(pool_modes))
        return sum(int(channels) * multiplier for channels in feature_channels.values())

    def _pool_single(
        self,
        name: str,
        feature: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        scaled_mask = F.interpolate(mask, size=feature.shape[-2:], mode="nearest")
        scaled_mask = (scaled_mask > 0.5).float()
        mask_area = scaled_mask.sum(dim=(2, 3))
        global_avg = feature.mean(dim=(2, 3))
        global_max = feature.amax(dim=(2, 3))
        has_mask = (mask_area > 0).expand(-1, feature.shape[1])

        pooled_parts: list[torch.Tensor] = []
        if "avg" in self.pool_modes:
            masked_avg = (feature * scaled_mask).sum(dim=(2, 3)) / mask_area.clamp_min(self.eps)
            masked_avg = torch.where(has_mask, masked_avg, global_avg)
            pooled_parts.append(masked_avg)

        if "max" in self.pool_modes:
            mask_bool = scaled_mask.expand(-1, feature.shape[1], -1, -1) > 0.5
            masked_feature = feature.masked_fill(~mask_bool, torch.finfo(feature.dtype).min)
            masked_max = masked_feature.amax(dim=(2, 3))
            masked_max = torch.where(torch.isfinite(masked_max), masked_max, global_max)
            masked_max = torch.where(has_mask, masked_max, global_max)
            pooled_parts.append(masked_max)

        if "attention" in self.pool_modes:
            attention_pool = self.attention_pools[name]
            pooled_parts.append(attention_pool(feature, mask, eps=self.eps))

        return torch.cat(pooled_parts, dim=1)

    def forward(
        self,
        features: Iterable[torch.Tensor] | Mapping[str, torch.Tensor] | OrderedDict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(features, Mapping):
            items = list(features.items())
        else:
            feature_list = list(features)
            if self.feature_names is not None and len(self.feature_names) == len(feature_list):
                items = list(zip(self.feature_names, feature_list))
            else:
                items = [(f"scale_{index}", feature) for index, feature in enumerate(feature_list)]

        pooled = [self._pool_single(name, feature, mask) for name, feature in items]
        return torch.cat(pooled, dim=1)
