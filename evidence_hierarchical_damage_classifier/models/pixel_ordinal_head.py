from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hierarchical_head import corn_logits_to_threshold_probabilities, decode_corn_probabilities


def _resolve_groups(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def resize_mask_to_feature_map(mask: torch.Tensor | None, spatial_size: tuple[int, int]) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    return F.interpolate(mask.float(), size=spatial_size, mode="nearest")


def _normalize_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    return probabilities.clamp_min(1.0e-8) / probabilities.clamp_min(1.0e-8).sum(dim=1, keepdim=True).clamp_min(1.0e-8)


def _flatten_masked_probabilities(pixel_probs: torch.Tensor, mask: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    resized_mask = resize_mask_to_feature_map(mask, pixel_probs.shape[-2:])
    if resized_mask is None:
        raise ValueError("mask is required for masked pixel probability aggregation.")
    valid_mask = resized_mask > 0.5
    flat_probabilities: list[torch.Tensor] = []
    flat_masks: list[torch.Tensor] = []
    for batch_index in range(pixel_probs.size(0)):
        sample_mask = valid_mask[batch_index, 0]
        flat_probabilities.append(pixel_probs[batch_index].permute(1, 2, 0).reshape(-1, pixel_probs.size(1)))
        flat_masks.append(sample_mask.reshape(-1))
    return flat_probabilities, flat_masks


def masked_mean_probabilities(pixel_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    flat_probabilities, flat_masks = _flatten_masked_probabilities(pixel_probs, mask)
    outputs = []
    for sample_probabilities, sample_mask in zip(flat_probabilities, flat_masks, strict=False):
        valid_probabilities = sample_probabilities[sample_mask]
        if valid_probabilities.numel() == 0:
            outputs.append(sample_probabilities.new_full((sample_probabilities.size(1),), 1.0 / sample_probabilities.size(1)))
            continue
        outputs.append(valid_probabilities.mean(dim=0))
    return _normalize_probabilities(torch.stack(outputs, dim=0))


def masked_topk_probabilities(pixel_probs: torch.Tensor, mask: torch.Tensor, topk_ratio: float = 0.2) -> torch.Tensor:
    flat_probabilities, flat_masks = _flatten_masked_probabilities(pixel_probs, mask)
    outputs = []
    for sample_probabilities, sample_mask in zip(flat_probabilities, flat_masks, strict=False):
        valid_probabilities = sample_probabilities[sample_mask]
        if valid_probabilities.numel() == 0:
            outputs.append(sample_probabilities.new_full((sample_probabilities.size(1),), 1.0 / sample_probabilities.size(1)))
            continue
        k = max(1, min(valid_probabilities.size(0), int(round(valid_probabilities.size(0) * float(topk_ratio)))))
        outputs.append(torch.topk(valid_probabilities, k=k, dim=0).values.mean(dim=0))
    return _normalize_probabilities(torch.stack(outputs, dim=0))


def aggregate_pixel_probabilities(
    pixel_probs: torch.Tensor,
    mask: torch.Tensor,
    mode: str,
    *,
    topk_ratio: float = 0.2,
    mean_weight: float = 0.7,
    topk_weight: float = 0.3,
) -> dict[str, torch.Tensor]:
    mode = str(mode).lower()
    mean_probabilities = masked_mean_probabilities(pixel_probs, mask)
    topk_probabilities = masked_topk_probabilities(pixel_probs, mask, topk_ratio=topk_ratio)
    if mode == "mean":
        aggregated = mean_probabilities
    elif mode == "topk":
        aggregated = topk_probabilities
    elif mode == "mean_topk_mix":
        aggregated = (float(mean_weight) * mean_probabilities) + (float(topk_weight) * topk_probabilities)
    else:
        raise ValueError(f"Unsupported pixel aggregation mode '{mode}'.")
    aggregated = _normalize_probabilities(aggregated)
    return {
        "pixel_instance_probabilities_mean": mean_probabilities,
        "pixel_instance_probabilities_topk": topk_probabilities,
        "pixel_instance_probabilities": aggregated,
        "pixel_instance_pred_labels": aggregated.argmax(dim=1),
    }


def _dense_decode_corn(corn_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_thresholds, height, width = corn_logits.shape
    threshold_probabilities = corn_logits_to_threshold_probabilities(corn_logits.permute(0, 2, 3, 1).reshape(-1, num_thresholds))
    class_probabilities = decode_corn_probabilities(threshold_probabilities).reshape(batch_size, height, width, num_thresholds + 1)
    class_probabilities = class_probabilities.permute(0, 3, 1, 2).contiguous()
    class_probabilities = _normalize_probabilities(class_probabilities)
    pred_labels = class_probabilities.argmax(dim=1)
    return class_probabilities, pred_labels


class _PixelConvHead(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_resolve_groups(hidden_dim), hidden_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PixelTwoStageOrdinalHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        topk_ratio: float = 0.2,
        aggregation_mode: str = "mean_topk_mix",
        mean_weight: float = 0.7,
        topk_weight: float = 0.3,
        damage_decision_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.damage_head = _PixelConvHead(in_channels, hidden_dim, 1, dropout)
        self.severity_head = _PixelConvHead(in_channels, hidden_dim, 2, dropout)
        self.topk_ratio = float(topk_ratio)
        self.aggregation_mode = str(aggregation_mode)
        self.mean_weight = float(mean_weight)
        self.topk_weight = float(topk_weight)
        self.damage_decision_threshold = float(damage_decision_threshold)

    def forward(self, feature_map: torch.Tensor, target_mask: torch.Tensor | None) -> dict[str, torch.Tensor | None]:
        damage_binary_logit = self.damage_head(feature_map)
        severity_corn_logits = self.severity_head(feature_map)
        severity_class_probabilities, _ = _dense_decode_corn(severity_corn_logits)
        p_damaged = torch.sigmoid(damage_binary_logit.float())
        no_damage = 1.0 - p_damaged
        damaged_class_probabilities = p_damaged * severity_class_probabilities
        pixel_class_probabilities = torch.cat([no_damage, damaged_class_probabilities], dim=1)
        pixel_class_probabilities = _normalize_probabilities(pixel_class_probabilities)
        severity_pred = severity_class_probabilities.argmax(dim=1) + 1
        pixel_pred_labels = torch.where(
            p_damaged.squeeze(1) >= float(self.damage_decision_threshold),
            severity_pred,
            torch.zeros_like(severity_pred),
        )
        pixel_valid_mask = resize_mask_to_feature_map(target_mask, pixel_class_probabilities.shape[-2:])
        if pixel_valid_mask is None:
            pixel_valid_mask = torch.ones(
                feature_map.size(0),
                1,
                feature_map.size(2),
                feature_map.size(3),
                device=feature_map.device,
                dtype=feature_map.dtype,
            )
        aggregated = aggregate_pixel_probabilities(
            pixel_class_probabilities,
            pixel_valid_mask,
            self.aggregation_mode,
            topk_ratio=self.topk_ratio,
            mean_weight=self.mean_weight,
            topk_weight=self.topk_weight,
        )
        return {
            "pixel_damage_binary_logit": damage_binary_logit,
            "pixel_severity_corn_logits": severity_corn_logits,
            "pixel_corn_logits": None,
            "pixel_class_probabilities": pixel_class_probabilities,
            "pixel_pred_labels": pixel_pred_labels,
            "pixel_valid_mask": pixel_valid_mask,
            **aggregated,
        }


class PixelFlatCORNHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        topk_ratio: float = 0.2,
        aggregation_mode: str = "mean_topk_mix",
        mean_weight: float = 0.7,
        topk_weight: float = 0.3,
    ) -> None:
        super().__init__()
        self.corn_head = _PixelConvHead(in_channels, hidden_dim, 3, dropout)
        self.topk_ratio = float(topk_ratio)
        self.aggregation_mode = str(aggregation_mode)
        self.mean_weight = float(mean_weight)
        self.topk_weight = float(topk_weight)

    def forward(self, feature_map: torch.Tensor, target_mask: torch.Tensor | None) -> dict[str, torch.Tensor | None]:
        pixel_corn_logits = self.corn_head(feature_map)
        pixel_class_probabilities, pixel_pred_labels = _dense_decode_corn(pixel_corn_logits)
        pixel_valid_mask = resize_mask_to_feature_map(target_mask, pixel_class_probabilities.shape[-2:])
        if pixel_valid_mask is None:
            pixel_valid_mask = torch.ones(
                feature_map.size(0),
                1,
                feature_map.size(2),
                feature_map.size(3),
                device=feature_map.device,
                dtype=feature_map.dtype,
            )
        aggregated = aggregate_pixel_probabilities(
            pixel_class_probabilities,
            pixel_valid_mask,
            self.aggregation_mode,
            topk_ratio=self.topk_ratio,
            mean_weight=self.mean_weight,
            topk_weight=self.topk_weight,
        )
        return {
            "pixel_damage_binary_logit": None,
            "pixel_severity_corn_logits": None,
            "pixel_corn_logits": pixel_corn_logits,
            "pixel_class_probabilities": pixel_class_probabilities,
            "pixel_pred_labels": pixel_pred_labels,
            "pixel_valid_mask": pixel_valid_mask,
            **aggregated,
        }


def build_pixel_ordinal_head(
    head_type: str,
    *,
    in_channels: int,
    hidden_dim: int,
    dropout: float,
    topk_ratio: float,
    aggregation_mode: str,
    mean_weight: float,
    topk_weight: float,
    damage_decision_threshold: float,
) -> nn.Module:
    head_type = str(head_type).lower()
    common_kwargs: dict[str, Any] = {
        "in_channels": in_channels,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "topk_ratio": topk_ratio,
        "aggregation_mode": aggregation_mode,
        "mean_weight": mean_weight,
        "topk_weight": topk_weight,
    }
    if head_type == "two_stage_ordinal":
        return PixelTwoStageOrdinalHead(
            **common_kwargs,
            damage_decision_threshold=damage_decision_threshold,
        )
    if head_type == "flat_corn":
        return PixelFlatCORNHead(**common_kwargs)
    raise ValueError(f"Unsupported pixel_line_head_type '{head_type}'.")
