"""Polygon masked pooling for instance-level auxiliary supervision."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw


def _rasterize_polygon(polygon: Dict[str, object], height: int, width: int) -> np.ndarray:
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    parts = polygon.get("parts", [])
    holes = polygon.get("holes", [])

    for part_idx, part in enumerate(parts):
        if len(part) < 3:
            continue
        draw.polygon(part, outline=1, fill=1)
        polygon_holes = holes[part_idx] if part_idx < len(holes) else []
        for hole in polygon_holes:
            if len(hole) >= 3:
                draw.polygon(hole, outline=0, fill=0)
    return np.asarray(canvas, dtype=np.uint8) > 0


class PolygonPooling(nn.Module):
    """Pool per-pixel representations inside each GT polygon."""

    def __init__(self, source: str = "ordinal_logits", min_pixels: int = 16, ignore_index: int = 255) -> None:
        super().__init__()
        source = str(source).lower()
        alias = {"logits": "ordinal_logits", "feature": "fused_feature"}
        self.source = alias.get(source, source)
        if self.source not in {"ordinal_logits", "fused_feature", "both"}:
            raise ValueError(f"Unsupported polygon pool source: {source}")
        self.min_pixels = int(min_pixels)
        self.ignore_index = int(ignore_index)

    @staticmethod
    def _pool_masked_mean(feature_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled = feature_map[:, mask]
        if pooled.numel() == 0:
            return feature_map.new_zeros((feature_map.shape[0],))
        return pooled.mean(dim=1)

    def _resolve_target_label(
        self,
        polygon: Dict[str, object],
        damage_targets: torch.Tensor | None,
        mask: torch.Tensor,
    ) -> tuple[int | None, str]:
        polygon_rank = polygon.get("damage_rank")
        if polygon_rank is not None:
            return int(polygon_rank), "polygon_subtype"
        if damage_targets is None:
            return None, "missing"
        labels = damage_targets[mask]
        labels = labels[labels != self.ignore_index]
        if labels.numel() == 0:
            return None, "missing"
        return int(torch.bincount(labels, minlength=4).argmax().item()), "pixel_majority"

    def forward(
        self,
        ordinal_logits: torch.Tensor,
        polygons: List[List[Dict[str, object]]],
        damage_targets: torch.Tensor | None = None,
        fused_feature: torch.Tensor | None = None,
    ) -> Dict[str, object]:
        if self.source in {"ordinal_logits", "both"} and ordinal_logits is None:
            raise ValueError("PolygonPooling requires ordinal_logits for source=ordinal_logits/both")
        if self.source in {"fused_feature", "both"} and fused_feature is None:
            raise ValueError("PolygonPooling requires fused_feature for source=fused_feature/both")

        batch_size = len(polygons)
        ref_tensor = ordinal_logits if ordinal_logits is not None else fused_feature
        if ref_tensor is None:
            raise ValueError("PolygonPooling needs at least one tensor source.")
        height, width = ref_tensor.shape[-2:]

        pooled_repr = []
        pooled_targets = []
        pooled_counts = []
        label_source_counts = {"polygon_subtype": 0, "pixel_majority": 0}

        for batch_idx in range(batch_size):
            polygon_list = polygons[batch_idx]
            for polygon in polygon_list:
                mask_np = _rasterize_polygon(polygon, height=height, width=width)
                mask = torch.from_numpy(mask_np).to(device=ref_tensor.device)

                if damage_targets is not None:
                    valid_mask = damage_targets[batch_idx] != self.ignore_index
                    mask = mask & valid_mask

                num_pixels = int(mask.sum().item())
                if num_pixels < self.min_pixels:
                    continue

                polygon_rank, label_source = self._resolve_target_label(polygon, damage_targets[batch_idx] if damage_targets is not None else None, mask)
                if polygon_rank is None:
                    continue
                if label_source in label_source_counts:
                    label_source_counts[label_source] += 1

                pooled_parts = []
                if self.source in {"ordinal_logits", "both"}:
                    pooled_parts.append(self._pool_masked_mean(ordinal_logits[batch_idx], mask))
                if self.source in {"fused_feature", "both"} and fused_feature is not None:
                    pooled_parts.append(self._pool_masked_mean(fused_feature[batch_idx], mask))

                pooled_repr.append(torch.cat(pooled_parts, dim=0))
                pooled_targets.append(int(polygon_rank))
                pooled_counts.append(num_pixels)

        if not pooled_repr:
            return {
                "pooled_representations": None,
                "targets": None,
                "valid_instances": 0,
                "instance_pixel_counts": [],
                "label_source_counts": label_source_counts,
                "pool_source": self.source,
            }

        return {
            "pooled_representations": torch.stack(pooled_repr, dim=0),
            "targets": torch.tensor(pooled_targets, device=ref_tensor.device, dtype=torch.long),
            "valid_instances": len(pooled_targets),
            "instance_pixel_counts": pooled_counts,
            "label_source_counts": label_source_counts,
            "pool_source": self.source,
        }

