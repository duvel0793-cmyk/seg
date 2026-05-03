from __future__ import annotations

from typing import Any

import torch


def xbd_query_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    collated = {
        "pre_image": torch.stack([item["pre_image"] for item in batch], dim=0),
        "post_image": torch.stack([item["post_image"] for item in batch], dim=0),
        "targets": [item["target"] for item in batch],
        "image_id": [item["image_id"] for item in batch],
        "original_size": [item["original_size"] for item in batch],
        "scaled_size": [item["scaled_size"] for item in batch],
        "meta": [item["meta"] for item in batch],
    }
    if all(item.get("gt_damage_map") is not None for item in batch):
        collated["gt_damage_map"] = torch.stack([item["gt_damage_map"] for item in batch], dim=0)
    return collated
