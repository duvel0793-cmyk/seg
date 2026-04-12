"""Custom collate for mixed tensor/list samples."""

from __future__ import annotations

from typing import Dict, List

import torch


def xbd_collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "pre_image": torch.stack([item["pre_image"] for item in batch], dim=0),
        "post_image": torch.stack([item["post_image"] for item in batch], dim=0),
        "post_target": torch.stack([item["post_target"] for item in batch], dim=0),
        "loc_target": torch.stack([item["loc_target"] for item in batch], dim=0),
        "damage_rank_target": torch.stack([item["damage_rank_target"] for item in batch], dim=0),
        "polygons": [item["polygons"] for item in batch],
        "image_id": [item["image_id"] for item in batch],
        "meta": [item["meta"] for item in batch],
    }
