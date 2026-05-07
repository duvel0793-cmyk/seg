from __future__ import annotations

import torch

from utils.misc import SCALE_NAMES


def xbd_instance_collate_fn(batch: list[dict]) -> dict:
    collated = {
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "sample_index": torch.tensor([item["sample_index"] for item in batch], dtype=torch.long),
        "meta": [item["meta"] for item in batch],
    }
    for scale_name in SCALE_NAMES:
        pre_key = f"pre_{scale_name}"
        post_key = f"post_{scale_name}"
        mask_key = f"mask_{scale_name}"
        if pre_key not in batch[0]:
            continue
        collated[pre_key] = torch.stack([item[pre_key] for item in batch], dim=0)
        collated[post_key] = torch.stack([item[post_key] for item in batch], dim=0)
        collated[mask_key] = torch.stack([item[mask_key] for item in batch], dim=0)
        target_key = f"post_target_{scale_name}"
        if target_key in batch[0]:
            collated[target_key] = torch.stack([item[target_key] for item in batch], dim=0)
    if "augmentation_stats" in batch[0]:
        collated["augmentation_stats"] = [item["augmentation_stats"] for item in batch]
    return collated
