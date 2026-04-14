from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from datasets.transforms import build_transforms
from datasets.xbd_json import build_instance_samples, polygon_to_mask

ImageFile.LOAD_TRUNCATED_IMAGES = True


def xbd_instance_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pre_image": torch.stack([item["pre_image"] for item in batch], dim=0),
        "post_image": torch.stack([item["post_image"] for item in batch], dim=0),
        "instance_mask": torch.stack([item["instance_mask"] for item in batch], dim=0),
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "sample_index": torch.tensor([item["sample_index"] for item in batch], dtype=torch.long),
        "meta": [item["meta"] for item in batch],
    }


class XBDInstanceDamageDataset(Dataset):
    def __init__(self, config: dict[str, Any], split_name: str, is_train: bool = False) -> None:
        super().__init__()
        self.config = config
        self.split_name = str(split_name)
        self.is_train = bool(is_train)
        self.transform = build_transforms(config, is_train=is_train)
        self.samples = build_instance_samples(
            root=config["data"]["root"],
            split=self.split_name,
            crop_context_ratio=float(config["data"]["crop_context_ratio"]),
            min_crop_size=int(config["data"]["min_crop_size"]),
            min_polygon_area=float(config["data"].get("min_polygon_area", 16.0)),
            min_mask_pixels=int(config["data"].get("min_mask_pixels", 16)),
            max_out_of_bound_ratio=float(config["data"].get("max_out_of_bound_ratio", 0.4)),
            allow_tier3=bool(config["data"].get("allow_tier3", False)),
        )
        self.class_counts = [0, 0, 0, 0]
        for sample in self.samples:
            self.class_counts[int(sample["label"])] += 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        pre_image = Image.open(sample["pre_image"]).convert("RGB")
        post_image = Image.open(sample["post_image"]).convert("RGB")
        x1, y1, x2, y2 = sample["crop_bbox_xyxy"]
        pre_crop = pre_image.crop((x1, y1, x2, y2))
        post_crop = post_image.crop((x1, y1, x2, y2))

        crop_w = x2 - x1
        crop_h = y2 - y1
        mask_np = polygon_to_mask(sample["polygon_xy"], crop_h, crop_w, offset=(x1, y1))
        mask = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")

        pre_tensor, post_tensor, mask_tensor = self.transform(pre_crop, post_crop, mask)
        if float(mask_tensor.sum().item()) <= 0:
            mask_tensor = torch.ones_like(mask_tensor)

        return {
            "pre_image": pre_tensor,
            "post_image": post_tensor,
            "instance_mask": mask_tensor,
            "label": int(sample["label"]),
            "sample_index": int(index),
            "meta": {
                "tile_id": sample["tile_id"],
                "building_idx": int(sample["building_idx"]),
                "uid": sample.get("uid"),
                "source_subset": sample["source_subset"],
                "split": sample["split"],
                "original_subtype": sample["original_subtype"],
                "crop_bbox_xyxy": list(sample["crop_bbox_xyxy"]),
            },
        }
