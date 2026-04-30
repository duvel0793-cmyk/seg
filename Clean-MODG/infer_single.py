"""Single-sample inference helper for Clean-MODG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from data.crop_policy import build_tight_context_crops, polygon_to_bbox
from data.mask_ops import bbox_to_mask, polygon_to_mask
from data.transforms import apply_sync_transform, build_sync_transform, image_to_tensor, mask_to_tensor
from losses.corn_loss import corn_predict
from models import build_model
from utils.checkpoint import load_checkpoint
from utils.common import CLASS_NAMES, outputs_to_probs
from utils.config import load_config
from utils.distributed import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-sample inference for Clean-MODG.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path.")
    parser.add_argument("--pre-image", type=str, required=True, help="Pre-disaster image path.")
    parser.add_argument("--post-image", type=str, required=True, help="Post-disaster image path.")
    parser.add_argument("--polygon", type=str, default="", help="Polygon JSON string.")
    parser.add_argument("--bbox", type=str, default="", help="BBox JSON string [x0,y0,x1,y1].")
    return parser.parse_args()


def _parse_geometry(polygon_str: str, bbox_str: str, image_shape: tuple[int, int]) -> tuple[list[list[float]] | None, list[float], np.ndarray]:
    polygon = json.loads(polygon_str) if polygon_str else None
    bbox = json.loads(bbox_str) if bbox_str else None
    if polygon is not None:
        bbox = polygon_to_bbox(polygon)
        mask = polygon_to_mask(image_shape, polygon)
        return polygon, bbox, mask
    if bbox is not None:
        mask = bbox_to_mask(image_shape, bbox)
        return None, bbox, mask
    raise ValueError("Please provide either --polygon or --bbox.")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = get_device()
    model = build_model(config).to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    model.eval()

    pre_image = np.asarray(Image.open(args.pre_image).convert("RGB"))
    post_image = np.asarray(Image.open(args.post_image).convert("RGB"))
    polygon, bbox, mask = _parse_geometry(args.polygon, args.bbox, pre_image.shape[:2])
    crops = build_tight_context_crops(
        pre_image=pre_image,
        post_image=post_image,
        mask=mask,
        bbox=bbox,
        tight_size=int(config["data"].get("tight_size", 256)),
        context_size=int(config["data"].get("context_size", 256)),
        tight_padding_ratio=float(config["data"].get("tight_padding_ratio", 0.15)),
        context_padding_ratio=float(config["data"].get("context_padding_ratio", 1.0)),
    )
    transform = build_sync_transform(config["data"], is_train=False)
    transformed = apply_sync_transform(transform, crops)
    mask_tight = transformed["mask_tight"].astype(np.uint8)
    mask_context = transformed["mask_context"].astype(np.uint8)
    pre_tight = transformed["pre_tight"]
    post_tight = transformed["post_tight"]
    pre_context = transformed["pre_context"]
    post_context = transformed["post_context"]
    if config["data"].get("mask_mode") == "rgbm":
        pre_tight = np.concatenate([pre_tight, mask_tight[..., None].astype(np.float32)], axis=2)
        post_tight = np.concatenate([post_tight, mask_tight[..., None].astype(np.float32)], axis=2)
        pre_context = np.concatenate([pre_context, mask_context[..., None].astype(np.float32)], axis=2)
        post_context = np.concatenate([post_context, mask_context[..., None].astype(np.float32)], axis=2)

    batch = {
        "pre_tight": image_to_tensor(pre_tight).unsqueeze(0).to(device),
        "post_tight": image_to_tensor(post_tight).unsqueeze(0).to(device),
        "pre_context": image_to_tensor(pre_context).unsqueeze(0).to(device),
        "post_context": image_to_tensor(post_context).unsqueeze(0).to(device),
        "mask_tight": mask_to_tensor(mask_tight).unsqueeze(0).to(device),
        "mask_context": mask_to_tensor(mask_context).unsqueeze(0).to(device),
    }
    with torch.no_grad():
        outputs = model(batch)
        probs = outputs_to_probs(outputs)[0].detach().cpu().numpy()
        pred = int(np.argmax(probs))
    print({"pred": pred, "label_name": CLASS_NAMES[pred], "probabilities": probs.tolist()})


if __name__ == "__main__":
    main()
