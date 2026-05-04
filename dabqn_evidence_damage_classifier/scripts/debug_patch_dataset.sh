#!/usr/bin/env bash
set -euo pipefail

cd /home/lky/code/dabqn_evidence_damage_classifier

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_PATH="${1:-configs/stage1_localization.yaml}"
OUTPUT_DIR="${2:-outputs/debug_patch_dataset}"
NUM_PATCHES="${NUM_PATCHES:-200}"
SAVE_VISUALS="${SAVE_VISUALS:-24}"

"$PYTHON_BIN" - "$CONFIG_PATH" "$OUTPUT_DIR" "$NUM_PATCHES" "$SAVE_VISUALS" <<'PY'
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from datasets import XBDQueryDataset
from utils.misc import CLASS_NAMES, ensure_dir, load_yaml


def denormalize_image(tensor, mean, std):
    array = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = (array * np.asarray(std).reshape(1, 1, 3)) + np.asarray(mean).reshape(1, 1, 3)
    return np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)


def save_overlay(item, out_path: Path, mean, std):
    colors = [
        (255, 99, 71),
        (255, 215, 0),
        (135, 206, 250),
        (144, 238, 144),
    ]
    image = denormalize_image(item["post_image"], mean, std)
    canvas = Image.fromarray(image).convert("RGBA")
    draw = ImageDraw.Draw(canvas, mode="RGBA")
    target = item["target"]
    masks = target["masks"].numpy() if target["masks"].numel() > 0 else np.zeros((0, image.shape[0], image.shape[1]), dtype=np.float32)
    boxes = target["boxes"].numpy() if target["boxes"].numel() > 0 else np.zeros((0, 4), dtype=np.float32)
    labels = target["labels"].numpy() if target["labels"].numel() > 0 else np.zeros((0,), dtype=np.int64)

    for mask, box, label in zip(masks, boxes, labels):
        color = colors[int(label) % len(colors)]
        mask_bool = mask > 0.5
        overlay = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        overlay[mask_bool] = (color[0], color[1], color[2], 90)
        canvas = Image.alpha_composite(canvas, Image.fromarray(overlay, mode="RGBA"))
        draw = ImageDraw.Draw(canvas, mode="RGBA")
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        draw.rectangle((x1, y1, x2, y2), outline=color + (255,), width=2)
        draw.text((x1 + 2, y1 + 2), CLASS_NAMES[int(label)], fill=color + (255,))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out_path)


config_path = Path(sys.argv[1])
output_dir = ensure_dir(sys.argv[2])
num_patches = int(sys.argv[3])
save_visuals = int(sys.argv[4])

config = load_yaml(config_path)
dataset = XBDQueryDataset(config=config, split="train", is_train=True)
mean = list(config.get("augmentation", {}).get("normalize_mean", [0.485, 0.456, 0.406]))
std = list(config.get("augmentation", {}).get("normalize_std", [0.229, 0.224, 0.225]))

random.seed(42)
np.random.seed(42)

counts = []
failures = []
adaptive_shrink_count = 0
fallback_keep_count = 0
gt_over_query_count = 0
gt_over_max_instances_count = 0
empty_patch_count = 0
positive_patch_count = 0
saved_regular = 0
saved_fallback = 0

for patch_index in range(num_patches):
    sample_index = random.randrange(len(dataset))
    item = dataset[sample_index]
    meta = item["meta"]
    target = item["target"]
    labels = target["labels"].numpy()
    boxes = target["boxes"].numpy()
    boxes_norm = target["boxes_norm"].numpy()
    masks = target["masks"].numpy()
    h, w = item["scaled_size"]

    counts.append(int(meta["num_instances"]))
    adaptive_shrink_count += int(bool(meta["adaptive_shrink_applied"]))
    fallback_keep_count += int(bool(meta["fallback_keep_applied"]))
    gt_over_query_count += int(bool(meta["gt_over_query"]))
    gt_over_max_instances_count += int(bool(meta["gt_over_max_instances_per_patch"]))
    empty_patch_count += int(bool(meta["is_empty_patch"]))
    positive_patch_count += int(bool(meta["is_positive_patch"]))

    if labels.shape[0] != boxes.shape[0] or labels.shape[0] != boxes_norm.shape[0] or labels.shape[0] != masks.shape[0]:
        failures.append(("shape_mismatch", meta["tile_id"], meta["patch_box"], labels.shape, boxes.shape, boxes_norm.shape, masks.shape))
        continue

    for inst_idx, (label, box, box_norm, mask) in enumerate(zip(labels, boxes, boxes_norm, masks)):
        ys, xs = np.where(mask > 0.5)
        if ys.size == 0 or xs.size == 0:
            failures.append(("empty_mask", meta["tile_id"], meta["patch_box"], inst_idx))
            continue
        expected = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)
        if np.max(np.abs(expected - box)) > 1.01:
            failures.append(("box_mask_mismatch", meta["tile_id"], meta["patch_box"], inst_idx, expected.tolist(), box.tolist()))
        if not (0 <= box[0] <= box[2] <= w and 0 <= box[1] <= box[3] <= h):
            failures.append(("box_out_of_bounds", meta["tile_id"], meta["patch_box"], inst_idx, box.tolist(), (h, w)))
        if not np.all((box_norm >= -1e-6) & (box_norm <= 1.0 + 1e-6)):
            failures.append(("box_norm_out_of_bounds", meta["tile_id"], meta["patch_box"], inst_idx, box_norm.tolist()))
        if int(label) < 0 or int(label) > 3:
            failures.append(("bad_label", meta["tile_id"], meta["patch_box"], inst_idx, int(label)))

    if saved_regular < save_visuals:
        patch_name = f"patch_{patch_index:03d}_{meta['tile_id']}_{meta['patch_box'][0]}_{meta['patch_box'][1]}.png"
        save_overlay(item, Path(output_dir) / "patches" / patch_name, mean, std)
        saved_regular += 1
    if bool(meta["fallback_keep_applied"]) and saved_fallback < save_visuals:
        fallback_name = f"fallback_{patch_index:03d}_{meta['tile_id']}_{meta['patch_box'][0]}_{meta['patch_box'][1]}.png"
        save_overlay(item, Path(output_dir) / "fallback_patches" / fallback_name, mean, std)
        saved_fallback += 1

arr = np.asarray(counts, dtype=np.int64) if counts else np.zeros((0,), dtype=np.int64)
summary = {
    "num_patches": int(arr.size),
    "min": int(arr.min()) if arr.size else 0,
    "p50": float(np.percentile(arr, 50)) if arr.size else 0.0,
    "p90": float(np.percentile(arr, 90)) if arr.size else 0.0,
    "p95": float(np.percentile(arr, 95)) if arr.size else 0.0,
    "max": int(arr.max()) if arr.size else 0,
    "mean": float(arr.mean()) if arr.size else 0.0,
    "adaptive_shrink_count": int(adaptive_shrink_count),
    "fallback_keep_count": int(fallback_keep_count),
    "gt_over_num_queries_count": int(gt_over_query_count),
    "gt_over_max_instances_count": int(gt_over_max_instances_count),
    "empty_patch_count": int(empty_patch_count),
    "positive_patch_count": int(positive_patch_count),
}

print("PATCH_GT_FAILURES", len(failures))
for failure in failures[:10]:
    print("FAIL", failure)
print("PATCH_DISTRIBUTION", summary)
PY
