#!/usr/bin/env bash
set -euo pipefail

cd /home/lky/code/dabqn_evidence_damage_classifier

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_PATH="${1:-configs/stage1_localization.yaml}"
OUTPUT_DIR="${2:-outputs/debug_sliding_nms}"
IMAGE_ID="${IMAGE_ID:-}"

"$PYTHON_BIN" - "$CONFIG_PATH" "$OUTPUT_DIR" "$IMAGE_ID" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from datasets import XBDQueryDataset
from utils.mask_nms import compute_patch_center_weight, duplicate_metrics, suppress_duplicate_predictions
from utils.misc import ensure_dir, load_yaml


def draw_predictions(base_image: Image.Image, predictions, output_path: Path, color):
    canvas = base_image.convert("RGBA")
    draw = ImageDraw.Draw(canvas, mode="RGBA")
    for prediction in predictions:
        overlay = np.zeros((canvas.height, canvas.width, 4), dtype=np.uint8)
        patch_x1, patch_y1, _, _ = prediction["patch_box"]
        mask = np.asarray(prediction["mask"], dtype=bool)
        h, w = mask.shape
        overlay[patch_y1 : patch_y1 + h, patch_x1 : patch_x1 + w][mask] = color
        canvas = Image.alpha_composite(canvas, Image.fromarray(overlay, mode="RGBA"))
        draw = ImageDraw.Draw(canvas, mode="RGBA")
        draw.rectangle(tuple(prediction["box_xyxy"]), outline=color[:3] + (255,), width=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output_path)


def draw_suppressed_pairs(base_image: Image.Image, suppressed_pairs, output_path: Path):
    canvas = base_image.convert("RGBA")
    draw = ImageDraw.Draw(canvas, mode="RGBA")
    palette = [
        (255, 99, 71, 110),
        (65, 105, 225, 110),
        (50, 205, 50, 110),
        (255, 215, 0, 110),
    ]
    for pair_index, pair in enumerate(suppressed_pairs[:25]):
        kept = pair["kept_prediction"]
        suppressed = pair["suppressed_prediction"]
        kept_color = palette[(pair_index * 2) % len(palette)]
        suppressed_color = palette[(pair_index * 2 + 1) % len(palette)]
        for prediction, color in ((kept, kept_color), (suppressed, suppressed_color)):
            overlay = np.zeros((canvas.height, canvas.width, 4), dtype=np.uint8)
            patch_x1, patch_y1, _, _ = prediction["patch_box"]
            mask = np.asarray(prediction["mask"], dtype=bool)
            h, w = mask.shape
            overlay[patch_y1 : patch_y1 + h, patch_x1 : patch_x1 + w][mask] = color
            canvas = Image.alpha_composite(canvas, Image.fromarray(overlay, mode="RGBA"))
            draw = ImageDraw.Draw(canvas, mode="RGBA")
            draw.rectangle(tuple(prediction["box_xyxy"]), outline=color[:3] + (255,), width=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output_path)


config = load_yaml(sys.argv[1])
output_dir = ensure_dir(sys.argv[2])
target_image_id = sys.argv[3].strip()
dataset = XBDQueryDataset(config=config, split="val", is_train=False)
eval_cfg = config.get("evaluation", config.get("eval", {}))

sample_index = None
for idx, sample in enumerate(dataset.samples):
    if target_image_id and sample["tile_id"] != target_image_id:
        continue
    gt = dataset.build_full_image_ground_truth(idx)
    if len(gt["gt_instances"]) > 0:
        sample_index = idx
        break
if sample_index is None:
    raise RuntimeError("No suitable validation image found for sliding NMS debug.")

sample = dataset.samples[sample_index]
ground_truth = dataset.build_full_image_ground_truth(sample_index)
patch_boxes = dataset.get_sliding_window_boxes(sample_index, patch_size=int(eval_cfg.get("patch_size", 512)), stride=int(eval_cfg.get("stride", 384)))
base_image = Image.open(sample["paths"].post_image).convert("RGB")

pseudo_predictions = []
for gt_index, instance in enumerate(ground_truth["gt_instances"]):
    ys, xs = np.where(instance["mask"])
    if ys.size == 0:
        continue
    global_box = [float(xs.min()), float(ys.min()), float(xs.max()) + 1.0, float(ys.max()) + 1.0]
    for patch_box in patch_boxes:
        px1, py1, px2, py2 = patch_box
        local_mask = instance["mask"][py1:py2, px1:px2]
        if not local_mask.any():
            continue
        local_ys, local_xs = np.where(local_mask)
        local_box = [float(local_xs.min()), float(local_ys.min()), float(local_xs.max()) + 1.0, float(local_ys.max()) + 1.0]
        center_weight, touches_border, distance_ratio = compute_patch_center_weight(
            local_box_xyxy=local_box,
            patch_width=int(px2 - px1),
            patch_height=int(py2 - py1),
            use_patch_center_score=bool(eval_cfg.get("use_patch_center_score", True)),
            patch_border_penalty=float(eval_cfg.get("patch_border_penalty", 0.85)),
            border_margin=int(eval_cfg.get("border_margin", 32)),
        )
        pseudo_predictions.append(
            {
                "image_id": ground_truth["image_id"],
                "query_index": len(pseudo_predictions),
                "score": 0.99,
                "adjusted_score": 0.99 * center_weight,
                "center_prior_weight": center_weight,
                "touches_patch_border": touches_border,
                "patch_center_distance_ratio": distance_ratio,
                "label": int(instance["label"]),
                "box_xyxy": global_box,
                "local_box_xyxy": local_box,
                "patch_box": (int(px1), int(py1), int(px2), int(py2)),
                "mask": local_mask.astype(bool),
                "mask_area": int(local_mask.sum()),
                "probabilities": None,
            }
        )

kept_predictions, merge_stats, suppressed_pairs = suppress_duplicate_predictions(
    pseudo_predictions,
    mask_nms_iou=float(eval_cfg.get("mask_nms_iou", 0.45)),
    box_nms_iou=float(eval_cfg.get("box_nms_iou", 0.55)),
    containment_nms_thr=float(eval_cfg.get("containment_nms_thr", 0.75)),
    center_distance_ratio=float(eval_cfg.get("center_distance_ratio", 0.35)),
    max_predictions_per_image=int(eval_cfg.get("max_predictions_per_image", 1000)),
)

draw_predictions(base_image, pseudo_predictions, Path(output_dir) / "debug_nms_before.png", (255, 99, 71, 70))
draw_predictions(base_image, kept_predictions, Path(output_dir) / "debug_nms_after.png", (65, 105, 225, 90))
draw_suppressed_pairs(base_image, suppressed_pairs, Path(output_dir) / "debug_nms_suppressed_pairs.png")

print("SLIDING_NMS_DEBUG", {
    "image_id": ground_truth["image_id"],
    "gt_instances": len(ground_truth["gt_instances"]),
    "predictions_before_nms": len(pseudo_predictions),
    "predictions_after_nms": len(kept_predictions),
    "suppressed_pairs_count": len(suppressed_pairs),
    "suppressed_by_mask_iou": int(merge_stats["suppressed_by_mask_iou"]),
    "suppressed_by_box_iou": int(merge_stats["suppressed_by_box_iou"]),
    "suppressed_by_containment": int(merge_stats["suppressed_by_containment"]),
    "suppressed_by_center_distance": int(merge_stats["suppressed_by_center_distance"]),
})
for pair in suppressed_pairs[:25]:
    metrics = duplicate_metrics(pair["suppressed_prediction"], pair["kept_prediction"])
    print("PAIR", {
        "reason": pair["reason"],
        "mask_iou": float(metrics["mask_iou"]),
        "box_iou": float(metrics["box_iou"]),
        "containment": float(metrics["containment"]),
        "center_distance": float(metrics["center_distance"]),
        "center_distance_ratio": float(metrics["center_distance_ratio"]),
        "score_i": float(pair["kept_prediction"]["adjusted_score"]),
        "score_j": float(pair["suppressed_prediction"]["adjusted_score"]),
    })
PY
