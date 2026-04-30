from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from utils.misc import ensure_dir, load_label_png, polygon_to_mask, read_json, save_label_png, write_json


PIXEL_LABEL_OFFSET = 1
LABEL_COLORS = np.asarray(
    [
        [0, 0, 0],
        [78, 121, 167],
        [242, 142, 43],
        [225, 87, 89],
        [118, 183, 178],
    ],
    dtype=np.uint8,
)


def convert_instance_to_pixel_label(instance_label: int) -> int:
    return int(instance_label) + PIXEL_LABEL_OFFSET


def colorize_label_map(label_map: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(label_map, dtype=np.int64), 0, len(LABEL_COLORS) - 1)
    return LABEL_COLORS[clipped]


def save_colorized_label_png(path: str | Path, label_map: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    Image.fromarray(colorize_label_map(label_map), mode="RGB").save(path)


def load_tile_shape(post_label_path: str | Path) -> tuple[int, int]:
    payload = read_json(post_label_path)
    meta = payload.get("metadata", {})
    width = int(meta.get("width", meta.get("original_width", 1024)))
    height = int(meta.get("height", meta.get("original_height", 1024)))
    return height, width


def build_tile_instance_groups(dataset_samples: list[dict[str, Any]], prediction_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_index = {int(record["sample_index"]): record for record in prediction_records}
    groups: dict[str, dict[str, Any]] = {}
    for sample_index, record in by_index.items():
        sample = dataset_samples[sample_index]
        tile_id = str(sample["tile_id"])
        if tile_id not in groups:
            groups[tile_id] = {"tile_sample": sample, "instances": []}
        groups[tile_id]["instances"].append(
            {
                "polygon_xy": sample["polygon_xy"],
                "building_idx": sample["building_idx"],
                "gt_label": int(record["gt_label"]),
                "pred_label": int(record["pred_label"]),
            }
        )
    return groups


def _resolve_damage_target_png(tile_sample: dict[str, Any]) -> Path | None:
    post_image_path = Path(tile_sample["post_image"])
    targets_dir = post_image_path.parent.parent / "targets"
    tile_id = str(tile_sample["tile_id"])
    candidates = [
        targets_dir / f"{tile_id}_post_disaster_target.png",
        targets_dir / f"{tile_id}_damage_target.png",
        targets_dir / f"{tile_id}_post_disaster_damage_target.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def rasterize_instances(instances: list[dict[str, Any]], *, height: int, width: int, field: str, overlap_policy: str) -> np.ndarray:
    label_map = np.zeros((height, width), dtype=np.uint8)
    for instance in sorted(instances, key=lambda item: int(item["building_idx"])):
        mask = polygon_to_mask(instance["polygon_xy"], height, width, offset=(0.0, 0.0)).astype(bool)
        label = convert_instance_to_pixel_label(int(instance[field]))
        if overlap_policy == "max_label":
            label_map[mask] = np.maximum(label_map[mask], label)
        elif overlap_policy == "last_wins":
            label_map[mask] = label
        else:
            raise ValueError(f"Unsupported overlap policy '{overlap_policy}'")
    return label_map


def compute_bridge_metrics(pred_map: np.ndarray, gt_map: np.ndarray) -> dict[str, Any]:
    gt_building = gt_map > 0
    pred_binary = (pred_map > 0).astype(np.uint8)
    gt_binary = gt_building.astype(np.uint8)
    tp = int(np.logical_and(pred_binary == 1, gt_binary == 1).sum())
    fp = int(np.logical_and(pred_binary == 1, gt_binary == 0).sum())
    fn = int(np.logical_and(pred_binary == 0, gt_binary == 1).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if (precision + recall) <= 0.0 else (2.0 * precision * recall) / (precision + recall)
    per_class = {}
    f1_values = []
    for class_id, name in zip([1, 2, 3, 4], ["no_damage", "minor", "major", "destroyed"]):
        tp_c = int(np.logical_and(pred_map == class_id, gt_map == class_id).sum())
        fp_c = int(np.logical_and(pred_map == class_id, gt_map != class_id).sum())
        fn_c = int(np.logical_and(pred_map != class_id, gt_map == class_id).sum())
        precision_c = tp_c / max(tp_c + fp_c, 1)
        recall_c = tp_c / max(tp_c + fn_c, 1)
        f1_c = 0.0 if (precision_c + recall_c) <= 0.0 else (2.0 * precision_c * recall_c) / (precision_c + recall_c)
        per_class[name] = {"precision": precision_c, "recall": recall_c, "f1": f1_c}
        f1_values.append(max(f1_c, 1e-8))
    hmean = float(len(f1_values) / np.sum([1.0 / value for value in f1_values])) if f1_values else 0.0
    return {
        "localization_f1": float(f1),
        "damage_f1_hmean": hmean,
        "per_class": per_class,
    }


def build_tile_bridge_result(tile_group: dict[str, Any], *, overlap_policy: str = "max_label") -> dict[str, Any]:
    tile_sample = tile_group["tile_sample"]
    instances = tile_group["instances"]
    height, width = load_tile_shape(tile_sample["post_label"])
    pred_map = rasterize_instances(instances, height=height, width=width, field="pred_label", overlap_policy=overlap_policy)
    target_png = _resolve_damage_target_png(tile_sample)
    if target_png is not None:
        gt_map = load_label_png(target_png)
    else:
        gt_map = rasterize_instances(instances, height=height, width=width, field="gt_label", overlap_policy=overlap_policy)
    tile_scores = compute_bridge_metrics(pred_map, gt_map)
    tile_scores["tile_id"] = tile_sample["tile_id"]
    return {
        "prediction_map": pred_map,
        "gt_damage_map": gt_map,
        "tile_scores": tile_scores,
    }


def aggregate_xview2_style_metrics(per_tile_scores: list[dict[str, Any]]) -> dict[str, Any]:
    if not per_tile_scores:
        return {"localization_f1": 0.0, "damage_f1_hmean": 0.0, "num_tiles": 0}
    return {
        "localization_f1": float(np.mean([item["localization_f1"] for item in per_tile_scores])),
        "damage_f1_hmean": float(np.mean([item["damage_f1_hmean"] for item in per_tile_scores])),
        "num_tiles": len(per_tile_scores),
    }


def run_bridge_evaluation(dataset, prediction_records: list[dict[str, Any]], *, config: dict[str, Any], save_dir: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bridge_cfg = config["bridge"]
    overlap_policy = str(bridge_cfg.get("overlap_policy", "max_label"))
    save_dir = ensure_dir(save_dir)
    tile_groups = build_tile_instance_groups(dataset.samples, prediction_records)
    per_tile_scores = []
    for tile_id, tile_group in sorted(tile_groups.items()):
        tile_result = build_tile_bridge_result(tile_group, overlap_policy=overlap_policy)
        per_tile_scores.append(tile_result["tile_scores"])
        if bool(bridge_cfg.get("save_pixel_predictions", True)):
            save_label_png(Path(save_dir) / "pixel_predictions" / f"{tile_id}_damage_prediction.png", tile_result["prediction_map"])
        if bool(bridge_cfg.get("save_visuals", True)):
            save_colorized_label_png(Path(save_dir) / "visuals" / f"{tile_id}_pred_color.png", tile_result["prediction_map"])
            save_colorized_label_png(Path(save_dir) / "visuals" / f"{tile_id}_gt_color.png", tile_result["gt_damage_map"])
    metrics = aggregate_xview2_style_metrics(per_tile_scores)
    write_json(Path(save_dir) / "bridge_metrics.json", metrics)
    write_json(Path(save_dir) / "per_tile_scores.json", per_tile_scores)
    return metrics, per_tile_scores

