from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from utils.misc import (
    ensure_dir,
    load_label_png,
    parse_wkt_polygon,
    polygon_area,
    polygon_to_mask,
    read_json,
    save_label_png,
    write_json,
    write_text,
)


PIXEL_LABEL_OFFSET = 1
PIXEL_CLASS_NAMES = {
    0: "background",
    1: "no_damage",
    2: "minor_damage",
    3: "major_damage",
    4: "destroyed",
}
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


def instance_label_to_pixel_label(label: int) -> int:
    return int(label) + 1


def colorize_label_map(label_map: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(label_map, dtype=np.int64), 0, len(LABEL_COLORS) - 1)
    return LABEL_COLORS[clipped]


def save_colorized_label_png(path: str | Path, label_map: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    Image.fromarray(colorize_label_map(label_map), mode="RGB").save(path)


def save_overlay_png(path: str | Path, pred_map: np.ndarray, gt_map: np.ndarray) -> None:
    pred_color = colorize_label_map(pred_map).astype(np.float32)
    gt_color = colorize_label_map(gt_map).astype(np.float32)
    mismatch = pred_map != gt_map
    overlay = gt_color.copy()
    overlay[mismatch] = (0.55 * pred_color[mismatch]) + (0.45 * gt_color[mismatch])
    overlay = np.clip(overlay, 0.0, 255.0).astype(np.uint8)
    path = Path(path)
    ensure_dir(path.parent)
    Image.fromarray(overlay, mode="RGB").save(path)


def _safe_divide(numerator: float, denominator: float) -> float:
    if float(denominator) <= 0.0:
        return 0.0
    return float(numerator) / float(denominator)


def _f1_from_precision_recall(precision: float, recall: float) -> float:
    if (precision + recall) <= 0.0:
        return 0.0
    return float((2.0 * precision * recall) / (precision + recall))


def _validate_label_range(label_map: np.ndarray, num_classes: int) -> None:
    if label_map.size == 0:
        return
    arr = np.asarray(label_map)
    min_value = int(arr.min())
    max_value = int(arr.max())
    if min_value < 0 or max_value >= int(num_classes):
        raise ValueError(
            f"Label map contains values outside [0, {num_classes - 1}]: "
            f"min={min_value}, max={max_value}."
        )


def _compute_confusion_matrix(pred_map: np.ndarray, gt_map: np.ndarray, num_classes: int) -> np.ndarray:
    pred = np.asarray(pred_map, dtype=np.int64)
    gt = np.asarray(gt_map, dtype=np.int64)
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction/GT shape mismatch: pred={pred.shape}, gt={gt.shape}.")
    _validate_label_range(pred, num_classes)
    _validate_label_range(gt, num_classes)
    flat = (gt.reshape(-1) * int(num_classes)) + pred.reshape(-1)
    counts = np.bincount(flat, minlength=int(num_classes) * int(num_classes))
    return counts.reshape(int(num_classes), int(num_classes))


def _compute_metrics_from_confusion_matrix(
    confusion_matrix: np.ndarray,
    *,
    num_classes: int = 5,
    eps: float = 1.0e-8,
) -> dict[str, Any]:
    cm = np.asarray(confusion_matrix, dtype=np.int64)
    if cm.shape != (num_classes, num_classes):
        raise ValueError(f"Expected confusion matrix shape {(num_classes, num_classes)}, got {cm.shape}.")
    total = int(cm.sum())
    tp_loc = int(cm[1:, 1:].sum())
    fp_loc = int(cm[0, 1:].sum())
    fn_loc = int(cm[1:, 0].sum())
    tn_loc = int(cm[0, 0])
    loc_precision = _safe_divide(tp_loc, tp_loc + fp_loc)
    loc_recall = _safe_divide(tp_loc, tp_loc + fn_loc)
    localization_f1 = _f1_from_precision_recall(loc_precision, loc_recall)
    localization_accuracy = _safe_divide(tp_loc + tn_loc, total)
    localization_iou = _safe_divide(tp_loc, tp_loc + fp_loc + fn_loc)

    damage_per_class: dict[str, Any] = {}
    damage_f1_values: list[float] = []
    damage_supports: list[int] = []
    damage_tp_sum = 0
    damage_fp_sum = 0
    damage_fn_sum = 0
    correct_damage_pixels = 0
    for class_id in range(1, num_classes):
        class_name = PIXEL_CLASS_NAMES[class_id]
        tp = int(cm[class_id, class_id])
        fp = int(cm[:, class_id].sum() - tp)
        fn = int(cm[class_id, :].sum() - tp)
        support = int(cm[class_id, :].sum())
        pred_support = int(cm[:, class_id].sum())
        if support == 0 and pred_support == 0:
            precision = 1.0
            recall = 1.0
            f1 = 1.0
            iou = 1.0
        else:
            precision = _safe_divide(tp, tp + fp)
            recall = _safe_divide(tp, tp + fn)
            f1 = _f1_from_precision_recall(precision, recall)
            iou = _safe_divide(tp, tp + fp + fn)
        damage_per_class[class_name] = {
            "class_id": class_id,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "support": support,
        }
        damage_f1_values.append(max(f1, float(eps)))
        damage_supports.append(support)
        damage_tp_sum += tp
        damage_fp_sum += fp
        damage_fn_sum += fn
        correct_damage_pixels += tp

    damage_f1_hmean = 0.0
    if damage_f1_values:
        damage_f1_hmean = float(len(damage_f1_values) / sum((1.0 / value) for value in damage_f1_values))
    damage_macro_f1 = float(np.mean([values["f1"] for values in damage_per_class.values()])) if damage_per_class else 0.0
    total_damage_support = int(sum(damage_supports))
    damage_weighted_f1 = 0.0
    if total_damage_support > 0:
        damage_weighted_f1 = float(
            sum(values["f1"] * values["support"] for values in damage_per_class.values()) / total_damage_support
        )
    damage_micro_precision = _safe_divide(damage_tp_sum, damage_tp_sum + damage_fp_sum)
    damage_micro_recall = _safe_divide(damage_tp_sum, damage_tp_sum + damage_fn_sum)
    damage_micro_f1 = _f1_from_precision_recall(damage_micro_precision, damage_micro_recall)
    damage_pixel_accuracy_on_building_gt = _safe_divide(correct_damage_pixels, int(cm[1:, :].sum()))
    damage_pixel_accuracy_on_union_building = _safe_divide(correct_damage_pixels, total - int(cm[0, 0]))
    overall_pixel_accuracy_all = _safe_divide(int(np.trace(cm)), total)

    xview2_overall_score = (0.3 * localization_f1) + (0.7 * damage_f1_hmean)
    return {
        "num_pixels": total,
        "localization_tp": tp_loc,
        "localization_fp": fp_loc,
        "localization_fn": fn_loc,
        "localization_tn": tn_loc,
        "localization_precision": loc_precision,
        "localization_recall": loc_recall,
        "localization_f1": localization_f1,
        "localization_accuracy": localization_accuracy,
        "localization_iou": localization_iou,
        "damage_per_class": damage_per_class,
        "damage_f1_hmean": damage_f1_hmean,
        "damage_macro_f1": damage_macro_f1,
        "damage_weighted_f1": damage_weighted_f1,
        "damage_micro_precision": damage_micro_precision,
        "damage_micro_recall": damage_micro_recall,
        "damage_micro_f1": damage_micro_f1,
        "damage_pixel_accuracy_on_building_gt": damage_pixel_accuracy_on_building_gt,
        "damage_pixel_accuracy_on_union_building": damage_pixel_accuracy_on_union_building,
        "overall_pixel_accuracy_all": overall_pixel_accuracy_all,
        "confusion_matrix_5x5": cm.tolist(),
        "xview2_overall_score": xview2_overall_score,
    }


def compute_pixel_bridge_metrics(
    pred_map: np.ndarray,
    gt_map: np.ndarray,
    num_classes: int = 5,
) -> dict[str, Any]:
    confusion_matrix = _compute_confusion_matrix(pred_map=pred_map, gt_map=gt_map, num_classes=num_classes)
    return _compute_metrics_from_confusion_matrix(confusion_matrix, num_classes=num_classes)


def compute_confusion_matrix(
    pred_map: np.ndarray,
    gt_map: np.ndarray,
    num_classes: int = 5,
) -> np.ndarray:
    return _compute_confusion_matrix(pred_map=pred_map, gt_map=gt_map, num_classes=num_classes)


def compute_metrics_from_confusion_matrix(
    confusion_matrix: np.ndarray,
    *,
    num_classes: int = 5,
    eps: float = 1.0e-8,
) -> dict[str, Any]:
    return _compute_metrics_from_confusion_matrix(confusion_matrix, num_classes=num_classes, eps=eps)


def _resolve_field(sample: dict[str, Any], candidates: list[str], *, required_name: str) -> Any:
    for key in candidates:
        if key in sample and sample[key] is not None:
            return sample[key]
    raise KeyError(f"Missing required field '{required_name}'. Looked for keys: {candidates}.")


def _resolve_tile_id(sample: dict[str, Any]) -> str:
    return str(_resolve_field(sample, ["tile_id"], required_name="tile_id"))


def _resolve_post_image_path(sample: dict[str, Any]) -> Path:
    return Path(str(_resolve_field(sample, ["post_image", "post_image_path"], required_name="post_image")))


def _resolve_post_label_path(sample: dict[str, Any]) -> Path:
    return Path(str(_resolve_field(sample, ["post_label", "post_label_path", "label_path"], required_name="post_label")))


def _resolve_building_id(sample: dict[str, Any]) -> int | str:
    value = _resolve_field(sample, ["building_idx", "building_id"], required_name="building_idx")
    return int(value) if isinstance(value, (int, np.integer, float)) else value


def _resolve_polygon_xy(sample: dict[str, Any]) -> list[tuple[float, float]]:
    if sample.get("polygon_xy") is not None:
        return [(float(x), float(y)) for x, y in sample["polygon_xy"]]
    if sample.get("polygon") is not None:
        return [(float(x), float(y)) for x, y in sample["polygon"]]
    if sample.get("wkt") is not None:
        return [(float(x), float(y)) for x, y in parse_wkt_polygon(str(sample["wkt"]))]
    raise KeyError("Missing polygon field. Expected one of: polygon_xy, polygon, wkt.")


def _resolve_image_shape(sample: dict[str, Any]) -> tuple[int, int]:
    image_size = sample.get("image_size")
    if isinstance(image_size, (list, tuple)) and len(image_size) >= 2:
        return int(image_size[0]), int(image_size[1])
    post_label_path = _resolve_post_label_path(sample)
    if post_label_path.exists():
        payload = read_json(post_label_path)
        meta = payload.get("metadata", {})
        width = meta.get("width", meta.get("original_width"))
        height = meta.get("height", meta.get("original_height"))
        if width is not None and height is not None:
            return int(height), int(width)
    post_image_path = _resolve_post_image_path(sample)
    if post_image_path.exists():
        with Image.open(post_image_path) as image:
            width, height = image.size
        return int(height), int(width)
    raise RuntimeError(
        "Unable to determine tile shape. Expected `image_size`, a readable post label json, or a readable post image."
    )


def _instance_to_binary_mask(instance: dict[str, Any], *, height: int, width: int) -> np.ndarray:
    predicted_mask = instance.get("predicted_mask")
    if predicted_mask is not None:
        mask = np.asarray(predicted_mask)
        if mask.shape != (height, width):
            raise ValueError(
                f"predicted_mask shape mismatch for tile instance: expected {(height, width)}, got {mask.shape}."
            )
        return mask.astype(bool)

    mask_path = instance.get("mask_path")
    if mask_path:
        mask = load_label_png(mask_path)
        if mask.shape != (height, width):
            raise ValueError(f"mask_path shape mismatch: expected {(height, width)}, got {mask.shape} from {mask_path}.")
        return mask.astype(bool)

    if instance.get("rle") is not None:
        raise NotImplementedError(
            "RLE-backed instance masks are not implemented yet in bridge eval. "
            "The rasterization entry point is ready, but this repository currently expects polygons or mask_path."
        )

    polygon_xy = _resolve_polygon_xy(instance)
    return polygon_to_mask(polygon_xy, height, width, offset=(0.0, 0.0)).astype(bool)


def rasterize_instances_to_damage_map(
    instances: list[dict[str, Any]],
    height: int,
    width: int,
    label_field: str,
    overlap_policy: str = "max_label",
) -> np.ndarray:
    if int(height) <= 0 or int(width) <= 0:
        raise ValueError(f"Invalid raster size: height={height}, width={width}.")
    label_map = np.zeros((int(height), int(width)), dtype=np.uint8)
    area_winner = np.full((int(height), int(width)), fill_value=-1.0, dtype=np.float32)

    def sort_key(item: dict[str, Any]) -> tuple[int, int]:
        building_id = item.get("building_idx", item.get("building_id", 0))
        sample_index = item.get("sample_index", 0)
        return int(building_id), int(sample_index)

    for instance in sorted(instances, key=sort_key):
        if label_field not in instance:
            raise KeyError(f"Instance is missing label field '{label_field}'.")
        instance_label = int(instance[label_field])
        pixel_label = instance_label_to_pixel_label(instance_label)
        mask = _instance_to_binary_mask(instance, height=int(height), width=int(width))
        if overlap_policy == "max_label":
            label_map[mask] = np.maximum(label_map[mask], pixel_label)
        elif overlap_policy == "last_wins":
            label_map[mask] = pixel_label
        elif overlap_policy == "area_larger_wins":
            region_area = float(instance.get("polygon_area", float(mask.sum())))
            replace = mask & (region_area >= area_winner)
            label_map[replace] = pixel_label
            area_winner[replace] = region_area
        else:
            raise ValueError(
                f"Unsupported overlap policy '{overlap_policy}'. "
                "Expected one of: max_label, last_wins, area_larger_wins."
            )
    return label_map


def _candidate_target_filenames(tile_id: str) -> list[str]:
    return [
        f"{tile_id}_post_disaster_target.png",
        f"{tile_id}_damage_target.png",
        f"{tile_id}_post_disaster_damage_target.png",
        f"{tile_id}_target.png",
    ]


def _resolve_official_target_png(tile_sample: dict[str, Any], config: dict[str, Any]) -> Path | None:
    tile_id = _resolve_tile_id(tile_sample)
    bridge_cfg = config.get("bridge", {}) if isinstance(config, dict) else {}
    candidate_paths: list[Path] = []
    existing_post_target = tile_sample.get("post_target")
    if existing_post_target:
        candidate_paths.append(Path(str(existing_post_target)))

    explicit_target_dir = bridge_cfg.get("target_dir")
    if explicit_target_dir:
        for name in _candidate_target_filenames(tile_id):
            candidate_paths.append(Path(str(explicit_target_dir)) / name)

    post_image_path = _resolve_post_image_path(tile_sample)
    candidate_dirs = [
        post_image_path.parent.parent / "targets",
        Path(str(config.get("data", {}).get("root", ""))) / "targets",
    ]
    source_subset = tile_sample.get("source_subset")
    if source_subset:
        candidate_dirs.append(Path(str(config.get("data", {}).get("root", ""))) / str(source_subset) / "targets")
    for directory in candidate_dirs:
        for name in _candidate_target_filenames(tile_id):
            candidate_paths.append(directory / name)

    seen: set[Path] = set()
    for candidate in candidate_paths:
        resolved = candidate.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


def _load_or_build_gt_map(
    tile_sample: dict[str, Any],
    gt_instances: list[dict[str, Any]],
    *,
    height: int,
    width: int,
    config: dict[str, Any],
    overlap_policy: str,
) -> tuple[np.ndarray, str]:
    target_png = _resolve_official_target_png(tile_sample, config)
    if target_png is not None:
        gt_map = load_label_png(target_png)
        if gt_map.shape != (height, width):
            raise ValueError(
                f"Official target PNG shape mismatch for tile {_resolve_tile_id(tile_sample)}: "
                f"expected {(height, width)}, got {gt_map.shape} from {target_png}."
            )
        _validate_label_range(gt_map, 5)
        return gt_map.astype(np.uint8), str(target_png)
    if gt_instances:
        gt_map = rasterize_instances_to_damage_map(
            gt_instances,
            height=height,
            width=width,
            label_field="gt_label",
            overlap_policy=overlap_policy,
        )
        return gt_map.astype(np.uint8), "rasterized_gt_polygons"
    raise RuntimeError(
        f"Unable to build GT damage map for tile {_resolve_tile_id(tile_sample)}. "
        "No official target PNG was found, and no GT polygons were available in dataset.samples."
    )


def _build_tile_groups(
    dataset_samples: list[dict[str, Any]],
    prediction_records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not prediction_records:
        raise ValueError("Bridge evaluation requires non-empty prediction_records.")

    gt_by_tile: dict[str, dict[str, Any]] = {}
    for sample_index, sample in enumerate(dataset_samples):
        tile_id = _resolve_tile_id(sample)
        group = gt_by_tile.setdefault(
            tile_id,
            {
                "tile_sample": sample,
                "gt_instances": [],
                "pred_instances": [],
            },
        )
        group["gt_instances"].append(
            {
                "sample_index": int(sample_index),
                "tile_id": tile_id,
                "building_idx": _resolve_building_id(sample),
                "polygon_xy": _resolve_polygon_xy(sample),
                "polygon_area": float(sample.get("polygon_area", polygon_area(_resolve_polygon_xy(sample)))),
                "gt_label": int(sample.get("label", sample.get("gt_label"))),
            }
        )

    active_tiles: dict[str, dict[str, Any]] = {}
    for record in prediction_records:
        sample_index = int(record["sample_index"])
        if sample_index < 0 or sample_index >= len(dataset_samples):
            raise IndexError(f"prediction_record sample_index out of bounds: {sample_index}.")
        sample = dataset_samples[sample_index]
        tile_id = _resolve_tile_id(sample)
        if tile_id not in gt_by_tile:
            raise RuntimeError(f"Tile {tile_id} was not indexed from dataset.samples.")
        tile_group = active_tiles.setdefault(
            tile_id,
            {
                "tile_sample": gt_by_tile[tile_id]["tile_sample"],
                "gt_instances": gt_by_tile[tile_id]["gt_instances"],
                "pred_instances": [],
            },
        )
        tile_group["pred_instances"].append(
            {
                "sample_index": sample_index,
                "tile_id": tile_id,
                "building_idx": _resolve_building_id(sample),
                "polygon_xy": _resolve_polygon_xy(sample),
                "polygon_area": float(sample.get("polygon_area", polygon_area(_resolve_polygon_xy(sample)))),
                "gt_label": int(record.get("gt_label", sample.get("label"))),
                "pred_label": int(record.get("final_pred_label", record.get("pred_label"))),
                "instance_pred_label": record.get("instance_pred_label", record.get("corn_pred_label")),
                "pixel_agg_pred_label": record.get("pixel_agg_pred_label"),
                "pixel_pred_map": record.get("pixel_pred_map"),
                "pixel_valid_mask": record.get("pixel_valid_mask"),
                "pixel_feature_source": record.get("pixel_feature_source", "tight"),
                "tight_bbox_xyxy": record.get("tight_bbox_xyxy", sample.get("tight_bbox_xyxy")),
                "context_bbox_xyxy": record.get("context_bbox_xyxy", sample.get("context_bbox_xyxy")),
                "neighborhood_bbox_xyxy": record.get("neighborhood_bbox_xyxy", sample.get("neighborhood_bbox_xyxy")),
                "probabilities": record.get("final_class_probabilities", record.get("class_probabilities")),
                "pixel_probabilities": record.get("pixel_instance_probabilities"),
            }
        )
    return active_tiles


def _normalize_prediction_source(prediction_source: str) -> str:
    source = str(prediction_source).lower()
    if source == "pixel":
        return "pixel_agg"
    if source not in {"instance", "pixel_agg", "pixel_dense"}:
        raise ValueError(f"Unsupported prediction_source '{prediction_source}'.")
    return source


def _resolve_prediction_label(instance: dict[str, Any], prediction_source: str) -> int:
    source = _normalize_prediction_source(prediction_source)
    if source == "instance":
        return int(instance.get("pred_label", instance.get("instance_pred_label", 0)))
    if source == "pixel_agg":
        value = instance.get("pixel_agg_pred_label")
        if value is None:
            raise KeyError("pixel_agg prediction requested, but prediction record is missing pixel_agg_pred_label.")
        return int(value)
    raise ValueError(f"Label-based prediction source '{prediction_source}' is not supported here.")


def _resize_uint8_map(label_map: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    return np.asarray(Image.fromarray(label_map.astype(np.uint8), mode="L").resize((width, height), resample=Image.NEAREST))


def _render_dense_instance_predictions(
    instances: list[dict[str, Any]],
    *,
    height: int,
    width: int,
    overlap_policy: str,
) -> np.ndarray:
    label_map = np.zeros((int(height), int(width)), dtype=np.uint8)
    area_winner = np.full((int(height), int(width)), fill_value=-1.0, dtype=np.float32)
    for instance in sorted(instances, key=lambda item: (int(item.get("building_idx", 0)), int(item.get("sample_index", 0)))):
        dense_pred = instance.get("pixel_pred_map")
        if dense_pred is None:
            raise KeyError("pixel_dense prediction requested, but prediction record is missing pixel_pred_map.")
        feature_source = str(instance.get("pixel_feature_source", "tight")).lower()
        bbox = instance.get(f"{feature_source}_bbox_xyxy") or instance.get("tight_bbox_xyxy")
        if bbox is None:
            raise KeyError(f"Missing bbox metadata required for pixel_dense bridge rendering (source={feature_source}).")
        x1, y1, x2, y2 = [int(v) for v in bbox]
        bbox_width = max(1, x2 - x1)
        bbox_height = max(1, y2 - y1)
        dense_pred_arr = np.asarray(dense_pred, dtype=np.uint8)
        dense_pred_resized = _resize_uint8_map(dense_pred_arr, (bbox_width, bbox_height))
        valid_mask = instance.get("pixel_valid_mask")
        if valid_mask is None:
            valid_mask_arr = np.ones((bbox_height, bbox_width), dtype=bool)
        else:
            valid_mask_arr = _resize_uint8_map(np.asarray(valid_mask, dtype=np.uint8), (bbox_width, bbox_height)) > 0
        polygon_mask = polygon_to_mask(instance["polygon_xy"], bbox_height, bbox_width, offset=(x1, y1)).astype(bool)
        effective_mask = valid_mask_arr & polygon_mask
        dense_pixel_label = np.where(effective_mask, dense_pred_resized + PIXEL_LABEL_OFFSET, 0).astype(np.uint8)
        canvas = np.zeros_like(label_map)
        canvas[y1:y2, x1:x2] = dense_pixel_label
        instance_mask = canvas > 0
        if overlap_policy == "max_label":
            label_map[instance_mask] = np.maximum(label_map[instance_mask], canvas[instance_mask])
        elif overlap_policy == "last_wins":
            label_map[instance_mask] = canvas[instance_mask]
        elif overlap_policy == "area_larger_wins":
            region_area = float(instance.get("polygon_area", float(instance_mask.sum())))
            replace = instance_mask & (region_area >= area_winner)
            label_map[replace] = canvas[replace]
            area_winner[replace] = region_area
        else:
            raise ValueError(
                f"Unsupported overlap policy '{overlap_policy}'. "
                "Expected one of: max_label, last_wins, area_larger_wins."
            )
    return label_map


def _average_metric_dicts(metric_dicts: list[dict[str, Any]]) -> dict[str, Any]:
    if not metric_dicts:
        return {}
    accumulator: dict[str, list[float]] = {}

    def collect(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "confusion_matrix_5x5":
                    continue
                next_prefix = f"{prefix}.{key}" if prefix else key
                collect(next_prefix, child)
        elif isinstance(value, (int, float, np.integer, np.floating)) and prefix:
            accumulator.setdefault(prefix, []).append(float(value))

    for item in metric_dicts:
        collect("", item)

    averaged: dict[str, Any] = {}
    for dotted_key, values in accumulator.items():
        cursor = averaged
        parts = dotted_key.split(".")
        for key in parts[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[parts[-1]] = float(np.mean(values)) if values else 0.0
    return averaged


def _build_class_report_text(
    *,
    global_metrics: dict[str, Any],
    per_tile_average: dict[str, Any],
    num_tiles: int,
    num_instances: int,
    num_gt_instances_in_evaluated_tiles: int,
    prediction_source: str = "instance",
) -> str:
    lines = [
        "Pixel Bridge Evaluation",
        f"prediction_source: {prediction_source}",
        f"num_tiles: {num_tiles}",
        f"num_instances: {num_instances}",
        f"num_gt_instances_in_evaluated_tiles: {num_gt_instances_in_evaluated_tiles}",
        "",
        "Global Summary",
        f"localization_f1: {global_metrics['localization_f1']:.6f}",
        f"damage_f1_hmean: {global_metrics['damage_f1_hmean']:.6f}",
        f"xview2_overall_score: {global_metrics['xview2_overall_score']:.6f}",
        f"damage_macro_f1: {global_metrics['damage_macro_f1']:.6f}",
        f"damage_weighted_f1: {global_metrics['damage_weighted_f1']:.6f}",
        f"damage_micro_f1: {global_metrics['damage_micro_f1']:.6f}",
        f"damage_pixel_accuracy_on_building_gt: {global_metrics['damage_pixel_accuracy_on_building_gt']:.6f}",
        f"damage_pixel_accuracy_on_union_building: {global_metrics['damage_pixel_accuracy_on_union_building']:.6f}",
        f"overall_pixel_accuracy_all: {global_metrics['overall_pixel_accuracy_all']:.6f}",
        "",
        "Metric Notes",
        "Global Summary is the primary metric.",
        "Per Tile Average is diagnostic only and can be misleading when some classes are absent in a tile.",
        "",
        "Localization",
        f"precision: {global_metrics['localization_precision']:.6f}",
        f"recall: {global_metrics['localization_recall']:.6f}",
        f"f1: {global_metrics['localization_f1']:.6f}",
        f"iou: {global_metrics['localization_iou']:.6f}",
        f"accuracy: {global_metrics['localization_accuracy']:.6f}",
        "",
        "Damage Per Class",
    ]
    for class_name in ["no_damage", "minor_damage", "major_damage", "destroyed"]:
        metrics = global_metrics["damage_per_class"][class_name]
        lines.append(
            f"{class_name}: precision={metrics['precision']:.6f} recall={metrics['recall']:.6f} "
            f"f1={metrics['f1']:.6f} iou={metrics['iou']:.6f} support={metrics['support']}"
        )
    if per_tile_average:
        lines.extend(
            [
                "",
                "Per Tile Average",
                f"localization_f1: {float(per_tile_average.get('localization_f1', 0.0)):.6f}",
                f"damage_f1_hmean: {float(per_tile_average.get('damage_f1_hmean', 0.0)):.6f}",
                f"xview2_overall_score: {float(per_tile_average.get('xview2_overall_score', 0.0)):.6f}",
            ]
        )
    return "\n".join(lines) + "\n"


def build_class_report_text(
    *,
    global_metrics: dict[str, Any],
    per_tile_average: dict[str, Any],
    num_tiles: int,
    num_instances: int,
    num_gt_instances_in_evaluated_tiles: int,
    prediction_source: str = "instance",
) -> str:
    return _build_class_report_text(
        global_metrics=global_metrics,
        per_tile_average=per_tile_average,
        num_tiles=num_tiles,
        num_instances=num_instances,
        num_gt_instances_in_evaluated_tiles=num_gt_instances_in_evaluated_tiles,
        prediction_source=prediction_source,
    )


def _save_tile_outputs(
    save_dir: Path,
    tile_id: str,
    pred_map: np.ndarray,
    gt_map: np.ndarray,
    *,
    bridge_cfg: dict[str, Any],
) -> None:
    if bool(bridge_cfg.get("save_pixel_predictions", True)):
        save_label_png(save_dir / "pixel_predictions" / f"{tile_id}_damage_prediction.png", pred_map)
    if bool(bridge_cfg.get("save_ground_truth", True)):
        save_label_png(save_dir / "pixel_ground_truth" / f"{tile_id}_damage_gt.png", gt_map)
    if bool(bridge_cfg.get("save_official_style", True)):
        save_label_png(save_dir / "official_style" / f"{tile_id}_prediction.png", pred_map)
    if bool(bridge_cfg.get("save_visuals", True)):
        save_colorized_label_png(save_dir / "visuals" / f"{tile_id}_pred_color.png", pred_map)
        save_colorized_label_png(save_dir / "visuals" / f"{tile_id}_gt_color.png", gt_map)
        save_overlay_png(save_dir / "visuals" / f"{tile_id}_overlay.png", pred_map, gt_map)


def save_tile_outputs(
    save_dir: Path,
    tile_id: str,
    pred_map: np.ndarray,
    gt_map: np.ndarray,
    *,
    bridge_cfg: dict[str, Any],
) -> None:
    _save_tile_outputs(save_dir, tile_id, pred_map, gt_map, bridge_cfg=bridge_cfg)


def resolve_image_shape(sample: dict[str, Any]) -> tuple[int, int]:
    return _resolve_image_shape(sample)


def resolve_official_target_png(tile_sample: dict[str, Any], config: dict[str, Any]) -> Path | None:
    return _resolve_official_target_png(tile_sample, config)


def run_bridge_evaluation(
    dataset: Any,
    prediction_records: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    save_dir: str | Path,
    prediction_source: str = "instance",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    save_dir = ensure_dir(save_dir)
    bridge_cfg = dict(config.get("bridge", {}))
    overlap_policy = str(bridge_cfg.get("overlap_policy", "max_label"))
    normalized_prediction_source = _normalize_prediction_source(prediction_source)
    tile_groups = _build_tile_groups(dataset.samples, prediction_records)

    per_tile_metrics: list[dict[str, Any]] = []
    global_confusion_matrix = np.zeros((5, 5), dtype=np.int64)
    num_gt_instances_in_evaluated_tiles = 0
    for tile_id, tile_group in sorted(tile_groups.items()):
        tile_sample = tile_group["tile_sample"]
        height, width = _resolve_image_shape(tile_sample)
        if normalized_prediction_source == "pixel_dense":
            pred_map = _render_dense_instance_predictions(
                tile_group["pred_instances"],
                height=height,
                width=width,
                overlap_policy=overlap_policy,
            )
        else:
            pred_instances = []
            for instance in tile_group["pred_instances"]:
                pred_instances.append(
                    {
                        **instance,
                        "bridge_pred_label": _resolve_prediction_label(instance, normalized_prediction_source),
                    }
                )
            pred_map = rasterize_instances_to_damage_map(
                pred_instances,
                height=height,
                width=width,
                label_field="bridge_pred_label",
                overlap_policy=overlap_policy,
            )
        gt_map, gt_source = _load_or_build_gt_map(
            tile_sample,
            tile_group["gt_instances"],
            height=height,
            width=width,
            config=config,
            overlap_policy=overlap_policy,
        )
        tile_metric = compute_pixel_bridge_metrics(pred_map=pred_map, gt_map=gt_map, num_classes=5)
        tile_metric["tile_id"] = tile_id
        tile_metric["gt_source"] = gt_source
        tile_metric["prediction_source"] = normalized_prediction_source
        tile_metric["num_pred_instances"] = int(len(tile_group["pred_instances"]))
        tile_metric["num_gt_instances"] = int(len(tile_group["gt_instances"]))
        per_tile_metrics.append(tile_metric)
        global_confusion_matrix += np.asarray(tile_metric["confusion_matrix_5x5"], dtype=np.int64)
        num_gt_instances_in_evaluated_tiles += int(len(tile_group["gt_instances"]))
        _save_tile_outputs(save_dir, tile_id, pred_map, gt_map, bridge_cfg=bridge_cfg)

    global_metrics = _compute_metrics_from_confusion_matrix(
        global_confusion_matrix,
        num_classes=5,
        eps=float(bridge_cfg.get("eps", 1.0e-8)),
    )
    per_tile_average = _average_metric_dicts(per_tile_metrics)
    summary = {
        "num_tiles": int(len(per_tile_metrics)),
        "num_instances": int(len(prediction_records)),
        "num_gt_instances_in_evaluated_tiles": int(num_gt_instances_in_evaluated_tiles),
        "global": global_metrics,
        "per_tile_average": per_tile_average,
        "prediction_source": normalized_prediction_source,
        "xview2_overall_score": float(global_metrics["xview2_overall_score"]),
        "localization_f1": float(global_metrics["localization_f1"]),
        "damage_f1_hmean": float(global_metrics["damage_f1_hmean"]),
    }

    write_json(save_dir / "global_metrics.json", global_metrics)
    write_json(save_dir / "per_tile_metrics.json", per_tile_metrics)
    write_json(
        save_dir / "confusion_matrix_5x5.json",
        {
            "class_names": [PIXEL_CLASS_NAMES[idx] for idx in range(5)],
            "matrix": global_metrics["confusion_matrix_5x5"],
        },
    )
    write_json(save_dir / "bridge_summary.json", summary)
    write_text(
        save_dir / "class_report_pixel.txt",
        _build_class_report_text(
            global_metrics=global_metrics,
            per_tile_average=per_tile_average,
            num_tiles=int(len(per_tile_metrics)),
            num_instances=int(len(prediction_records)),
            num_gt_instances_in_evaluated_tiles=int(num_gt_instances_in_evaluated_tiles),
            prediction_source=normalized_prediction_source,
        ),
    )
    return summary, per_tile_metrics
