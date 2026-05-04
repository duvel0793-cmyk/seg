from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
from PIL import Image
import torch

from datasets import XBDQueryDataset, xbd_query_collate_fn
from losses.box_losses import box_cxcywh_to_xyxy
from metrics.damage_instance_metrics import compute_end_to_end_damage_metrics, compute_matched_damage_metrics
from metrics.localization_metrics import compute_localization_metrics
from metrics.pixel_bridge_metrics import compute_pixel_bridge_metrics, metrics_from_confusion_matrix, rasterize_instances_to_damage_map
from utils.mask_nms import compute_patch_center_weight, suppress_duplicate_predictions

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    tqdm = None


def _autocast_context(device: torch.device, enabled: bool, dtype_name: str):
    if device.type != "cuda" or not enabled:
        return nullcontext()
    amp_dtype = torch.bfloat16 if dtype_name.lower() == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def _evaluation_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("evaluation", config.get("eval", {}))


def _progress(sequence, *, total: int | None, desc: str, enabled: bool):
    if not enabled or tqdm is None:
        return sequence
    return tqdm(sequence, total=total, desc=desc, dynamic_ncols=True)


def _attach_empty_damage_outputs(outputs: dict[str, Any]) -> dict[str, Any]:
    outputs["damage_logits"] = None
    outputs["damage_probabilities"] = None
    outputs["damage_pred_labels"] = None
    outputs["damage_binary_logits"] = None
    outputs["damage_severity_logits"] = None
    return outputs


def _need_damage_metrics(config: dict[str, Any], stage: str) -> bool:
    eval_cfg = _evaluation_config(config)
    if "need_damage_metrics" in eval_cfg:
        return bool(eval_cfg["need_damage_metrics"])
    return str(stage) != "localization"


def move_targets_to_device(targets: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    moved: list[dict[str, Any]] = []
    for target in targets:
        moved_target: dict[str, Any] = {}
        for key, value in target.items():
            moved_target[key] = value.to(device) if torch.is_tensor(value) else value
        moved.append(moved_target)
    return moved


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    moved["pre_image"] = batch["pre_image"].to(device, non_blocking=True)
    moved["post_image"] = batch["post_image"].to(device, non_blocking=True)
    moved["targets"] = move_targets_to_device(batch["targets"], device)
    if "gt_damage_map" in batch:
        moved["gt_damage_map"] = batch["gt_damage_map"].to(device, non_blocking=True)
    return moved


def _boxes_to_absolute_xyxy(boxes_cxcywh: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    height, width = int(image_size[0]), int(image_size[1])
    boxes_xyxy = box_cxcywh_to_xyxy(boxes_cxcywh)
    scale = torch.tensor([width, height, width, height], device=boxes_xyxy.device, dtype=boxes_xyxy.dtype)
    return boxes_xyxy * scale


def _target_instances(target: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    instances = []
    masks = target["masks"].detach().cpu().numpy()
    labels = target["labels"].detach().cpu().numpy()
    boxes = target["boxes"].detach().cpu().numpy()
    instance_ids = target["instance_ids"].detach().cpu().numpy() if "instance_ids" in target else np.arange(len(labels))
    for mask, label, box, instance_id in zip(masks, labels, boxes, instance_ids):
        instances.append(
            {
                "mask": mask.astype(bool),
                "label": int(label),
                "box_xyxy": [float(value) for value in box.tolist()],
                "instance_id": int(instance_id),
            }
        )
    return instances


def _prediction_instances(
    outputs: dict[str, Any],
    batch: dict[str, Any],
    *,
    objectness_threshold: float,
    mask_threshold: float,
    top_k: int,
    box_size_key: str = "scaled_size",
) -> list[list[dict[str, Any]]]:
    pred_logits = outputs["pred_logits"].detach()
    pred_masks_tensor = outputs.get("pred_masks")
    if pred_masks_tensor is None:
        pred_masks_tensor = torch.nn.functional.interpolate(
            outputs["pred_masks_lowres"],
            size=batch["pre_image"].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    pred_masks = pred_masks_tensor.detach().sigmoid()
    pred_boxes = outputs["pred_boxes"].detach()
    object_scores = pred_logits.softmax(dim=-1)[..., 1]
    if outputs.get("damage_probabilities") is not None:
        damage_probs = outputs["damage_probabilities"].detach()
        damage_labels = damage_probs.argmax(dim=-1)
    else:
        damage_probs = None
        damage_labels = torch.zeros_like(object_scores, dtype=torch.long)

    all_predictions: list[list[dict[str, Any]]] = []
    for batch_index in range(pred_logits.shape[0]):
        scores = object_scores[batch_index]
        keep = scores >= float(objectness_threshold)
        candidate_indices = torch.nonzero(keep, as_tuple=False).flatten()
        if candidate_indices.numel() == 0:
            top_indices = torch.topk(scores, k=min(top_k, scores.numel()), largest=True).indices
            candidate_indices = top_indices[scores[top_indices] > 0.0]
        if candidate_indices.numel() > top_k:
            top_order = torch.topk(scores[candidate_indices], k=top_k, largest=True).indices
            candidate_indices = candidate_indices[top_order]
        image_size = batch[box_size_key][batch_index]
        abs_boxes = _boxes_to_absolute_xyxy(pred_boxes[batch_index, candidate_indices], image_size)
        image_predictions: list[dict[str, Any]] = []
        for local_index, query_index in enumerate(candidate_indices.tolist()):
            mask = pred_masks[batch_index, query_index] >= float(mask_threshold)
            if int(mask.sum().item()) == 0:
                continue
            image_predictions.append(
                {
                    "query_index": int(query_index),
                    "score": float(scores[query_index].item()),
                    "mask": mask.cpu().numpy().astype(bool),
                    "label": int(damage_labels[batch_index, query_index].item()),
                    "box_xyxy": [float(value) for value in abs_boxes[local_index].cpu().tolist()],
                    "probabilities": None if damage_probs is None else damage_probs[batch_index, query_index].cpu().tolist(),
                }
            )
        all_predictions.append(image_predictions)
    return all_predictions


def _resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    if mask.shape == (height, width):
        return np.asarray(mask, dtype=bool)
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")
    resized = image.resize((int(width), int(height)), resample=Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8) > 127


def _patch_prediction_to_full_image(
    prediction: dict[str, Any],
    patch_box: tuple[int, int, int, int],
    original_patch_size: tuple[int, int],
    image_id: str,
    eval_cfg: dict[str, Any],
) -> dict[str, Any]:
    patch_x1, patch_y1, _, _ = [int(value) for value in patch_box]
    patch_height, patch_width = int(original_patch_size[0]), int(original_patch_size[1])
    local_mask = _resize_mask(prediction["mask"], patch_height, patch_width)
    local_box = [float(value) for value in prediction["box_xyxy"]]
    center_weight, touches_border, distance_ratio = compute_patch_center_weight(
        local_box_xyxy=local_box,
        patch_width=patch_width,
        patch_height=patch_height,
        use_patch_center_score=bool(eval_cfg.get("use_patch_center_score", True)),
        patch_border_penalty=float(eval_cfg.get("patch_border_penalty", 0.85)),
        border_margin=int(eval_cfg.get("border_margin", 32)),
    )
    adjusted_score = float(prediction["score"]) * center_weight
    global_box = [
        float(local_box[0] + patch_x1),
        float(local_box[1] + patch_y1),
        float(local_box[2] + patch_x1),
        float(local_box[3] + patch_y1),
    ]
    return {
        "image_id": image_id,
        "query_index": int(prediction["query_index"]),
        "score": float(prediction["score"]),
        "adjusted_score": float(adjusted_score),
        "center_prior_weight": float(center_weight),
        "touches_patch_border": bool(touches_border),
        "patch_center_distance_ratio": float(distance_ratio),
        "label": int(prediction["label"]),
        "box_xyxy": global_box,
        "local_box_xyxy": local_box,
        "patch_box": (int(patch_box[0]), int(patch_box[1]), int(patch_box[2]), int(patch_box[3])),
        "mask": local_mask,
        "mask_area": int(local_mask.sum()),
        "probabilities": prediction.get("probabilities"),
    }


def _finalize_full_image_predictions(
    predictions: list[dict[str, Any]],
    *,
    height: int,
    width: int,
) -> list[dict[str, Any]]:
    finalized: list[dict[str, Any]] = []
    for prediction in predictions:
        full_mask = np.zeros((int(height), int(width)), dtype=bool)
        patch_x1, patch_y1, _, _ = prediction["patch_box"]
        patch_mask = np.asarray(prediction["mask"], dtype=bool)
        valid_width = max(min(int(width) - int(patch_x1), patch_mask.shape[1]), 0)
        valid_height = max(min(int(height) - int(patch_y1), patch_mask.shape[0]), 0)
        if valid_width > 0 and valid_height > 0:
            full_mask[int(patch_y1) : int(patch_y1) + valid_height, int(patch_x1) : int(patch_x1) + valid_width] = patch_mask[:valid_height, :valid_width]
        finalized.append(
            {
                "mask": full_mask,
                "label": int(prediction["label"]),
                "score": float(prediction.get("adjusted_score", prediction["score"])),
                "raw_score": float(prediction["score"]),
                "adjusted_score": float(prediction.get("adjusted_score", prediction["score"])),
                "box_xyxy": [float(value) for value in prediction["box_xyxy"]],
                "query_index": int(prediction["query_index"]),
                "probabilities": prediction.get("probabilities"),
            }
        )
    return finalized


def _empty_merge_stats() -> dict[str, int]:
    return {
        "predictions_before_nms": 0,
        "predictions_after_nms": 0,
        "suppressed_pairs_count": 0,
        "suppressed_by_mask_iou": 0,
        "suppressed_by_box_iou": 0,
        "suppressed_by_containment": 0,
        "suppressed_by_center_distance": 0,
    }


def _empty_damage_summary() -> dict[str, Any]:
    return {
        "accuracy": 0.0,
        "macro_f1": 0.0,
        "per_class": {},
        "num_matched": 0,
    }


def _empty_end_to_end_summary() -> dict[str, Any]:
    return {
        "per_class": {},
        "macro_f1": 0.0,
        "weighted_f1": 0.0,
        "total_support": 0,
    }


def _empty_pixel_bridge() -> dict[str, Any]:
    return {
        "localization_f1": 0.0,
        "damage_f1_hmean": 0.0,
        "damage_macro_f1": 0.0,
        "xview2_overall_score": 0.0,
    }


def _accumulate_merge_stats(total: dict[str, int], update: dict[str, Any]) -> None:
    for key in total:
        total[key] += int(update.get(key, 0))


def _evaluate_prediction_groups(
    prediction_groups: list[dict[str, Any]],
    *,
    fallback_size: int,
    need_damage_metrics: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    localization = compute_localization_metrics(prediction_groups)
    if not need_damage_metrics:
        return localization, _empty_damage_summary(), _empty_end_to_end_summary(), _empty_pixel_bridge()
    matched_damage = compute_matched_damage_metrics(prediction_groups)
    end_to_end_damage = compute_end_to_end_damage_metrics(prediction_groups)

    pixel_scores = []
    global_confusion = np.zeros((5, 5), dtype=np.int64)
    for group in prediction_groups:
        gt_map = group["gt_damage_map"]
        if gt_map is None:
            gt_masks = [instance["mask"] for instance in group["gt_instances"]]
            if gt_masks:
                height, width = gt_masks[0].shape
            else:
                height = width = int(fallback_size)
            gt_map = rasterize_instances_to_damage_map(group["gt_instances"], height=height, width=width)
        pred_map = rasterize_instances_to_damage_map(
            group["pred_instances"],
            height=int(gt_map.shape[0]),
            width=int(gt_map.shape[1]),
        )
        score = compute_pixel_bridge_metrics(pred_map, gt_map)
        pixel_scores.append(score)
        global_confusion += np.asarray(score["confusion_matrix_5x5"], dtype=np.int64)

    if pixel_scores:
        pixel_bridge = metrics_from_confusion_matrix(global_confusion)
        pixel_bridge["per_image_average"] = {
            "localization_f1": float(np.mean([score["localization_f1"] for score in pixel_scores])),
            "damage_f1_hmean": float(np.mean([score["damage_f1_hmean"] for score in pixel_scores])),
            "damage_macro_f1": float(np.mean([score["damage_macro_f1"] for score in pixel_scores])),
            "xview2_overall_score": float(np.mean([score["xview2_overall_score"] for score in pixel_scores])),
        }
    else:
        pixel_bridge = {
            "localization_f1": 0.0,
            "damage_f1_hmean": 0.0,
            "damage_macro_f1": 0.0,
            "xview2_overall_score": 0.0,
        }
    return localization, matched_damage, end_to_end_damage, pixel_bridge


def _evaluate_full_image_mode(
    *,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    matcher: Any,
    criterion: torch.nn.Module | None,
    device: torch.device,
    config: dict[str, Any],
    stage: str,
    epoch: int,
    max_batches: int | None,
    max_images: int | None,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    prediction_groups: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []

    eval_cfg = _evaluation_config(config)
    need_damage_metrics = _need_damage_metrics(config, stage)
    progress_enabled = bool(config["training"].get("progress_bar", True))
    num_batches = len(loader) if hasattr(loader, "__len__") else None
    processed_images = 0
    with torch.no_grad():
        batch_iterator = _progress(
            enumerate(loader),
            total=num_batches,
            desc="Val",
            enabled=progress_enabled,
        )
        for batch_index, batch in batch_iterator:
            if max_batches is not None and batch_index >= int(max_batches):
                break
            if max_images is not None and processed_images >= int(max_images):
                break
            batch = move_batch_to_device(batch, device)
            with _autocast_context(device, bool(config["training"].get("amp", True)), str(config["training"].get("amp_dtype", "bf16"))):
                outputs = model.forward_localization(batch, stage=stage, return_full_masks=True)
                matches = matcher(outputs, batch["targets"])
                if str(stage) == "localization":
                    outputs = _attach_empty_damage_outputs(outputs)
                else:
                    outputs = model.forward_damage(batch, outputs, targets=batch["targets"], matches=matches, epoch=epoch, stage=stage)
                if criterion is not None:
                    loss_terms = criterion(outputs, batch["targets"], matches, stage=stage)
                    total_loss += float(loss_terms["loss_total"].detach().item())
            total_batches += 1
            if progress_enabled and tqdm is not None and hasattr(batch_iterator, "set_postfix"):
                batch_iterator.set_postfix(loss=f"{(total_loss / max(total_batches, 1)):.4f}")

            pred_instances_batch = _prediction_instances(
                outputs,
                batch,
                objectness_threshold=float(eval_cfg.get("objectness_threshold", 0.35)),
                mask_threshold=float(eval_cfg.get("mask_threshold", 0.5)),
                top_k=int(eval_cfg.get("top_k_predictions", 100)),
            )
            for item_index, pred_instances in enumerate(pred_instances_batch):
                if max_images is not None and processed_images >= int(max_images):
                    break
                target = batch["targets"][item_index]
                gt_instances = _target_instances(target)
                gt_damage_map = None
                if "gt_damage_map" in batch:
                    gt_damage_map = batch["gt_damage_map"][item_index].detach().cpu().numpy().astype(np.uint8)
                prediction_groups.append(
                    {
                        "image_id": target["image_id"],
                        "pred_instances": pred_instances,
                        "gt_instances": gt_instances,
                        "gt_damage_map": gt_damage_map,
                    }
                )
                for pred in pred_instances:
                    prediction_records.append(
                        {
                            "image_id": target["image_id"],
                            "query_index": pred["query_index"],
                            "score": pred["score"],
                            "adjusted_score": pred["score"],
                            "pred_label": pred["label"],
                            "pred_probabilities": pred.get("probabilities"),
                        }
                    )
                processed_images += 1

    localization, matched_damage, end_to_end_damage, pixel_bridge = _evaluate_prediction_groups(
        prediction_groups,
        fallback_size=int(loader.dataset.image_size),
        need_damage_metrics=need_damage_metrics,
    )
    mean_loss = total_loss / max(total_batches, 1)
    return {
        "loss": mean_loss,
        "localization": localization,
        "matched_damage": matched_damage,
        "end_to_end_damage": end_to_end_damage,
        "pixel_bridge": pixel_bridge,
        "prediction_groups": prediction_groups,
        "prediction_records": prediction_records,
        "merge_stats": _empty_merge_stats(),
        "merge_debug": [],
    }


def _suppress_predictions(
    full_predictions: list[dict[str, Any]],
    *,
    eval_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    if not bool(eval_cfg.get("merge_predictions", True)):
        stats = _empty_merge_stats()
        stats["predictions_before_nms"] = int(len(full_predictions))
        stats["predictions_after_nms"] = int(len(full_predictions))
        return full_predictions[: int(eval_cfg.get("max_predictions_per_image", 1000))], stats, []

    if bool(eval_cfg.get("apply_class_agnostic_nms", True)):
        return suppress_duplicate_predictions(
            full_predictions,
            mask_nms_iou=float(eval_cfg.get("mask_nms_iou", 0.45)),
            box_nms_iou=float(eval_cfg.get("box_nms_iou", 0.55)),
            containment_nms_thr=float(eval_cfg.get("containment_nms_thr", 0.75)),
            center_distance_ratio=float(eval_cfg.get("center_distance_ratio", 0.35)),
            max_predictions_per_image=int(eval_cfg.get("max_predictions_per_image", 1000)),
        )

    kept_total: list[dict[str, Any]] = []
    stats_total = _empty_merge_stats()
    suppressed_total: list[dict[str, Any]] = []
    labels = sorted({int(prediction["label"]) for prediction in full_predictions})
    for label in labels:
        label_predictions = [prediction for prediction in full_predictions if int(prediction["label"]) == label]
        kept, stats, suppressed = suppress_duplicate_predictions(
            label_predictions,
            mask_nms_iou=float(eval_cfg.get("mask_nms_iou", 0.45)),
            box_nms_iou=float(eval_cfg.get("box_nms_iou", 0.55)),
            containment_nms_thr=float(eval_cfg.get("containment_nms_thr", 0.75)),
            center_distance_ratio=float(eval_cfg.get("center_distance_ratio", 0.35)),
            max_predictions_per_image=int(eval_cfg.get("max_predictions_per_image", 1000)),
        )
        kept_total.extend(kept)
        suppressed_total.extend(suppressed)
        _accumulate_merge_stats(stats_total, stats)
    kept_total = sorted(kept_total, key=lambda item: float(item.get("adjusted_score", item["score"])), reverse=True)[: int(eval_cfg.get("max_predictions_per_image", 1000))]
    stats_total["predictions_after_nms"] = int(len(kept_total))
    return kept_total, stats_total, suppressed_total


def _evaluate_sliding_window_mode(
    *,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    matcher: Any,
    criterion: torch.nn.Module | None,
    device: torch.device,
    config: dict[str, Any],
    stage: str,
    epoch: int,
    max_batches: int | None,
    max_images: int | None,
) -> dict[str, Any]:
    model.eval()
    dataset = loader.dataset
    if not isinstance(dataset, XBDQueryDataset):
        raise TypeError("Sliding-window evaluation requires XBDQueryDataset.")

    eval_cfg = _evaluation_config(config)
    need_damage_metrics = _need_damage_metrics(config, stage)
    progress_enabled = bool(config["training"].get("progress_bar", True))
    total_loss = 0.0
    total_batches = 0
    prediction_groups: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    merge_stats_total = _empty_merge_stats()
    merge_debug: list[dict[str, Any]] = []

    patch_batch_size = int(eval_cfg.get("batch_size", 2))
    patch_size = int(eval_cfg.get("patch_size", dataset.val_patch_size))
    stride = int(eval_cfg.get("stride", dataset.val_stride))
    top_k_predictions = int(eval_cfg.get("top_k_predictions", 300))
    processed_images = 0

    with torch.no_grad():
        outer_total = len(loader) if hasattr(loader, "__len__") else None
        outer_iterator = _progress(
            enumerate(loader),
            total=outer_total,
            desc="ValImages",
            enabled=progress_enabled,
        )
        for batch_index, batch in outer_iterator:
            if max_batches is not None and batch_index >= int(max_batches):
                break
            for meta in batch["meta"]:
                if max_images is not None and processed_images >= int(max_images):
                    break
                sample_index = int(meta["sample_index"])
                ground_truth = dataset.build_full_image_ground_truth(sample_index)
                patch_boxes = dataset.get_sliding_window_boxes(sample_index, patch_size=patch_size, stride=stride)

                full_predictions: list[dict[str, Any]] = []
                patch_ranges = range(0, len(patch_boxes), max(patch_batch_size, 1))
                patch_iterator = _progress(
                    patch_ranges,
                    total=len(patch_ranges),
                    desc=f"Patches {ground_truth['image_id']}",
                    enabled=progress_enabled,
                )
                for start in patch_iterator:
                    chunk_boxes = patch_boxes[start : start + max(patch_batch_size, 1)]
                    patch_items = [dataset.build_inference_patch_item(sample_index, patch_box) for patch_box in chunk_boxes]
                    patch_batch = xbd_query_collate_fn(patch_items)
                    patch_batch = move_batch_to_device(patch_batch, device)
                    with _autocast_context(device, bool(config["training"].get("amp", True)), str(config["training"].get("amp_dtype", "bf16"))):
                        outputs = model.forward_localization(patch_batch, stage=stage, return_full_masks=True)
                        matches = matcher(outputs, patch_batch["targets"])
                        if str(stage) == "localization":
                            outputs = _attach_empty_damage_outputs(outputs)
                        else:
                            outputs = model.forward_damage(
                                patch_batch,
                                outputs,
                                targets=patch_batch["targets"],
                                matches=matches,
                                epoch=epoch,
                                stage=stage,
                            )
                        if criterion is not None:
                            loss_terms = criterion(outputs, patch_batch["targets"], matches, stage=stage)
                            total_loss += float(loss_terms["loss_total"].detach().item())
                    total_batches += 1
                    if progress_enabled and tqdm is not None and hasattr(patch_iterator, "set_postfix"):
                        patch_iterator.set_postfix(
                            loss=f"{(total_loss / max(total_batches, 1)):.4f}",
                            preds=len(full_predictions),
                        )

                    patch_predictions = _prediction_instances(
                        outputs,
                        patch_batch,
                        objectness_threshold=float(eval_cfg.get("objectness_threshold", 0.35)),
                        mask_threshold=float(eval_cfg.get("mask_threshold", 0.5)),
                        top_k=top_k_predictions,
                        box_size_key="original_size",
                    )
                    for local_patch_index, predictions in enumerate(patch_predictions):
                        patch_meta = patch_batch["meta"][local_patch_index]
                        patch_box = tuple(int(value) for value in patch_meta["patch_box"])
                        original_patch_size = tuple(int(value) for value in patch_batch["original_size"][local_patch_index])
                        for prediction in predictions:
                            mapped = _patch_prediction_to_full_image(
                                prediction,
                                patch_box=patch_box,
                                original_patch_size=original_patch_size,
                                image_id=str(ground_truth["image_id"]),
                                eval_cfg=eval_cfg,
                            )
                            full_predictions.append(mapped)
                            prediction_records.append(
                                {
                                    "image_id": str(ground_truth["image_id"]),
                                    "patch_box": list(patch_box),
                                    "query_index": mapped["query_index"],
                                    "score": mapped["score"],
                                    "adjusted_score": mapped["adjusted_score"],
                                    "center_prior_weight": mapped["center_prior_weight"],
                                    "touches_patch_border": mapped["touches_patch_border"],
                                    "pred_label": mapped["label"],
                                    "pred_probabilities": mapped.get("probabilities"),
                                }
                            )

                kept_predictions, merge_stats, suppressed_pairs = _suppress_predictions(full_predictions, eval_cfg=eval_cfg)
                _accumulate_merge_stats(merge_stats_total, merge_stats)
                merge_debug.append(
                    {
                        "image_id": str(ground_truth["image_id"]),
                        **merge_stats,
                        "gt_instances": int(len(ground_truth["gt_instances"])),
                        "suppressed_pairs": [
                            {
                                "reason": item["reason"],
                                "mask_iou": float(item["metrics"]["mask_iou"]),
                                "box_iou": float(item["metrics"]["box_iou"]),
                                "containment": float(item["metrics"]["containment"]),
                                "center_distance": float(item["metrics"]["center_distance"]),
                                "center_distance_ratio": float(item["metrics"]["center_distance_ratio"]),
                                "kept_score": float(item["kept_prediction"]["adjusted_score"]),
                                "suppressed_score": float(item["suppressed_prediction"]["adjusted_score"]),
                            }
                            for item in suppressed_pairs[:50]
                        ],
                    }
                )

                finalized_predictions = _finalize_full_image_predictions(
                    kept_predictions,
                    height=int(ground_truth["height"]),
                    width=int(ground_truth["width"]),
                )
                prediction_groups.append(
                    {
                        "image_id": str(ground_truth["image_id"]),
                        "pred_instances": finalized_predictions,
                        "gt_instances": ground_truth["gt_instances"],
                        "gt_damage_map": ground_truth["gt_damage_map"],
                    }
                )
                processed_images += 1

    localization, matched_damage, end_to_end_damage, pixel_bridge = _evaluate_prediction_groups(
        prediction_groups,
        fallback_size=int(dataset.patch_size),
        need_damage_metrics=need_damage_metrics,
    )
    mean_loss = total_loss / max(total_batches, 1)
    return {
        "loss": mean_loss,
        "localization": localization,
        "matched_damage": matched_damage,
        "end_to_end_damage": end_to_end_damage,
        "pixel_bridge": pixel_bridge,
        "prediction_groups": prediction_groups,
        "prediction_records": prediction_records,
        "merge_stats": merge_stats_total,
        "merge_debug": merge_debug,
    }


def evaluate_model(
    *,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    matcher: Any,
    criterion: torch.nn.Module | None,
    device: torch.device,
    config: dict[str, Any],
    stage: str,
    epoch: int = 0,
    max_batches: int | None = None,
    max_images: int | None = None,
) -> dict[str, Any]:
    eval_cfg = _evaluation_config(config)
    val_mode = str(eval_cfg.get("val_mode", "sliding_window"))
    if val_mode == "sliding_window":
        return _evaluate_sliding_window_mode(
            model=model,
            loader=loader,
            matcher=matcher,
            criterion=criterion,
            device=device,
            config=config,
            stage=stage,
            epoch=epoch,
            max_batches=max_batches,
            max_images=max_images,
        )
    return _evaluate_full_image_mode(
        model=model,
        loader=loader,
        matcher=matcher,
        criterion=criterion,
        device=device,
        config=config,
        stage=stage,
        epoch=epoch,
        max_batches=max_batches,
        max_images=max_images,
    )
