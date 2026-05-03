from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
import torch

from losses.box_losses import box_cxcywh_to_xyxy
from metrics.damage_instance_metrics import compute_end_to_end_damage_metrics, compute_matched_damage_metrics
from metrics.localization_metrics import compute_localization_metrics
from metrics.pixel_bridge_metrics import compute_pixel_bridge_metrics, metrics_from_confusion_matrix, rasterize_instances_to_damage_map


def _autocast_context(device: torch.device, enabled: bool, dtype_name: str):
    if device.type != "cuda" or not enabled:
        return nullcontext()
    amp_dtype = torch.bfloat16 if dtype_name.lower() == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


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
) -> list[list[dict[str, Any]]]:
    pred_logits = outputs["pred_logits"].detach()
    pred_masks = outputs["pred_masks"].detach().sigmoid()
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
        image_size = batch["scaled_size"][batch_index]
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
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    prediction_groups: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []

    eval_cfg = config.get("eval", {})
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = move_batch_to_device(batch, device)
            with _autocast_context(device, bool(config["training"].get("amp", True)), str(config["training"].get("amp_dtype", "bf16"))):
                outputs = model.forward_localization(batch)
                matches = matcher(outputs, batch["targets"])
                outputs = model.forward_damage(batch, outputs, targets=batch["targets"], matches=matches, epoch=epoch, stage=stage)
                if criterion is not None:
                    loss_terms = criterion(outputs, batch["targets"], matches, stage=stage)
                    total_loss += float(loss_terms["loss_total"].detach().item())
            total_batches += 1

            pred_instances_batch = _prediction_instances(
                outputs,
                batch,
                objectness_threshold=float(eval_cfg.get("objectness_threshold", 0.35)),
                mask_threshold=float(eval_cfg.get("mask_threshold", 0.5)),
                top_k=int(eval_cfg.get("top_k_predictions", 100)),
            )
            for item_index, pred_instances in enumerate(pred_instances_batch):
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
                            "pred_label": pred["label"],
                            "pred_probabilities": pred.get("probabilities"),
                        }
                    )

    localization = compute_localization_metrics(prediction_groups)
    matched_damage = compute_matched_damage_metrics(prediction_groups)
    end_to_end_damage = compute_end_to_end_damage_metrics(prediction_groups)

    pixel_scores = []
    global_confusion = np.zeros((5, 5), dtype=np.int64)
    for group in prediction_groups:
        pred_map = rasterize_instances_to_damage_map(
            group["pred_instances"],
            height=int(loader.dataset.image_size),
            width=int(loader.dataset.image_size),
        )
        if group.get("gt_damage_map") is not None:
            gt_map = group["gt_damage_map"]
        else:
            gt_map = rasterize_instances_to_damage_map(group["gt_instances"], height=pred_map.shape[0], width=pred_map.shape[1])
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

    mean_loss = total_loss / max(total_batches, 1)
    return {
        "loss": mean_loss,
        "localization": localization,
        "matched_damage": matched_damage,
        "end_to_end_damage": end_to_end_damage,
        "pixel_bridge": pixel_bridge,
        "prediction_groups": prediction_groups,
        "prediction_records": prediction_records,
    }
