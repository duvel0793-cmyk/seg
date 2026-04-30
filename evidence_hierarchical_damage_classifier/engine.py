from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.losses import build_loss
from utils.metrics import (
    build_classification_report,
    build_diagnostics,
    compute_classification_metrics,
    decode_predictions_with_damage_threshold,
    run_damage_threshold_sweep,
)
from utils.misc import resolve_amp_dtype


def build_epoch_loss(config: dict[str, Any]):
    return build_loss(config)


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def _maybe_channels_last(batch: dict[str, Any], enabled: bool) -> dict[str, Any]:
    if not enabled:
        return batch
    converted = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim == 4 and value.dtype.is_floating_point:
            converted[key] = value.contiguous(memory_format=torch.channels_last)
        else:
            converted[key] = value
    return converted


def _autocast_context(device: torch.device, dtype: torch.dtype | None):
    if device.type != "cuda" or dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _derive_binary_and_severity_probabilities(class_probabilities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    binary_probs = 1.0 - class_probabilities[:, 0]
    damaged_probs = class_probabilities[:, 1:]
    severity_denominator = damaged_probs.sum(dim=1, keepdim=True)
    safe_denominator = severity_denominator.clamp_min(1e-8)
    severity_probs = damaged_probs / safe_denominator
    severity_probs = torch.where(severity_denominator > 0, severity_probs, torch.zeros_like(severity_probs))
    return binary_probs, severity_probs


def run_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    ema: Any | None = None,
    epoch_index: int = 0,
    collect_predictions: bool = False,
    apply_best_damage_threshold: str = "none",
) -> dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    criterion.train(is_train)
    amp_dtype = resolve_amp_dtype(config, device)
    all_targets: list[int] = []
    all_predictions: list[int] = []
    all_probabilities: list[list[float]] = []
    all_damage_probabilities: list[float] = []
    all_severity_probabilities: list[list[float]] = []
    has_damage_binary_head = False
    prediction_records: list[dict[str, Any]] = []
    loss_sums: dict[str, float] = {}
    logging_sums: dict[str, float] = {}
    total_samples = 0
    total_loss = 0.0
    channels_last = bool(config["training"].get("channels_last", False))

    progress = tqdm(loader, desc=f"{'train' if is_train else 'eval'} {epoch_index + 1}", leave=False)
    for batch_idx, batch in enumerate(progress):
        batch = _move_batch_to_device(batch, device)
        batch = _maybe_channels_last(batch, channels_last)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, amp_dtype):
            outputs = model(batch)
            loss_terms = criterion(outputs, batch, epoch_index=epoch_index, batch_idx=batch_idx, is_train=is_train)
            loss = loss_terms["loss_total"]
        if is_train:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"].get("grad_clip_norm", 1.0)))
                scaler.step(optimizer)
                scaler.update()
                if ema is not None:
                    ema.update(model)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"].get("grad_clip_norm", 1.0)))
                optimizer.step()
                if ema is not None:
                    ema.update(model)
        batch_size = int(batch["label"].size(0))
        total_samples += batch_size
        total_loss += float(loss.detach().item()) * batch_size
        for key, value in loss_terms.items():
            loss_sums[key] = loss_sums.get(key, 0.0) + (float(value.detach().item()) * batch_size)
        for key, value in outputs.get("feature_stats", {}).items():
            if torch.is_tensor(value):
                logging_sums[key] = logging_sums.get(key, 0.0) + (float(value.detach().float().mean().item()) * batch_size)

        probabilities = outputs["class_probabilities"].detach().float().cpu()
        pred_tensor = outputs.get("pred_label", outputs["pred_labels"])
        predictions = pred_tensor.detach().cpu().tolist()
        targets = batch["label"].detach().cpu().tolist()
        damage_binary_logit = outputs["damage_binary_logit"]
        severity_prob_tensor = outputs["severity_class_probabilities"]
        if damage_binary_logit is not None:
            has_damage_binary_head = True
            binary_prob_tensor = torch.sigmoid(damage_binary_logit.detach().float().cpu())
        else:
            derived_binary, derived_severity = _derive_binary_and_severity_probabilities(probabilities)
            binary_prob_tensor = derived_binary
            if severity_prob_tensor is None:
                severity_prob_tensor = derived_severity
        if severity_prob_tensor is not None:
            severity_prob_tensor = severity_prob_tensor.detach().float().cpu()
        else:
            _, severity_prob_tensor = _derive_binary_and_severity_probabilities(probabilities)
        binary_probs = binary_prob_tensor.tolist()
        severity_probs = severity_prob_tensor.tolist()
        all_probabilities.extend(probabilities.tolist())
        all_predictions.extend(predictions)
        all_targets.extend(targets)
        all_damage_probabilities.extend([float(v) for v in binary_probs])
        all_severity_probabilities.extend([[float(item) for item in row] for row in severity_probs])
        if collect_predictions:
            evidence_stats = outputs.get("evidence_stats") or {}
            evidence_enabled = outputs.get("evidence_enabled", {})
            for row_idx in range(batch_size):
                meta = batch["meta"][row_idx]
                record_evidence_stats = None
                if evidence_stats:
                    record_evidence_stats = {
                        scale_name: (
                            evidence_stats[scale_name][row_idx].detach().cpu().tolist()
                            if (
                                scale_name in evidence_stats
                                and evidence_stats[scale_name] is not None
                                and bool(evidence_enabled.get(scale_name, True))
                            )
                            else None
                        )
                        for scale_name in evidence_stats.keys()
                    }
                prediction_records.append(
                    {
                        "sample_index": int(batch["sample_index"][row_idx].item()),
                        "image_id": meta["image_id"],
                        "polygon": meta["target_polygon"],
                        "bbox": meta["bbox"],
                        "gt_label": int(targets[row_idx]),
                        "pred_label": int(predictions[row_idx]),
                        "class_probabilities": probabilities[row_idx].tolist(),
                        "binary_damaged_probability": float(binary_probs[row_idx]),
                        "severity_probabilities": severity_probs[row_idx],
                        "evidence_stats": record_evidence_stats,
                    }
                )
        if (batch_idx + 1) % int(config["logging"].get("log_interval", 50)) == 0:
            progress.set_postfix(loss=f"{(total_loss / max(total_samples, 1)):.4f}")

    probabilities_np = np.asarray(all_probabilities, dtype=np.float64) if all_probabilities else np.zeros((0, 4), dtype=np.float64)
    metrics = compute_classification_metrics(all_targets, all_predictions, probabilities_np)
    threshold_sweep = None
    thresholds = [float(v) for v in config["eval"].get("damage_threshold_values", [])]
    if bool(config["eval"].get("damage_threshold_sweep", False)) and thresholds and all_severity_probabilities:
        threshold_sweep = run_damage_threshold_sweep(
            all_targets,
            np.asarray(all_damage_probabilities, dtype=np.float64),
            np.asarray(all_severity_probabilities, dtype=np.float64),
            probabilities_np,
            thresholds,
        )
        metrics["best_threshold_by_macro_f1"] = threshold_sweep["best_threshold_by_macro_f1"]
        metrics["best_threshold_by_qwk"] = threshold_sweep["best_threshold_by_qwk"]

    applied_damage_threshold = None
    threshold_metric = str(apply_best_damage_threshold or "none").lower()
    if threshold_metric != "none" and threshold_sweep is not None:
        best_entry = threshold_sweep["best_threshold_by_macro_f1"] if threshold_metric == "macro_f1" else threshold_sweep["best_threshold_by_qwk"]
        if best_entry is not None:
            applied_damage_threshold = float(best_entry["threshold"])
            threshold_predictions = decode_predictions_with_damage_threshold(
                np.asarray(all_damage_probabilities, dtype=np.float64),
                np.asarray(all_severity_probabilities, dtype=np.float64),
                applied_damage_threshold,
            ).tolist()
            all_predictions = threshold_predictions
            metrics = compute_classification_metrics(all_targets, all_predictions, probabilities_np)
            metrics["best_threshold_by_macro_f1"] = threshold_sweep["best_threshold_by_macro_f1"]
            metrics["best_threshold_by_qwk"] = threshold_sweep["best_threshold_by_qwk"]
            if collect_predictions:
                for row_idx, pred in enumerate(all_predictions):
                    prediction_records[row_idx]["pred_label"] = int(pred)

    feature_stats = {}
    if total_samples > 0 and hasattr(model, "backbone"):
        try:
            feature_stats["backbone_load_logs"] = list(getattr(model.backbone, "load_logs", []))
        except Exception:
            pass
    metrics["feature_stats"] = feature_stats
    diagnostics = build_diagnostics(prediction_records, metrics) if collect_predictions else None
    result = {
        "loss": total_loss / max(total_samples, 1),
        "loss_terms": {key: value / max(total_samples, 1) for key, value in loss_sums.items()},
        "logging_scalars": {key: value / max(total_samples, 1) for key, value in logging_sums.items()},
        "metrics": metrics,
        "prediction_records": prediction_records,
        "threshold_sweep": threshold_sweep,
        "applied_damage_threshold": applied_damage_threshold,
        "diagnostics": diagnostics,
        "report": build_classification_report(all_targets, all_predictions),
    }
    return result
