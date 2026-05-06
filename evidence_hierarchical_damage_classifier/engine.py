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


def _per_class_f1(metrics: dict[str, Any]) -> dict[str, float]:
    per_class = metrics.get("per_class", {})
    return {class_name: float(values.get("f1", 0.0)) for class_name, values in per_class.items()}


def _metrics_with_prefix(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_macro_f1": float(metrics.get("macro_f1", 0.0)),
        f"{prefix}_per_class_f1": _per_class_f1(metrics),
    }


def _ensure_optional_head_outputs(config: dict[str, Any], outputs: dict[str, Any]) -> None:
    model_cfg = config["model"]
    if bool(model_cfg.get("use_conditional_review_head", False)):
        required_review_keys = [
            "conditional_review_logits",
            "conditional_review_class_probabilities",
            "conditional_review_pred_labels",
            "conditional_review_low_logits",
            "conditional_review_high_logits",
            "conditional_review_alpha_low",
            "conditional_review_alpha_high",
        ]
        missing_keys = [key for key in required_review_keys if outputs.get(key) is None]
        if missing_keys:
            raise RuntimeError(
                "Conditional review is enabled, but forward did not return the expected outputs: "
                f"{', '.join(missing_keys)}. "
                "Refusing to continue because diagnostics/final prediction would be misleading."
            )
    if bool(model_cfg.get("use_scale_aux_fusion_head", False)):
        required_scale_aux_keys = [
            "scale_aux_tight_logits",
            "scale_aux_context_logits",
            "scale_aux_neighborhood_logits",
            "scale_aux_tight_class_probabilities",
            "scale_aux_context_class_probabilities",
            "scale_aux_neighborhood_class_probabilities",
            "scale_aux_fusion_logits",
            "scale_aux_fusion_weights",
            "scale_aux_fused_class_probabilities",
            "scale_aux_fused_pred_labels",
        ]
        missing_keys = [key for key in required_scale_aux_keys if outputs.get(key) is None]
        if missing_keys:
            raise RuntimeError(
                "Scale aux fusion head is enabled, but forward did not return the expected outputs: "
                f"{', '.join(missing_keys)}. "
                "Refusing to continue because diagnostics/final prediction would be misleading."
            )


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
    max_batches: int | None = None,
) -> dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    criterion.train(is_train)
    amp_dtype = resolve_amp_dtype(config, device)
    all_targets: list[int] = []
    all_predictions: list[int] = []
    all_probabilities: list[list[float]] = []
    all_corn_predictions: list[int] = []
    all_corn_probabilities: list[list[float]] = []
    all_structural_predictions: list[int] = []
    all_structural_probabilities: list[list[float]] = []
    all_conditional_review_predictions: list[int] = []
    all_conditional_review_probabilities: list[list[float]] = []
    all_router_predictions: list[int] = []
    all_router_probabilities: list[list[float]] = []
    all_scale_aux_tight_predictions: list[int] = []
    all_scale_aux_tight_probabilities: list[list[float]] = []
    all_scale_aux_context_predictions: list[int] = []
    all_scale_aux_context_probabilities: list[list[float]] = []
    all_scale_aux_neighborhood_predictions: list[int] = []
    all_scale_aux_neighborhood_probabilities: list[list[float]] = []
    all_scale_aux_fused_predictions: list[int] = []
    all_scale_aux_fused_probabilities: list[list[float]] = []
    all_damage_probabilities: list[float] = []
    all_severity_probabilities: list[list[float]] = []
    prediction_records: list[dict[str, Any]] = []
    loss_sums: dict[str, float] = {}
    logging_sums: dict[str, float] = {}
    total_samples = 0
    total_loss = 0.0
    channels_last = bool(config["training"].get("channels_last", False))
    last_scale_router_gates: list[float] | None = None

    progress = tqdm(loader, desc=f"{'train' if is_train else 'eval'} {epoch_index + 1}", leave=False)
    for batch_idx, batch in enumerate(progress):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        batch = _move_batch_to_device(batch, device)
        batch = _maybe_channels_last(batch, channels_last)
        batch["_epoch_index"] = int(epoch_index)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, amp_dtype):
            outputs = model(batch)
            _ensure_optional_head_outputs(config, outputs)
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
        if outputs.get("scale_router_gates") is not None:
            last_scale_router_gates = outputs["scale_router_gates"].detach().float().cpu().tolist()

        corn_probabilities = outputs["class_probabilities"].detach().float().cpu()
        final_probabilities = outputs.get("final_class_probabilities", outputs["class_probabilities"]).detach().float().cpu()
        corn_pred_tensor = outputs.get("pred_label", outputs["pred_labels"])
        final_pred_tensor = outputs.get("final_pred_label", outputs.get("final_pred_labels", corn_pred_tensor))
        predictions = final_pred_tensor.detach().cpu().tolist()
        corn_predictions = corn_pred_tensor.detach().cpu().tolist()
        targets = batch["label"].detach().cpu().tolist()
        damage_binary_logit = outputs["damage_binary_logit"]
        severity_prob_tensor = outputs["severity_class_probabilities"]
        if damage_binary_logit is not None:
            binary_prob_tensor = torch.sigmoid(damage_binary_logit.detach().float().cpu())
        else:
            derived_binary, derived_severity = _derive_binary_and_severity_probabilities(corn_probabilities)
            binary_prob_tensor = derived_binary
            if severity_prob_tensor is None:
                severity_prob_tensor = derived_severity
        if severity_prob_tensor is not None:
            severity_prob_tensor = severity_prob_tensor.detach().float().cpu()
        else:
            _, severity_prob_tensor = _derive_binary_and_severity_probabilities(corn_probabilities)
        structural_probs_tensor = outputs.get("structural_class_probabilities")
        structural_pred_tensor = outputs.get("structural_pred_labels")
        conditional_review_probs_tensor = outputs.get("conditional_review_class_probabilities")
        conditional_review_pred_tensor = outputs.get("conditional_review_pred_labels")
        router_probs_tensor = outputs.get("scale_router_probs")
        router_pred_tensor = outputs.get("scale_router_pred_labels")
        scale_aux_tight_probs_tensor = outputs.get("scale_aux_tight_class_probabilities")
        scale_aux_context_probs_tensor = outputs.get("scale_aux_context_class_probabilities")
        scale_aux_neighborhood_probs_tensor = outputs.get("scale_aux_neighborhood_class_probabilities")
        scale_aux_fused_probs_tensor = outputs.get("scale_aux_fused_class_probabilities")
        scale_aux_tight_pred_tensor = outputs.get("scale_aux_tight_pred_labels")
        scale_aux_context_pred_tensor = outputs.get("scale_aux_context_pred_labels")
        scale_aux_neighborhood_pred_tensor = outputs.get("scale_aux_neighborhood_pred_labels")
        scale_aux_fused_pred_tensor = outputs.get("scale_aux_fused_pred_labels")
        structural_probabilities = None if structural_probs_tensor is None else structural_probs_tensor.detach().float().cpu()
        conditional_review_probabilities = (
            None if conditional_review_probs_tensor is None else conditional_review_probs_tensor.detach().float().cpu()
        )
        router_probabilities = None if router_probs_tensor is None else router_probs_tensor.detach().float().cpu()
        scale_aux_tight_probabilities = (
            None if scale_aux_tight_probs_tensor is None else scale_aux_tight_probs_tensor.detach().float().cpu()
        )
        scale_aux_context_probabilities = (
            None if scale_aux_context_probs_tensor is None else scale_aux_context_probs_tensor.detach().float().cpu()
        )
        scale_aux_neighborhood_probabilities = (
            None if scale_aux_neighborhood_probs_tensor is None else scale_aux_neighborhood_probs_tensor.detach().float().cpu()
        )
        scale_aux_fused_probabilities = (
            None if scale_aux_fused_probs_tensor is None else scale_aux_fused_probs_tensor.detach().float().cpu()
        )
        structural_predictions = None if structural_pred_tensor is None else structural_pred_tensor.detach().cpu().tolist()
        conditional_review_predictions = (
            None if conditional_review_pred_tensor is None else conditional_review_pred_tensor.detach().cpu().tolist()
        )
        router_predictions = None if router_pred_tensor is None else router_pred_tensor.detach().cpu().tolist()
        scale_aux_tight_predictions = (
            None if scale_aux_tight_pred_tensor is None else scale_aux_tight_pred_tensor.detach().cpu().tolist()
        )
        scale_aux_context_predictions = (
            None if scale_aux_context_pred_tensor is None else scale_aux_context_pred_tensor.detach().cpu().tolist()
        )
        scale_aux_neighborhood_predictions = (
            None if scale_aux_neighborhood_pred_tensor is None else scale_aux_neighborhood_pred_tensor.detach().cpu().tolist()
        )
        scale_aux_fused_predictions = (
            None if scale_aux_fused_pred_tensor is None else scale_aux_fused_pred_tensor.detach().cpu().tolist()
        )
        binary_probs = binary_prob_tensor.tolist()
        severity_probs = severity_prob_tensor.tolist()
        all_probabilities.extend(final_probabilities.tolist())
        all_predictions.extend(predictions)
        all_corn_probabilities.extend(corn_probabilities.tolist())
        all_corn_predictions.extend(corn_predictions)
        all_targets.extend(targets)
        all_damage_probabilities.extend([float(v) for v in binary_probs])
        all_severity_probabilities.extend([[float(item) for item in row] for row in severity_probs])
        if structural_probabilities is not None:
            all_structural_probabilities.extend(structural_probabilities.tolist())
            all_structural_predictions.extend([] if structural_predictions is None else structural_predictions)
        if conditional_review_probabilities is not None:
            all_conditional_review_probabilities.extend(conditional_review_probabilities.tolist())
            all_conditional_review_predictions.extend([] if conditional_review_predictions is None else conditional_review_predictions)
        if router_probabilities is not None:
            all_router_probabilities.extend(router_probabilities.tolist())
            all_router_predictions.extend([] if router_predictions is None else router_predictions)
        if scale_aux_tight_probabilities is not None:
            all_scale_aux_tight_probabilities.extend(scale_aux_tight_probabilities.tolist())
            all_scale_aux_tight_predictions.extend([] if scale_aux_tight_predictions is None else scale_aux_tight_predictions)
        if scale_aux_context_probabilities is not None:
            all_scale_aux_context_probabilities.extend(scale_aux_context_probabilities.tolist())
            all_scale_aux_context_predictions.extend([] if scale_aux_context_predictions is None else scale_aux_context_predictions)
        if scale_aux_neighborhood_probabilities is not None:
            all_scale_aux_neighborhood_probabilities.extend(scale_aux_neighborhood_probabilities.tolist())
            all_scale_aux_neighborhood_predictions.extend(
                [] if scale_aux_neighborhood_predictions is None else scale_aux_neighborhood_predictions
            )
        if scale_aux_fused_probabilities is not None:
            all_scale_aux_fused_probabilities.extend(scale_aux_fused_probabilities.tolist())
            all_scale_aux_fused_predictions.extend([] if scale_aux_fused_predictions is None else scale_aux_fused_predictions)
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
                        "sample_id": int(batch["sample_index"][row_idx].item()),
                        "image_id": meta.get("image_id"),
                        "tile_id": meta.get("tile_id", meta.get("image_id")),
                        "building_idx": meta.get("building_idx"),
                        "building_id": meta.get("building_idx"),
                        "polygon": meta["target_polygon"],
                        "bbox": meta["bbox"],
                        "gt_label": int(targets[row_idx]),
                        "pred_label": int(predictions[row_idx]),
                        "corn_pred_label": int(corn_predictions[row_idx]),
                        "final_pred_label": int(predictions[row_idx]),
                        "structural_pred_label": None if structural_predictions is None else int(structural_predictions[row_idx]),
                        "conditional_review_pred_label": (
                            None if conditional_review_predictions is None else int(conditional_review_predictions[row_idx])
                        ),
                        "scale_router_pred_label": None if router_predictions is None else int(router_predictions[row_idx]),
                        "class_probabilities": final_probabilities[row_idx].tolist(),
                        "corn_class_probabilities": corn_probabilities[row_idx].tolist(),
                        "final_class_probabilities": final_probabilities[row_idx].tolist(),
                        "structural_class_probabilities": None if structural_probabilities is None else structural_probabilities[row_idx].tolist(),
                        "conditional_review_class_probabilities": (
                            None
                            if conditional_review_probabilities is None
                            else conditional_review_probabilities[row_idx].tolist()
                        ),
                        "scale_router_probabilities": None if router_probabilities is None else router_probabilities[row_idx].tolist(),
                        "binary_damaged_probability": float(binary_probs[row_idx]),
                        "severity_probabilities": severity_probs[row_idx],
                        "evidence_stats": record_evidence_stats,
                    }
                )
        if (batch_idx + 1) % int(config["logging"].get("log_interval", 50)) == 0:
            progress.set_postfix(loss=f"{(total_loss / max(total_samples, 1)):.4f}")

    probabilities_np = np.asarray(all_probabilities, dtype=np.float64) if all_probabilities else np.zeros((0, 4), dtype=np.float64)
    corn_probabilities_np = (
        np.asarray(all_corn_probabilities, dtype=np.float64) if all_corn_probabilities else np.zeros((0, 4), dtype=np.float64)
    )
    metrics = compute_classification_metrics(all_targets, all_predictions, probabilities_np)
    corn_metrics = compute_classification_metrics(all_targets, all_corn_predictions, corn_probabilities_np)
    structural_metrics = None
    if all_structural_predictions and len(all_structural_predictions) == len(all_targets):
        structural_metrics = compute_classification_metrics(
            all_targets,
            all_structural_predictions,
            np.asarray(all_structural_probabilities, dtype=np.float64),
        )
    conditional_review_metrics = None
    if all_conditional_review_predictions and len(all_conditional_review_predictions) == len(all_targets):
        conditional_review_metrics = compute_classification_metrics(
            all_targets,
            all_conditional_review_predictions,
            np.asarray(all_conditional_review_probabilities, dtype=np.float64),
        )
    router_metrics = None
    if all_router_predictions and len(all_router_predictions) == len(all_targets):
        router_metrics = compute_classification_metrics(
            all_targets,
            all_router_predictions,
            np.asarray(all_router_probabilities, dtype=np.float64),
        )
    scale_aux_tight_metrics = None
    if all_scale_aux_tight_predictions and len(all_scale_aux_tight_predictions) == len(all_targets):
        scale_aux_tight_metrics = compute_classification_metrics(
            all_targets,
            all_scale_aux_tight_predictions,
            np.asarray(all_scale_aux_tight_probabilities, dtype=np.float64),
        )
    scale_aux_context_metrics = None
    if all_scale_aux_context_predictions and len(all_scale_aux_context_predictions) == len(all_targets):
        scale_aux_context_metrics = compute_classification_metrics(
            all_targets,
            all_scale_aux_context_predictions,
            np.asarray(all_scale_aux_context_probabilities, dtype=np.float64),
        )
    scale_aux_neighborhood_metrics = None
    if all_scale_aux_neighborhood_predictions and len(all_scale_aux_neighborhood_predictions) == len(all_targets):
        scale_aux_neighborhood_metrics = compute_classification_metrics(
            all_targets,
            all_scale_aux_neighborhood_predictions,
            np.asarray(all_scale_aux_neighborhood_probabilities, dtype=np.float64),
        )
    scale_aux_fused_metrics = None
    if all_scale_aux_fused_predictions and len(all_scale_aux_fused_predictions) == len(all_targets):
        scale_aux_fused_metrics = compute_classification_metrics(
            all_targets,
            all_scale_aux_fused_predictions,
            np.asarray(all_scale_aux_fused_probabilities, dtype=np.float64),
        )
    threshold_sweep = None
    thresholds = [float(v) for v in config["eval"].get("damage_threshold_values", [])]
    if bool(config["eval"].get("damage_threshold_sweep", False)) and thresholds and all_severity_probabilities:
        threshold_sweep = run_damage_threshold_sweep(
            all_targets,
            np.asarray(all_damage_probabilities, dtype=np.float64),
            np.asarray(all_severity_probabilities, dtype=np.float64),
            corn_probabilities_np,
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
                    prediction_records[row_idx]["final_pred_label"] = int(pred)

    feature_stats = {}
    if total_samples > 0 and hasattr(model, "backbone"):
        try:
            feature_stats["backbone_load_logs"] = list(getattr(model.backbone, "load_logs", []))
        except Exception:
            pass
    structural_fusion_active = bool(config["model"].get("use_structural_two_stage_head", False)) and (
        bool(config["model"].get("structural_two_stage_use_for_prediction", False))
        or float(config["model"].get("structural_two_stage_prob_mix_weight", 0.0)) > 0.0
    )
    conditional_review_fusion_active = bool(config["model"].get("use_conditional_review_head", False)) and (
        bool(config["model"].get("conditional_review_use_for_prediction", False))
        and float(config["model"].get("conditional_review_prob_mix_weight", 0.0)) > 0.0
    )
    router_fusion_active = bool(config["model"].get("use_severity_aware_scale_router", False)) and (
        bool(config["model"].get("scale_router_use_for_prediction", False))
        or float(config["model"].get("scale_router_prob_mix_weight", 0.0)) > 0.0
    )
    scale_aux_fusion_active = bool(config["model"].get("use_scale_aux_fusion_head", False)) and bool(
        config["model"].get("scale_aux_use_for_prediction", False)
    )
    metrics["feature_stats"] = feature_stats
    metrics.update(_metrics_with_prefix(corn_metrics, "corn"))
    metrics.update(_metrics_with_prefix(metrics, "final"))
    metrics["final_prediction_active"] = bool(
        conditional_review_fusion_active or structural_fusion_active or router_fusion_active or scale_aux_fusion_active
    )
    metrics["scale_router_gates"] = last_scale_router_gates
    if structural_metrics is not None:
        metrics.update(_metrics_with_prefix(structural_metrics, "structural_head"))
    else:
        metrics["structural_head_macro_f1"] = None
        metrics["structural_head_per_class_f1"] = None
    if conditional_review_metrics is not None:
        metrics.update(_metrics_with_prefix(conditional_review_metrics, "conditional_review"))
    else:
        metrics["conditional_review_macro_f1"] = None
        metrics["conditional_review_per_class_f1"] = None
    if router_metrics is not None:
        metrics.update(_metrics_with_prefix(router_metrics, "scale_router"))
    else:
        metrics["scale_router_macro_f1"] = None
        metrics["scale_router_per_class_f1"] = None
    if scale_aux_tight_metrics is not None:
        metrics.update(_metrics_with_prefix(scale_aux_tight_metrics, "scale_aux_tight"))
    else:
        metrics["scale_aux_tight_macro_f1"] = None
        metrics["scale_aux_tight_per_class_f1"] = None
    if scale_aux_context_metrics is not None:
        metrics.update(_metrics_with_prefix(scale_aux_context_metrics, "scale_aux_context"))
    else:
        metrics["scale_aux_context_macro_f1"] = None
        metrics["scale_aux_context_per_class_f1"] = None
    if scale_aux_neighborhood_metrics is not None:
        metrics.update(_metrics_with_prefix(scale_aux_neighborhood_metrics, "scale_aux_neighborhood"))
    else:
        metrics["scale_aux_neighborhood_macro_f1"] = None
        metrics["scale_aux_neighborhood_per_class_f1"] = None
    if scale_aux_fused_metrics is not None:
        metrics.update(_metrics_with_prefix(scale_aux_fused_metrics, "scale_aux_fused"))
    else:
        metrics["scale_aux_fused_macro_f1"] = None
        metrics["scale_aux_fused_per_class_f1"] = None
    diagnostics_extra = {
        "corn_macro_f1": metrics.get("corn_macro_f1"),
        "corn_per_class_f1": metrics.get("corn_per_class_f1"),
        "structural_head_macro_f1": metrics.get("structural_head_macro_f1"),
        "structural_head_per_class_f1": metrics.get("structural_head_per_class_f1"),
        "conditional_review_macro_f1": metrics.get("conditional_review_macro_f1"),
        "conditional_review_per_class_f1": metrics.get("conditional_review_per_class_f1"),
        "scale_router_macro_f1": metrics.get("scale_router_macro_f1"),
        "scale_router_per_class_f1": metrics.get("scale_router_per_class_f1"),
        "scale_aux_tight_macro_f1": metrics.get("scale_aux_tight_macro_f1"),
        "scale_aux_tight_per_class_f1": metrics.get("scale_aux_tight_per_class_f1"),
        "scale_aux_context_macro_f1": metrics.get("scale_aux_context_macro_f1"),
        "scale_aux_context_per_class_f1": metrics.get("scale_aux_context_per_class_f1"),
        "scale_aux_neighborhood_macro_f1": metrics.get("scale_aux_neighborhood_macro_f1"),
        "scale_aux_neighborhood_per_class_f1": metrics.get("scale_aux_neighborhood_per_class_f1"),
        "scale_aux_fused_macro_f1": metrics.get("scale_aux_fused_macro_f1"),
        "scale_aux_fused_per_class_f1": metrics.get("scale_aux_fused_per_class_f1"),
        "final_macro_f1": metrics.get("final_macro_f1"),
        "final_per_class_f1": metrics.get("final_per_class_f1"),
        "scale_router_gates": metrics.get("scale_router_gates"),
        "scale_aux_weight_base_mean": logging_sums.get("scale_aux_weight_base_mean", 0.0) / max(total_samples, 1),
        "scale_aux_weight_tight_mean": logging_sums.get("scale_aux_weight_tight_mean", 0.0) / max(total_samples, 1),
        "scale_aux_weight_context_mean": logging_sums.get("scale_aux_weight_context_mean", 0.0) / max(total_samples, 1),
        "scale_aux_weight_neighborhood_mean": logging_sums.get("scale_aux_weight_neighborhood_mean", 0.0) / max(total_samples, 1),
        "scale_aux_weight_entropy_mean": logging_sums.get("scale_aux_weight_entropy", 0.0) / max(total_samples, 1),
        "scale_aux_weight_prior_kl_mean": loss_sums.get("loss_scale_aux_weight_prior_kl", 0.0) / max(total_samples, 1),
    }
    diagnostics = build_diagnostics(prediction_records, metrics, extra=diagnostics_extra) if collect_predictions else None
    result = {
        "loss": total_loss / max(total_samples, 1),
        "loss_terms": {key: value / max(total_samples, 1) for key, value in loss_sums.items()},
        "logging_scalars": {
            **{key: value / max(total_samples, 1) for key, value in logging_sums.items()},
            "scale_aux_weight_prior_kl_mean": loss_sums.get("loss_scale_aux_weight_prior_kl", 0.0) / max(total_samples, 1),
        },
        "metrics": metrics,
        "prediction_records": prediction_records,
        "threshold_sweep": threshold_sweep,
        "applied_damage_threshold": applied_damage_threshold,
        "diagnostics": diagnostics,
        "report": build_classification_report(all_targets, all_predictions),
    }
    return result
