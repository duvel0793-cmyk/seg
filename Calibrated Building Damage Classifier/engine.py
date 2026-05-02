from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.xbd_oracle_instance_damage import CLASS_NAMES
from utils.io import append_jsonl, ensure_dir
from utils.losses import (
    assert_finite_tensor,
    build_loss,
    compute_expected_severity_from_probabilities,
)
from utils.metrics import (
    compute_classification_metrics,
    compute_expected_severity_bias,
    compute_expected_severity_error,
    compute_minor_major_bidirectional_analysis,
    summarize_mean_predicted_distribution_by_true_class,
)

SCALE_NAMES = ("tight", "context", "neighborhood")


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_amp_dtype(config: dict[str, Any], device: torch.device) -> tuple[torch.dtype | None, str, str | None]:
    requested = str(config["training"]["amp_dtype"]).lower()
    if device.type != "cuda":
        return None, "disabled", None
    bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    if requested == "auto":
        return (torch.bfloat16, "bf16", None) if bf16_supported else (torch.float16, "fp16", None)
    if requested == "bf16":
        if bf16_supported:
            return torch.bfloat16, "bf16", None
        return torch.float16, "fp16", "bf16_requested_but_not_supported_fallback_to_fp16"
    if requested == "fp16":
        return torch.float16, "fp16", None
    raise ValueError(f"Unsupported AMP dtype request '{requested}'.")


def resolve_amp_settings(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    amp_enabled = bool(config["training"]["amp_enabled"]) and device.type == "cuda"
    amp_dtype, amp_dtype_name, fallback_reason = resolve_amp_dtype(config, device)
    autocast_enabled = bool(amp_enabled and amp_dtype is not None)
    scaler_enabled = bool(autocast_enabled and amp_dtype == torch.float16)
    return {
        "enabled": autocast_enabled,
        "dtype": amp_dtype,
        "dtype_name": amp_dtype_name,
        "scaler_enabled": scaler_enabled,
        "fallback_reason": fallback_reason,
    }


def resolve_autocast_context(device: torch.device, enabled: bool, amp_dtype: torch.dtype | None):
    if not enabled or amp_dtype is None:
        return nullcontext()
    if device.type not in {"cuda", "cpu"}:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def resolve_numerical_issue_log_path(config: dict[str, Any]) -> Path:
    return ensure_dir(Path(config["logging"]["output_root"]) / config["logging"]["exp_name"] / "logs") / "numerical_issues.jsonl"


def build_epoch_loss(config: dict[str, Any]):
    return build_loss(config)


def compute_collapse_diagnostics(metrics: dict[str, Any], *, epoch_index: int) -> dict[str, Any]:
    prediction_distribution = metrics.get("prediction_distribution", {})
    total_predictions = float(sum(float(value) for value in prediction_distribution.values()))
    dominant_class = None
    dominant_ratio = 0.0
    if total_predictions > 0.0 and prediction_distribution:
        dominant_class, dominant_count = max(prediction_distribution.items(), key=lambda item: float(item[1]))
        dominant_ratio = float(dominant_count) / total_predictions
    feature_means = []
    for feature_name in ("tight_feature_norm", "context_feature_norm", "neighborhood_feature_norm"):
        summary = metrics.get("feature_stats", {}).get(feature_name)
        if summary is not None and summary.get("mean") is not None:
            feature_means.append(float(summary["mean"]))
    final_feature_summary = metrics.get("feature_stats", {}).get("final_instance_feature_norm")
    final_feature_std = None if final_feature_summary is None else final_feature_summary.get("std")
    severity_values = [
        metrics.get("severity_score_mean_cls0"),
        metrics.get("severity_score_mean_cls1"),
        metrics.get("severity_score_mean_cls2"),
        metrics.get("severity_score_mean_cls3"),
    ]
    valid_severity_values = [float(value) for value in severity_values if value is not None]
    severity_collapsed = False
    if len(valid_severity_values) == len(severity_values):
        severity_collapsed = (max(valid_severity_values) - min(valid_severity_values)) < 0.05

    return {
        "collapse_pred_dominant_class": dominant_class if dominant_ratio > 0.98 and float(metrics.get("macro_f1", 0.0)) < 0.35 else None,
        "collapse_pred_dominant_ratio": dominant_ratio,
        "collapse_feature_norm_warning": bool(any(value > 3000.0 for value in feature_means)),
        "collapse_instance_feature_std_warning": bool(final_feature_std is not None and float(final_feature_std) < 0.005),
        "collapse_aux_auc_warning": bool(epoch_index + 1 > 2 and (metrics.get("damage_aux_auc_tight") is not None) and float(metrics["damage_aux_auc_tight"]) < 0.6),
        "collapse_severity_warning": bool(severity_collapsed),
    }


def summarize_values(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    tensor = torch.tensor(values, dtype=torch.float32)
    quantiles = torch.quantile(tensor, torch.tensor([0.10, 0.50, 0.90], dtype=tensor.dtype))
    return {
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()) if tensor.numel() > 1 else 0.0,
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "p10": float(quantiles[0].item()),
        "p50": float(quantiles[1].item()),
        "p90": float(quantiles[2].item()),
    }


def tensor_item_or_none(tensor: torch.Tensor | None, index: int) -> float | None:
    if tensor is None:
        return None
    detached = tensor.detach().cpu()
    if detached.ndim == 0:
        return float(detached.item())
    return float(detached[index].item())


def gradients_are_finite(parameters: list[torch.nn.Parameter]) -> bool:
    for parameter in parameters:
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            return False
    return True


def summarize_tensor_for_debug(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach().float()
    finite_mask = torch.isfinite(detached)
    summary: dict[str, Any] = {
        "shape": list(detached.shape),
        "numel": int(detached.numel()),
        "num_finite": int(finite_mask.sum().item()),
        "num_nonfinite": int((~finite_mask).sum().item()),
    }
    if finite_mask.any():
        finite_values = detached[finite_mask]
        summary.update(
            {
                "mean": float(finite_values.mean().item()),
                "std": float(finite_values.std(unbiased=False).item()) if finite_values.numel() > 1 else 0.0,
                "min": float(finite_values.min().item()),
                "max": float(finite_values.max().item()),
            }
        )
    else:
        summary.update({"mean": None, "std": None, "min": None, "max": None})
    return summary


def append_numerical_issue_record(config: dict[str, Any], record: dict[str, Any]) -> None:
    append_jsonl(resolve_numerical_issue_log_path(config), record)


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def run_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    epoch_index: int = 0,
    collect_predictions: bool = False,
    batch_transform: Callable[[int, dict[str, Any]], dict[str, Any]] | None = None,
    post_step_callback: Callable[[torch.nn.Module], None] | None = None,
) -> dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    criterion.train(is_train)
    logging_cfg = config.get("logging", {})
    collect_feature_stats = bool(logging_cfg.get("log_feature_stats", True) or not is_train or collect_predictions)
    collect_gate_stats = bool(logging_cfg.get("log_gate_stats", True) or not is_train or collect_predictions)
    collect_severity_aux_stats = bool(logging_cfg.get("log_severity_aux_stats", True) or not is_train or collect_predictions)
    collect_scale_metrics = bool(logging_cfg.get("log_scale_metrics", True) or not is_train or collect_predictions)
    collect_cross_scale_stats = bool(logging_cfg.get("log_cross_scale_attention", True) or not is_train or collect_predictions)
    collect_graph_stats = bool(logging_cfg.get("log_graph_metrics", True) or not is_train or collect_predictions)
    collect_context_diagnostic = bool(logging_cfg.get("log_context_diagnostic", True) or not is_train or collect_predictions)
    collect_diagnostics = bool(
        collect_gate_stats
        or collect_cross_scale_stats
        or collect_graph_stats
        or collect_context_diagnostic
    )
    if hasattr(model, "set_runtime_context"):
        model.set_runtime_context(
            epoch_index=int(epoch_index),
            is_train=is_train,
            collect_diagnostics=collect_diagnostics,
            collect_feature_stats=collect_feature_stats,
        )

    amp_settings = resolve_amp_settings(config, device)
    amp_enabled = bool(amp_settings["enabled"])
    finite_check_enabled = bool(config["training"]["finite_check_enabled"])
    prediction_positions_cpu = None
    if collect_predictions:
        prediction_positions_cpu = torch.linspace(0.0, 1.0, steps=len(CLASS_NAMES), dtype=torch.float32)

    total_loss = 0.0
    total_samples = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []
    collect_probability_analytics = bool((not is_train) or collect_predictions or collect_context_diagnostic or collect_severity_aux_stats)
    all_probabilities: list[list[float]] = []
    prediction_records: list[dict[str, Any]] = []
    feature_stat_values: dict[str, list[float]] = {}
    loss_term_sums: dict[str, float] = {}
    damage_aux_scores_by_scale = {scale_name: [] for scale_name in SCALE_NAMES} if collect_scale_metrics else {}
    damage_aux_targets_by_scale = {scale_name: [] for scale_name in SCALE_NAMES} if collect_scale_metrics else {}
    severity_aux_scores: list[float] = [] if collect_severity_aux_stats else []
    severity_aux_targets: list[float] = [] if collect_severity_aux_stats else []
    severity_aux_labels: list[int] = [] if collect_severity_aux_stats else []
    diagnostic_values: dict[str, list[float]] = {
        "tight_gate_inside": [],
        "tight_gate_outside": [],
        "tight_gate_gap": [],
        "tight_gate_mean": [],
        "tight_gate_std": [],
        "context_gate_inside": [],
        "context_gate_outside": [],
        "context_gate_gap": [],
        "context_gate_mean": [],
        "context_gate_std": [],
        "neighborhood_gate_inside": [],
        "neighborhood_gate_outside": [],
        "neighborhood_gate_gap": [],
        "neighborhood_gate_mean": [],
        "neighborhood_gate_std": [],
        "cross_attn_to_context_mean": [],
        "cross_attn_to_neighborhood_mean": [],
        "cross_scale_attention_entropy": [],
        "context_dropout_rate_actual": [],
        "neighborhood_dropout_rate_actual": [],
        "graph_gate_mean": [],
        "graph_attention_entropy": [],
        "graph_valid_neighbor_count": [],
        "neighborhood_branch_gate": [],
        "neighborhood_residual_scale": [],
    } if collect_diagnostics else {}
    skipped_nonfinite_grad_batches = 0
    split_name = "train" if is_train else "eval"

    progress = tqdm(loader, leave=False, desc=split_name)
    inference_context = nullcontext() if is_train else torch.inference_mode()
    with inference_context:
        for batch_index, batch in enumerate(progress):
            if batch_transform is not None:
                batch = batch_transform(batch_index, batch)
            device_batch = _move_batch_to_device(batch, device)
            labels = device_batch["label"]

            if is_train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)

            if finite_check_enabled:
                for scale_name in SCALE_NAMES:
                    assert_finite_tensor(f"pre_{scale_name}", device_batch[f"pre_{scale_name}"].float(), context={"split": split_name, "batch": batch_index})
                    assert_finite_tensor(f"post_{scale_name}", device_batch[f"post_{scale_name}"].float(), context={"split": split_name, "batch": batch_index})
                    assert_finite_tensor(f"mask_{scale_name}", device_batch[f"mask_{scale_name}"].float(), context={"split": split_name, "batch": batch_index})

            with resolve_autocast_context(device, amp_enabled, amp_settings["dtype"]):
                outputs = model(device_batch)
                loss_outputs = criterion(
                    outputs=outputs,
                    targets=labels,
                    batch=device_batch,
                    debug_context={"split": split_name, "epoch": int(epoch_index + 1), "batch": int(batch_index)},
                )
                loss_total = loss_outputs["loss"]

            if finite_check_enabled:
                assert_finite_tensor("corn_logits", outputs["corn_logits"].float(), context={"split": split_name, "batch": batch_index})
                assert_finite_tensor("loss_total", loss_total.float(), context={"split": split_name, "batch": batch_index})

            if is_train:
                assert optimizer is not None
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss_total).backward()
                    scaler.unscale_(optimizer)
                    if not gradients_are_finite([param for group in optimizer.param_groups for param in group["params"]]):
                        skipped_nonfinite_grad_batches += 1
                        append_numerical_issue_record(
                            config,
                            {
                                "event": "nonfinite_gradient",
                                "split": split_name,
                                "epoch": int(epoch_index + 1),
                                "batch_index": int(batch_index),
                                "batch_debug": {f"pre_{scale_name}": summarize_tensor_for_debug(device_batch[f"pre_{scale_name}"]) for scale_name in SCALE_NAMES},
                            },
                        )
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                        continue
                    if float(config["training"]["max_grad_norm"]) > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["max_grad_norm"]))
                    scaler.step(optimizer)
                    scaler.update()
                    if post_step_callback is not None:
                        post_step_callback(model)
                else:
                    loss_total.backward()
                    if not gradients_are_finite([param for group in optimizer.param_groups for param in group["params"]]):
                        skipped_nonfinite_grad_batches += 1
                        append_numerical_issue_record(
                            config,
                            {
                                "event": "nonfinite_gradient",
                                "split": split_name,
                                "epoch": int(epoch_index + 1),
                                "batch_index": int(batch_index),
                            },
                        )
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    if float(config["training"]["max_grad_norm"]) > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["max_grad_norm"]))
                    optimizer.step()
                    if post_step_callback is not None:
                        post_step_callback(model)

            batch_size = labels.size(0)
            total_loss += float(loss_total.detach().item()) * batch_size
            total_samples += batch_size

            class_probabilities = loss_outputs["class_probabilities"]
            predictions = class_probabilities.detach().argmax(dim=1).cpu().tolist()
            targets_cpu = labels.detach().cpu().tolist()
            class_probabilities_cpu = class_probabilities.detach().cpu()

            all_targets.extend(targets_cpu)
            all_predictions.extend(predictions)
            if collect_probability_analytics:
                all_probabilities.extend([row.tolist() for row in class_probabilities_cpu])

            if collect_feature_stats:
                for stat_name, stat_tensor in outputs.get("feature_stats", {}).items():
                    feature_stat_values.setdefault(stat_name, [])
                    feature_stat_values[stat_name].extend(float(value) for value in stat_tensor.detach().float().cpu().tolist())

            for loss_name, loss_value in loss_outputs["loss_terms"].items():
                loss_term_sums[loss_name] = loss_term_sums.get(loss_name, 0.0) + (float(loss_value.detach().item()) * batch_size)

            aux_metrics = loss_outputs.get("aux_metrics", {})
            if collect_scale_metrics:
                damage_targets = aux_metrics.get("damage_aux_targets")
                for scale_name in SCALE_NAMES:
                    scale_scores = aux_metrics.get("damage_aux_scores", {}).get(scale_name)
                    if scale_scores is not None and damage_targets is not None:
                        damage_aux_scores_by_scale[scale_name].extend(scale_scores.detach().cpu().tolist())
                        damage_aux_targets_by_scale[scale_name].extend(damage_targets.detach().cpu().tolist())
            if collect_severity_aux_stats:
                batch_severity_scores = aux_metrics.get("severity_scores")
                batch_severity_targets = aux_metrics.get("severity_targets")
                batch_severity_labels = aux_metrics.get("severity_labels")
                if batch_severity_scores is not None:
                    severity_aux_scores.extend(batch_severity_scores.detach().cpu().tolist())
                    severity_aux_targets.extend(batch_severity_targets.detach().cpu().tolist())
                    if batch_severity_labels is not None:
                        severity_aux_labels.extend(int(value) for value in batch_severity_labels.detach().cpu().tolist())
            diagnostics = outputs.get("diagnostics", {})
            if collect_diagnostics:
                for scale_name in SCALE_NAMES:
                    inside_key = f"{scale_name}_gate_inside"
                    outside_key = f"{scale_name}_gate_outside"
                    gap_key = f"{scale_name}_gate_gap"
                    mean_key = f"{scale_name}_gate_mean"
                    std_key = f"{scale_name}_gate_std"
                    inside_valid = diagnostics.get(f"{scale_name}_gate_inside_valid")
                    outside_valid = diagnostics.get(f"{scale_name}_gate_outside_valid")
                    inside_tensor = diagnostics.get(inside_key)
                    outside_tensor = diagnostics.get(outside_key)
                    gap_tensor = diagnostics.get(gap_key)
                    mean_tensor = diagnostics.get(mean_key)
                    std_tensor = diagnostics.get(std_key)
                    if inside_tensor is not None:
                        inside_values = inside_tensor.detach().float().cpu()
                        if inside_valid is not None:
                            mask_cpu = inside_valid.detach().bool().cpu()
                            inside_values = inside_values[mask_cpu]
                        diagnostic_values[inside_key].extend(float(value) for value in inside_values.tolist())
                    if outside_tensor is not None:
                        outside_values = outside_tensor.detach().float().cpu()
                        gap_values = gap_tensor.detach().float().cpu() if gap_tensor is not None else None
                        if outside_valid is not None:
                            mask_cpu = outside_valid.detach().bool().cpu()
                            outside_values = outside_values[mask_cpu]
                            if gap_values is not None:
                                gap_values = gap_values[mask_cpu]
                        diagnostic_values[outside_key].extend(float(value) for value in outside_values.tolist())
                        if gap_values is not None:
                            diagnostic_values[gap_key].extend(float(value) for value in gap_values.tolist())
                    if mean_tensor is not None:
                        diagnostic_values[mean_key].extend(float(value) for value in mean_tensor.detach().float().cpu().tolist())
                    if std_tensor is not None:
                        diagnostic_values[std_key].extend(float(value) for value in std_tensor.detach().float().cpu().tolist())
                for metric_name in [
                    "cross_attn_to_context_mean",
                    "cross_attn_to_neighborhood_mean",
                    "cross_scale_attention_entropy",
                    "context_dropout_rate_actual",
                    "neighborhood_dropout_rate_actual",
                    "graph_gate_mean",
                    "graph_attention_entropy",
                    "graph_valid_neighbor_count",
                    "neighborhood_branch_gate",
                    "neighborhood_residual_scale",
                ]:
                    metric_tensor = diagnostics.get(metric_name)
                    if metric_tensor is not None:
                        diagnostic_values[metric_name].extend(float(value) for value in metric_tensor.detach().float().cpu().tolist())

            if collect_predictions:
                assert prediction_positions_cpu is not None
                expected_severity = compute_expected_severity_from_probabilities(class_probabilities_cpu, prediction_positions_cpu)
                for row_index, meta in enumerate(batch["meta"]):
                    record = {
                        "sample_index": int(batch["sample_index"][row_index].item()),
                        "meta": meta,
                        "target": int(targets_cpu[row_index]),
                        "prediction": int(predictions[row_index]),
                        "target_name": CLASS_NAMES[int(targets_cpu[row_index])],
                        "prediction_name": CLASS_NAMES[int(predictions[row_index])],
                        "probabilities": [float(value) for value in class_probabilities_cpu[row_index].tolist()],
                        "expected_severity": float(expected_severity[row_index].item()),
                        "severity_score": tensor_item_or_none(outputs.get("severity_score"), row_index),
                        "damage_aux_score_tight": tensor_item_or_none(outputs.get("damage_aux_scores", {}).get("tight"), row_index),
                        "damage_aux_score_context": tensor_item_or_none(outputs.get("damage_aux_scores", {}).get("context"), row_index),
                        "damage_aux_score_neighborhood": tensor_item_or_none(outputs.get("damage_aux_scores", {}).get("neighborhood"), row_index),
                        "tight_gate_inside": tensor_item_or_none(diagnostics.get("tight_gate_inside"), row_index),
                        "tight_gate_outside": tensor_item_or_none(diagnostics.get("tight_gate_outside"), row_index),
                        "tight_gate_gap": tensor_item_or_none(diagnostics.get("tight_gate_gap"), row_index),
                        "context_gate_inside": tensor_item_or_none(diagnostics.get("context_gate_inside"), row_index),
                        "context_gate_outside": tensor_item_or_none(diagnostics.get("context_gate_outside"), row_index),
                        "context_gate_gap": tensor_item_or_none(diagnostics.get("context_gate_gap"), row_index),
                        "neighborhood_gate_inside": tensor_item_or_none(diagnostics.get("neighborhood_gate_inside"), row_index),
                        "neighborhood_gate_outside": tensor_item_or_none(diagnostics.get("neighborhood_gate_outside"), row_index),
                        "neighborhood_gate_gap": tensor_item_or_none(diagnostics.get("neighborhood_gate_gap"), row_index),
                        "cross_attn_to_context_mean": tensor_item_or_none(diagnostics.get("cross_attn_to_context_mean"), row_index),
                        "cross_attn_to_neighborhood_mean": tensor_item_or_none(diagnostics.get("cross_attn_to_neighborhood_mean"), row_index),
                        "cross_scale_attention_entropy": tensor_item_or_none(diagnostics.get("cross_scale_attention_entropy"), row_index),
                        "graph_gate_mean": tensor_item_or_none(diagnostics.get("graph_gate_mean"), row_index),
                        "graph_attention_entropy": tensor_item_or_none(diagnostics.get("graph_attention_entropy"), row_index),
                        "context_dependency_score": (
                            (tensor_item_or_none(diagnostics.get("cross_attn_to_context_mean"), row_index) or 0.0)
                            + (tensor_item_or_none(diagnostics.get("cross_attn_to_neighborhood_mean"), row_index) or 0.0)
                        ),
                    }
                    graph_weights = outputs.get("graph_attention_weights")
                    if graph_weights is not None:
                        weights_row = graph_weights[row_index].detach().float().cpu()
                        record["graph_attention_weights_summary"] = {
                            "mean": float(weights_row.mean().item()),
                            "max": float(weights_row.max().item()),
                            "min": float(weights_row.min().item()),
                        }
                    for stat_name, stat_tensor in outputs.get("feature_stats", {}).items():
                        record[stat_name] = float(stat_tensor[row_index].detach().float().cpu().item())
                    prediction_records.append(record)

            progress.set_postfix(loss=f"{loss_total.detach().item():.4f}")

    metrics, report = compute_classification_metrics(all_targets, all_predictions, CLASS_NAMES)
    metrics["loss"] = total_loss / max(total_samples, 1)
    metrics["loss_terms"] = {key: value / max(total_samples, 1) for key, value in loss_term_sums.items()}
    metrics["feature_stats"] = {stat_name: summarize_values(values) for stat_name, values in feature_stat_values.items()}
    metrics["amp_enabled"] = bool(amp_settings["enabled"])
    metrics["amp_dtype"] = amp_settings["dtype_name"]
    metrics["grad_scaler_enabled"] = bool(scaler is not None and scaler.is_enabled())
    metrics["skipped_nonfinite_grad_batches"] = int(skipped_nonfinite_grad_batches)
    metrics["num_valid_samples"] = int(total_samples)
    metrics["all_batches_skipped"] = bool(total_samples == 0)
    metrics["graph_enabled"] = bool(config["model"].get("enable_neighborhood_graph", False))

    for metric_name, values in diagnostic_values.items():
        metrics[metric_name] = _mean_or_none(values)

    for scale_name in SCALE_NAMES:
        scores = damage_aux_scores_by_scale.get(scale_name, [])
        targets_for_scale = damage_aux_targets_by_scale.get(scale_name, [])
        acc_key = f"damage_aux_acc_{scale_name}"
        auc_key = f"damage_aux_auc_{scale_name}"
        if scores:
            scores_tensor = torch.tensor(scores, dtype=torch.float32)
            targets_tensor = torch.tensor(targets_for_scale, dtype=torch.float32)
            predictions_tensor = (torch.sigmoid(scores_tensor) >= 0.5).to(dtype=torch.float32)
            metrics[acc_key] = float((predictions_tensor == targets_tensor).float().mean().item())
            if torch.unique(targets_tensor).numel() > 1:
                try:
                    metrics[auc_key] = float(roc_auc_score(targets_tensor.numpy(), torch.sigmoid(scores_tensor).numpy()))
                except ValueError:
                    metrics[auc_key] = None
            else:
                metrics[auc_key] = None
        else:
            metrics[acc_key] = None
            metrics[auc_key] = None

    metrics["damage_aux_acc"] = metrics.get("damage_aux_acc_tight")
    metrics["damage_aux_auc"] = metrics.get("damage_aux_auc_tight")

    if collect_severity_aux_stats and severity_aux_scores:
        severity_scores_tensor = torch.tensor(severity_aux_scores, dtype=torch.float32)
        severity_targets_tensor = torch.tensor(severity_aux_targets, dtype=torch.float32)
        severity_labels_tensor = torch.tensor(severity_aux_labels, dtype=torch.long) if severity_aux_labels else None
        metrics["severity_aux_mae"] = float((severity_scores_tensor - severity_targets_tensor).abs().mean().item())
        for class_index, _ in enumerate(CLASS_NAMES):
            key = f"severity_score_mean_cls{class_index}"
            if severity_labels_tensor is None:
                metrics[key] = None
                continue
            class_mask = severity_labels_tensor == class_index
            metrics[key] = float(severity_scores_tensor[class_mask].mean().item()) if class_mask.any() else None
    else:
        metrics["severity_aux_mae"] = None
        for class_index, _ in enumerate(CLASS_NAMES):
            metrics[f"severity_score_mean_cls{class_index}"] = None

    if all_probabilities:
        probabilities_np = torch.tensor(all_probabilities, dtype=torch.float64).numpy()
        y_true_np = torch.tensor(all_targets, dtype=torch.int64).numpy()
        y_pred_np = torch.tensor(all_predictions, dtype=torch.int64).numpy()
        positions_np = torch.linspace(0.0, 1.0, steps=len(CLASS_NAMES), dtype=torch.float64).numpy()
        expected_severity_error, _ = compute_expected_severity_error(probabilities_np, y_true_np, positions_np)
        expected_severity_bias, _ = compute_expected_severity_bias(probabilities_np, y_true_np, positions_np)
        metrics["expected_severity_error"] = float(expected_severity_error)
        metrics["expected_severity_bias"] = float(expected_severity_bias)
        metrics["mean_predicted_distribution_by_true_class"] = summarize_mean_predicted_distribution_by_true_class(
            probabilities_np,
            y_true_np,
            CLASS_NAMES,
            positions_np,
        )
        metrics["minor_major_bidirectional"] = compute_minor_major_bidirectional_analysis(
            probabilities_np,
            y_true_np,
            y_pred_np,
            CLASS_NAMES,
            positions_np,
        )
    minor_metrics = metrics["per_class"].get("minor-damage", {})
    metrics["minor_precision"] = float(minor_metrics.get("precision", 0.0))
    metrics["minor_recall"] = float(minor_metrics.get("recall", 0.0))
    metrics["minor_f1"] = float(minor_metrics.get("f1", 0.0))
    confusion = metrics.get("confusion_matrix", [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES])
    minor_index = CLASS_NAMES.index("minor-damage")
    major_index = CLASS_NAMES.index("major-damage")
    no_index = CLASS_NAMES.index("no-damage")
    minor_support = max(float(metrics["per_class"].get("minor-damage", {}).get("support", 0)), 1.0)
    major_support = max(float(metrics["per_class"].get("major-damage", {}).get("support", 0)), 1.0)
    metrics["gt_minor_pred_no_rate"] = float(confusion[minor_index][no_index] / minor_support)
    metrics["gt_minor_pred_major_rate"] = float(confusion[minor_index][major_index] / minor_support)
    metrics["gt_major_pred_minor_rate"] = float(confusion[major_index][minor_index] / major_support)

    return {
        "metrics": metrics,
        "report": report,
        "prediction_records": prediction_records,
    }
