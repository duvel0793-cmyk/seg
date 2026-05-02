from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from datasets import XBDInstanceDataset, xbd_instance_collate_fn
from engine import build_epoch_loss, run_epoch
from models import build_model
from utils.checkpoint import save_checkpoint
from utils.ema import ModelEMA
from utils.misc import (
    CLASS_NAMES,
    append_jsonl,
    ensure_dir,
    get_output_dir,
    get_enabled_scale_names,
    get_split_list,
    load_config,
    resolve_amp_dtype,
    resolve_device,
    seed_worker,
    set_seed,
    prediction_record_to_table_row,
    write_json,
    write_csv_rows,
    write_text,
    write_yaml,
)
from utils.scheduler import WarmupCosineScheduler

DEFAULT_TRAIN_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "evidence_clean_v1_next.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_TRAIN_CONFIG_PATH))
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def build_dataloader(dataset: XBDInstanceDataset, *, batch_size: int, num_workers: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    data_cfg = dataset.config.get("data", {})
    pin_memory = bool(data_cfg.get("pin_memory", torch.cuda.is_available()))
    persistent_workers = bool(data_cfg.get("persistent_workers", True)) and num_workers > 0
    prefetch_factor = int(data_cfg.get("prefetch_factor", 2))
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": xbd_instance_collate_fn,
        "worker_init_fn": seed_worker,
        "generator": generator,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(
        dataset,
        **loader_kwargs,
    )


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> tuple[torch.optim.Optimizer, list[dict[str, Any]]]:
    base_lr = float(
        config["training"].get(
            "lr",
            config["training"].get("new_module_lr", config["training"].get("backbone_lr", 2.0e-4)),
        )
    )
    backbone_lr_cfg = config["training"].get("backbone_lr")
    new_module_lr_cfg = config["training"].get("new_module_lr")
    backbone_lr = base_lr if backbone_lr_cfg in {None, ""} else float(backbone_lr_cfg)
    new_module_lr = base_lr if new_module_lr_cfg in {None, ""} else float(new_module_lr_cfg)
    weight_decay = float(config["training"]["weight_decay"])
    grouped_params = {
        "backbone_decay": {"params": [], "lr": backbone_lr, "weight_decay": weight_decay, "name": "backbone_decay"},
        "backbone_no_decay": {"params": [], "lr": backbone_lr, "weight_decay": 0.0, "name": "backbone_no_decay"},
        "new_decay": {"params": [], "lr": new_module_lr, "weight_decay": weight_decay, "name": "new_decay"},
        "new_no_decay": {"params": [], "lr": new_module_lr, "weight_decay": 0.0, "name": "new_no_decay"},
    }
    seen: set[int] = set()
    group_summaries: list[dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parameter_id = id(parameter)
        if parameter_id in seen:
            continue
        seen.add(parameter_id)
        is_backbone = name.startswith("backbone.")
        no_decay = parameter.ndim == 1 or name.endswith(".bias")
        group_key = (
            "backbone_no_decay"
            if is_backbone and no_decay
            else "backbone_decay"
            if is_backbone
            else "new_no_decay"
            if no_decay
            else "new_decay"
        )
        grouped_params[group_key]["params"].append(parameter)
    optimizer_groups = [group for group in grouped_params.values() if group["params"]]
    for group in optimizer_groups:
        num_params = int(sum(parameter.numel() for parameter in group["params"]))
        group_summaries.append(
            {
                "name": group["name"],
                "num_params": num_params,
                "lr": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
            }
        )
        print(
            f"optimizer_group name={group['name']} num_params={num_params} "
            f"lr={group['lr']:.6g} weight_decay={group['weight_decay']:.6g}"
        )
    optimizer = torch.optim.AdamW(optimizer_groups)
    return optimizer, group_summaries


def metric_value(metrics: dict[str, Any], name: str) -> float:
    metric_key = {
        "ema_macro_f1": "macro_f1",
        "raw_macro_f1": "macro_f1",
        "qwk": "QWK",
    }.get(name, name)
    value = metrics.get(metric_key)
    if value is None:
        raise KeyError(f"Metric '{metric_key}' is not available in validation metrics.")
    if isinstance(value, (dict, list, tuple)):
        raise ValueError(f"Metric '{metric_key}' is not a scalar metric and cannot be used for monitoring.")
    return float(value)


def select_validation_result(
    monitor_name: str,
    *,
    raw_val_result: dict[str, Any] | None,
    ema_val_result: dict[str, Any] | None,
) -> tuple[str, str, dict[str, Any]]:
    if monitor_name.startswith("raw_"):
        if raw_val_result is None:
            raise RuntimeError(f"Monitor '{monitor_name}' requires raw validation results, but validate_raw is disabled.")
        return "raw", monitor_name.removeprefix("raw_"), raw_val_result
    if monitor_name.startswith("ema_"):
        if ema_val_result is None:
            raise RuntimeError(f"Monitor '{monitor_name}' requires EMA validation results, but validate_ema is disabled.")
        return "ema", monitor_name.removeprefix("ema_"), ema_val_result
    if raw_val_result is not None:
        return "raw", monitor_name, raw_val_result
    if ema_val_result is not None:
        return "ema", monitor_name, ema_val_result
    raise RuntimeError("No validation results are available for monitoring.")


def per_class_f1_dict(metrics: dict[str, Any]) -> dict[str, float]:
    per_class = metrics.get("per_class", {})
    return {class_name: float((per_class.get(class_name) or {}).get("f1", 0.0)) for class_name in CLASS_NAMES}


def format_per_class_f1(metrics: dict[str, Any]) -> str:
    values = per_class_f1_dict(metrics)
    return ",".join(f"{class_name}:{values[class_name]:.4f}" for class_name in CLASS_NAMES)


def apply_debug_overrides(config: dict[str, Any]) -> dict[str, Any]:
    config["dataset"]["debug_subset"] = int(config["dataset"].get("debug_subset", 0) or 4)
    config["dataset"]["debug_max_samples"] = int(config["dataset"].get("debug_max_samples", 0) or 8)
    config["training"]["epochs"] = min(int(config["training"]["epochs"]), 1)
    config["training"]["num_workers"] = 0
    config["eval"]["num_workers"] = 0
    config["eval"]["validate_raw"] = bool(config["eval"].get("validate_raw", True)) or not bool(
        config["training"].get("ema_enabled", False)
    )
    config["eval"]["save_diagnostics"] = True
    config["project"]["output_dir"] = str(Path(config["project"]["output_dir"]) / "debug")
    return config


def _active_loss_weights(config: dict[str, Any]) -> dict[str, float]:
    return {
        "corn_weight": float(config["loss"].get("corn_weight", 0.0)),
        "binary_damage_weight": float(config["loss"].get("binary_damage_weight", 0.0)),
        "severity_corn_weight": float(config["loss"].get("severity_corn_weight", 0.0)),
        "final_ce_weight": float(config["loss"].get("final_ce_weight", 0.0)),
        "global_corn_aux_weight": float(config["loss"].get("global_corn_aux_weight", 0.0)),
        "evidence_pixel_weight": float(config["loss"].get("evidence_pixel_weight", 0.0)),
        "evidence_mil_weight": float(config["loss"].get("evidence_mil_weight", 0.0)),
        "severity_map_weight": float(config["loss"].get("severity_map_weight", 0.0)),
        "damage_aux_weight": float(config["loss"].get("damage_aux_weight", 0.0)),
        "unchanged_weight": float(config["loss"].get("unchanged_weight", 0.0)),
        "gate_bg_weight": float(config["loss"].get("gate_bg_weight", 0.0)),
        "gate_contrast_weight": float(config["loss"].get("gate_contrast_weight", 0.0)),
        "minor_no_aux_weight": float(config["loss"].get("minor_no_aux_weight", 0.0)),
        "minor_major_aux_weight": float(config["loss"].get("minor_major_aux_weight", 0.0)),
        "structural_binary_weight": float(config["loss"].get("structural_binary_weight", 0.0)),
        "low_stage_weight": float(config["loss"].get("low_stage_weight", 0.0)),
        "high_stage_weight": float(config["loss"].get("high_stage_weight", 0.0)),
        "scale_router_ce_weight": float(config["loss"].get("scale_router_ce_weight", 0.0)),
    }


def _resolve_change_block_type_for_summary(model_cfg: dict[str, Any]) -> str:
    block_type = str(model_cfg.get("change_block_type", "auto")).lower()
    if block_type == "auto":
        if bool(model_cfg.get("enable_damage_bdfm_lite", False)):
            return "bdfm_lite"
        if bool(model_cfg.get("enable_damage_aware_block", True)):
            return "legacy"
        return "none"
    return block_type


def build_run_summary(config: dict[str, Any], model: torch.nn.Module, optimizer_groups: list[dict[str, Any]], *, include_model_repr: bool) -> str:
    model_cfg = config["model"]
    crop_scales = config["dataset"]["crop_scales"]
    enabled_scales = get_enabled_scale_names(config)
    active_losses = {name: value for name, value in _active_loss_weights(config).items() if value > 0.0}
    effective_change_block_type = _resolve_change_block_type_for_summary(model_cfg)
    lines = [
        f"project_name: {config['project']['name']}",
        f"output_dir: {config['project']['output_dir']}",
        f"enabled_scales: {enabled_scales}",
        "scale_output_sizes:",
    ]
    for scale_name in ("tight", "context", "neighborhood"):
        lines.append(
            f"  {scale_name}: enabled={bool(crop_scales[scale_name].get('enabled', True))} "
            f"size={crop_scales[scale_name]['output_size']} context_factor={crop_scales[scale_name]['context_factor']}"
        )
    lines.extend(
        [
            f"input_mode: {model_cfg['input_mode']}",
            f"backbone: {model_cfg['backbone']}",
            f"use_neighborhood_branch: {bool(model_cfg['use_neighborhood_branch'])}",
            f"enable_alignment: {bool(model_cfg.get('enable_alignment', True))}",
            f"enable_damage_aware_block: {bool(model_cfg.get('enable_damage_aware_block', True))}",
            f"enable_change_gate: {bool(model_cfg.get('enable_change_gate', True))}",
            f"change_block_type_configured: {model_cfg.get('change_block_type', 'auto')}",
            f"change_block_type_effective: {effective_change_block_type}",
            f"enable_damage_bdfm_lite: {bool(model_cfg.get('enable_damage_bdfm_lite', False))}",
            f"enable_edqa_lite: {bool(model_cfg.get('enable_edqa_lite', False))}",
            f"edqa_lite_scales: {list(model_cfg.get('edqa_lite_scales', ['context']))}",
            f"edqa_lite_window_size: {model_cfg.get('edqa_lite_window_size')}",
            f"edqa_lite_heads: {model_cfg.get('edqa_lite_heads')}",
            f"use_evidence_head: {bool(model_cfg.get('use_evidence_head', True))}",
            f"evidence_scales: {list(model_cfg.get('evidence_scales', ['tight', 'context', 'neighborhood']))}",
            f"evidence_fusion_mode: {model_cfg.get('evidence_fusion_mode', 'concat_mlp_original')}",
            f"use_hierarchical_head: {bool(model_cfg.get('use_hierarchical_head', True))}",
            f"head_type: {model_cfg.get('head_type')}",
            f"use_global_corn_aux: {bool(model_cfg.get('use_global_corn_aux', True))}",
            f"enable_minor_boundary_aux: {bool(model_cfg.get('enable_minor_boundary_aux', False))}",
            f"use_structural_two_stage_head: {bool(model_cfg.get('use_structural_two_stage_head', False))}",
            f"structural_two_stage_use_for_prediction: {bool(model_cfg.get('structural_two_stage_use_for_prediction', False))}",
            f"structural_two_stage_prob_mix_weight: {float(model_cfg.get('structural_two_stage_prob_mix_weight', 0.0))}",
            f"use_severity_aware_scale_router: {bool(model_cfg.get('use_severity_aware_scale_router', False))}",
            (
                "local_attention_layers: "
                f"tight={model_cfg.get('local_attention_layers_tight', model_cfg.get('local_attention_layers'))}, "
                f"context={model_cfg.get('local_attention_layers_context', model_cfg.get('local_attention_layers'))}, "
                f"neighborhood={model_cfg.get('local_attention_layers_neighborhood', model_cfg.get('local_attention_layers'))}"
            ),
            f"cross_scale_layers: {model_cfg.get('cross_scale_layers', 0)}",
            f"cross_scale_fusion_mode: {model_cfg.get('cross_scale_fusion_mode', 'tight_query_all_kv')}",
            f"ema_enabled: {bool(config['training'].get('ema_enabled', False))}",
            "active_losses:",
        ]
    )
    if active_losses:
        for name, value in sorted(active_losses.items()):
            lines.append(f"  {name}: {value}")
    else:
        lines.append("  none")
    lines.append("optimizer_groups:")
    for group in optimizer_groups:
        lines.append(
            f"  {group['name']}: num_params={group['num_params']} lr={group['lr']} weight_decay={group['weight_decay']}"
        )
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    lines.extend(
        [
            f"total_params: {total_params}",
            f"trainable_params: {trainable_params}",
        ]
    )
    if include_model_repr:
        lines.extend(["", repr(model)])
    return "\n".join(lines)


def write_prediction_records(path: Path, records: list[dict[str, Any]]) -> None:
    if path.exists():
        path.unlink()
    for record in records:
        append_jsonl(path, record)


def write_prediction_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    rows = [prediction_record_to_table_row(record) for record in records]
    fieldnames = [
        "sample_id",
        "tile_id",
        "building_id",
        "gt_label",
        "corn_pred",
        "final_pred",
        "structural_pred",
        "scale_router_pred",
        "prob_no",
        "prob_minor",
        "prob_major",
        "prob_destroyed",
        "corn_prob_no",
        "corn_prob_minor",
        "corn_prob_major",
        "corn_prob_destroyed",
        "struct_prob_no",
        "struct_prob_minor",
        "struct_prob_major",
        "struct_prob_destroyed",
        "router_prob_no",
        "router_prob_minor",
        "router_prob_major",
        "router_prob_destroyed",
    ]
    write_csv_rows(path, fieldnames, rows)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.debug:
        config = apply_debug_overrides(config)
    set_seed(int(config["training"]["seed"]))
    output_dir = get_output_dir(config)
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    logs_dir = ensure_dir(output_dir / "logs")

    train_dataset = XBDInstanceDataset(config=config, split="train", list_path=get_split_list(config, "train"), is_train=True)
    val_dataset = XBDInstanceDataset(config=config, split="val", list_path=get_split_list(config, "val"), is_train=False)
    config["loss"]["dataset_class_counts"] = [int(value) for value in train_dataset.class_counts]
    config["data"]["train_instance_count"] = int(len(train_dataset))
    config["data"]["val_instance_count"] = int(len(val_dataset))
    write_yaml(output_dir / "run_config.yaml", config)
    train_loader = build_dataloader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["training"]["num_workers"]),
        shuffle=True,
        seed=int(config["training"]["seed"]),
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=int(config["eval"]["batch_size"]),
        num_workers=int(config["eval"]["num_workers"]),
        shuffle=False,
        seed=int(config["training"]["seed"]),
    )

    device = resolve_device()
    amp_dtype = resolve_amp_dtype(config, device)
    model = build_model(config).to(device)
    if bool(config["training"].get("channels_last", False)):
        model = model.to(memory_format=torch.channels_last)
    criterion = build_epoch_loss(config).to(device)
    optimizer, optimizer_group_summaries = build_optimizer(model, config)
    console_summary = build_run_summary(config, model, optimizer_group_summaries, include_model_repr=False)
    file_summary = build_run_summary(config, model, optimizer_group_summaries, include_model_repr=True)
    print(console_summary)
    write_text(output_dir / "model_summary.txt", file_summary + "\n")
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_epochs=int(config["training"]["epochs"]),
        warmup_epochs=int(config["training"]["warmup_epochs"]),
        min_lr_ratio=float(config["training"].get("min_lr_ratio", 0.01)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16))
    ema = ModelEMA(model, decay=float(config["training"]["ema_decay"])) if bool(config["training"]["ema_enabled"]) else None
    if bool(config["training"].get("compile_model", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)

    best_checkpoint_value = float("-inf")
    best_early_stop_value = float("-inf")
    epochs_without_improvement = 0
    early_stop_metric = str(config["training"]["early_stop_metric"])
    save_best_by = str(config["eval"].get("save_best_by", "ema_macro_f1"))
    history_path = logs_dir / "history.jsonl"
    if history_path.exists():
        history_path.unlink()
    write_json(output_dir / "optimizer_groups.json", optimizer_group_summaries)
    collect_eval_predictions = True

    for epoch in range(int(config["training"]["epochs"])):
        scheduler.step(epoch)
        train_result = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            config=config,
            optimizer=optimizer,
            scaler=scaler,
            ema=ema,
            epoch_index=epoch,
            collect_predictions=False,
        )

        raw_val_result = None
        collect_raw_predictions = collect_eval_predictions and (
            ema is None
            or not bool(config["eval"].get("validate_ema", True))
            or save_best_by.startswith("raw_")
        )
        if bool(config["eval"].get("validate_raw", True)) and ((epoch + 1) % int(config["eval"].get("validate_raw_every", 1)) == 0):
            raw_val_result = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                config=config,
                optimizer=None,
                scaler=None,
                ema=None,
                epoch_index=epoch,
                collect_predictions=collect_raw_predictions,
            )
        ema_val_result = None
        if ema is not None and bool(config["eval"].get("validate_ema", True)) and (
            (epoch + 1) % int(config["eval"].get("validate_ema_every", 1)) == 0
        ):
            ema_val_result = run_epoch(
                model=ema.module,
                loader=val_loader,
                criterion=criterion,
                device=device,
                config=config,
                optimizer=None,
                scaler=None,
                ema=None,
                epoch_index=epoch,
                collect_predictions=collect_eval_predictions,
            )

        primary_val_result = ema_val_result if ema_val_result is not None else raw_val_result
        if primary_val_result is None:
            raise RuntimeError("No validation pass was executed. Enable validate_raw or validate_ema.")

        save_monitor_source, save_metric_name, save_monitor_result = select_validation_result(
            save_best_by,
            raw_val_result=raw_val_result,
            ema_val_result=ema_val_result,
        )
        save_value = metric_value(save_monitor_result["metrics"], save_metric_name)
        improved = save_value > best_checkpoint_value
        if improved:
            best_checkpoint_value = save_value
        _, stop_metric_name, stop_monitor_result = select_validation_result(
            early_stop_metric,
            raw_val_result=raw_val_result,
            ema_val_result=ema_val_result,
        )
        stop_value = metric_value(stop_monitor_result["metrics"], stop_metric_name)
        if stop_value > best_early_stop_value:
            best_early_stop_value = stop_value
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if improved:
            selected_model_state = model.state_dict() if save_monitor_source == "raw" else ema.module.state_dict()
            checkpoint_state = {
                "epoch": epoch + 1,
                "model_state_dict": copy.deepcopy(selected_model_state),
                "raw_model_state_dict": copy.deepcopy(model.state_dict()),
                "ema_state_dict": None if ema is None else copy.deepcopy(ema.module.state_dict()),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "config": config,
                "best_model_source": save_monitor_source,
                "train_metrics": train_result["metrics"],
                "ema_val_metrics": None if ema_val_result is None else ema_val_result["metrics"],
                "raw_val_metrics": None if raw_val_result is None else raw_val_result["metrics"],
            }
            save_checkpoint(checkpoints_dir / f"best_{save_best_by}.pth", checkpoint_state)
            if save_monitor_result.get("diagnostics") is not None:
                write_json(output_dir / "diagnostics_best.json", save_monitor_result["diagnostics"])
            write_json(output_dir / "confusion_matrix_best.json", save_monitor_result["metrics"].get("confusion_matrix", []))
            if save_monitor_result.get("threshold_sweep") is not None:
                write_json(output_dir / "threshold_sweep_best.json", save_monitor_result["threshold_sweep"])
            if save_monitor_result.get("prediction_records"):
                write_prediction_records(output_dir / "predictions_best.jsonl", save_monitor_result["prediction_records"])
                write_prediction_records_csv(output_dir / "predictions_best.csv", save_monitor_result["prediction_records"])

        if primary_val_result.get("diagnostics") is not None:
            write_json(output_dir / f"diagnostics_epoch_{epoch + 1:02d}.json", primary_val_result["diagnostics"])
            write_json(output_dir / "diagnostics_last.json", primary_val_result["diagnostics"])
        write_json(output_dir / "confusion_matrix_last.json", primary_val_result["metrics"].get("confusion_matrix", []))
        if primary_val_result.get("threshold_sweep") is not None:
            write_json(output_dir / f"threshold_sweep_epoch_{epoch + 1:02d}.json", primary_val_result["threshold_sweep"])
            write_json(output_dir / "threshold_sweep.json", primary_val_result["threshold_sweep"])
        if primary_val_result.get("prediction_records"):
            write_prediction_records(output_dir / "predictions_last.jsonl", primary_val_result["prediction_records"])
            write_prediction_records_csv(output_dir / "predictions_last.csv", primary_val_result["prediction_records"])

        raw_metrics = None if raw_val_result is None else raw_val_result["metrics"]
        ema_metrics = None if ema_val_result is None else ema_val_result["metrics"]
        primary_metrics = primary_val_result["metrics"]

        history_record = {
            "epoch": epoch + 1,
            "train_loss": train_result["loss"],
            "corn_loss": train_result["loss_terms"].get("loss_corn"),
            "final_ce_loss": train_result["loss_terms"].get("loss_final_ce"),
            "structural_binary_loss": train_result["loss_terms"].get("loss_structural_binary"),
            "low_stage_loss": train_result["loss_terms"].get("loss_low_stage"),
            "high_stage_loss": train_result["loss_terms"].get("loss_high_stage"),
            "scale_router_ce_loss": train_result["loss_terms"].get("loss_scale_router_ce"),
            "raw_macro_f1": None if raw_metrics is None else raw_metrics["macro_f1"],
            "raw_per_class_f1": None if raw_metrics is None else per_class_f1_dict(raw_metrics),
            "ema_macro_f1": None if ema_metrics is None else ema_metrics["macro_f1"],
            "ema_per_class_f1": None if ema_metrics is None else per_class_f1_dict(ema_metrics),
            "raw_qwk": None if raw_metrics is None else raw_metrics["QWK"],
            "ema_qwk": None if ema_metrics is None else ema_metrics["QWK"],
            "raw_far_error": None if raw_metrics is None else raw_metrics["far_error"],
            "ema_far_error": None if ema_metrics is None else ema_metrics["far_error"],
            "no_f1": primary_metrics["no_f1"],
            "minor_f1": primary_metrics["minor_f1"],
            "major_f1": primary_metrics["major_f1"],
            "destroyed_f1": primary_metrics["destroyed_f1"],
            "binary_damage_f1": primary_metrics["binary_damage_f1"],
            "damaged_severity_macro_f1": primary_metrics["damaged_severity_macro_f1"],
            "gt_minor_pred_no_rate": primary_metrics["gt_minor_pred_no_rate"],
            "gt_minor_pred_major_rate": primary_metrics["gt_minor_pred_major_rate"],
            "gt_major_pred_minor_rate": primary_metrics["gt_major_pred_minor_rate"],
            "gt_major_pred_no_rate": primary_metrics["gt_major_pred_no_rate"],
            "gt_destroyed_pred_major_rate": primary_metrics["gt_destroyed_pred_major_rate"],
            "pred_label_distribution": primary_metrics["prediction_distribution"],
            "gt_label_distribution": primary_metrics["gt_distribution"],
            "scale_router_gates": primary_metrics.get("scale_router_gates"),
            "structural_head_macro_f1": primary_metrics.get("structural_head_macro_f1"),
            "scale_router_macro_f1": primary_metrics.get("scale_router_macro_f1"),
            "final_macro_f1": primary_metrics.get("final_macro_f1") if primary_metrics.get("final_prediction_active") else None,
            "loss_corn": train_result["loss_terms"].get("loss_corn"),
            "loss_binary_damage": train_result["loss_terms"].get("loss_binary_damage"),
            "loss_severity_corn": train_result["loss_terms"].get("loss_severity_corn"),
            "loss_final_ce": train_result["loss_terms"].get("loss_final_ce"),
            "loss_global_corn_aux": train_result["loss_terms"].get("loss_global_corn_aux"),
            "loss_evidence_pixel": train_result["loss_terms"].get("loss_evidence_pixel"),
            "loss_evidence_mil": train_result["loss_terms"].get("loss_evidence_mil"),
            "loss_severity_map": train_result["loss_terms"].get("loss_severity_map"),
            "loss_minor_no_aux": train_result["loss_terms"].get("loss_minor_no_aux"),
            "loss_minor_major_aux": train_result["loss_terms"].get("loss_minor_major_aux"),
            "loss_damage_aux": train_result["loss_terms"].get("loss_damage_aux"),
            "loss_gate_bg": train_result["loss_terms"].get("loss_gate_bg"),
            "loss_gate_contrast": train_result["loss_terms"].get("loss_gate_contrast"),
            "loss_unchanged": train_result["loss_terms"].get("loss_unchanged"),
            "loss_structural_binary": train_result["loss_terms"].get("loss_structural_binary"),
            "loss_low_stage": train_result["loss_terms"].get("loss_low_stage"),
            "loss_high_stage": train_result["loss_terms"].get("loss_high_stage"),
            "loss_scale_router_ce": train_result["loss_terms"].get("loss_scale_router_ce"),
            "loss_total": train_result["loss_terms"].get("loss_total"),
            "evidence_pixel_weight_eff": train_result["loss_terms"].get("evidence_pixel_weight_eff"),
            "evidence_mil_weight_eff": train_result["loss_terms"].get("evidence_mil_weight_eff"),
            "severity_map_weight_eff": train_result["loss_terms"].get("severity_map_weight_eff"),
            "evidence_gate_alpha": train_result["logging_scalars"].get("evidence_gate_alpha"),
            "val_best_threshold_macro_f1": None if primary_metrics.get("best_threshold_by_macro_f1") is None else primary_metrics["best_threshold_by_macro_f1"]["threshold"],
            "val_best_threshold_qwk": None if primary_metrics.get("best_threshold_by_qwk") is None else primary_metrics["best_threshold_by_qwk"]["threshold"],
            "best_checkpoint_value": best_checkpoint_value,
            "best_early_stop_value": best_early_stop_value,
            "epochs_without_improvement": epochs_without_improvement,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        append_jsonl(history_path, history_record)
        log_parts = [
            f"epoch={epoch + 1}",
            f"train_loss={train_result['loss']:.4f}",
        ]
        if raw_metrics is not None:
            log_parts.extend(
                [
                    f"raw_macro_f1={raw_metrics['macro_f1']:.4f}",
                    f"raw_per_class_f1={format_per_class_f1(raw_metrics)}",
                    f"raw_qwk={raw_metrics['QWK']:.4f}",
                    f"raw_far_error={raw_metrics['far_error']:.4f}",
                ]
            )
        if ema_metrics is not None:
            log_parts.extend(
                [
                    f"ema_macro_f1={ema_metrics['macro_f1']:.4f}",
                    f"ema_qwk={ema_metrics['QWK']:.4f}",
                    f"ema_far_error={ema_metrics['far_error']:.4f}",
                ]
            )
        log_parts.extend(
            [
                f"no_f1={primary_metrics['no_f1']:.4f}",
                f"minor_f1={primary_metrics['minor_f1']:.4f}",
                f"major_f1={primary_metrics['major_f1']:.4f}",
                f"destroyed_f1={primary_metrics['destroyed_f1']:.4f}",
                f"damaged_sev_f1={primary_metrics['damaged_severity_macro_f1']:.4f}",
                f"corn_loss={train_result['loss_terms'].get('loss_corn', 0.0):.4f}",
                f"final_ce_loss={train_result['loss_terms'].get('loss_final_ce', 0.0):.4f}",
                f"struct_bin_loss={train_result['loss_terms'].get('loss_structural_binary', 0.0):.4f}",
                f"low_stage_loss={train_result['loss_terms'].get('loss_low_stage', 0.0):.4f}",
                f"high_stage_loss={train_result['loss_terms'].get('loss_high_stage', 0.0):.4f}",
                f"router_ce_loss={train_result['loss_terms'].get('loss_scale_router_ce', 0.0):.4f}",
                f"minor_no_rate={primary_metrics['gt_minor_pred_no_rate']:.4f}",
                f"minor_major_rate={primary_metrics['gt_minor_pred_major_rate']:.4f}",
                f"major_minor_rate={primary_metrics['gt_major_pred_minor_rate']:.4f}",
                f"evidence_gate_alpha={train_result['logging_scalars'].get('evidence_gate_alpha', 0.0):.4f}",
                f"best_thr_macro={history_record['val_best_threshold_macro_f1']}",
            ]
        )
        print(" ".join(log_parts))
        if epochs_without_improvement >= int(config["training"]["early_stop_patience"]):
            print("Early stopping triggered.")
            break


if __name__ == "__main__":
    main()
