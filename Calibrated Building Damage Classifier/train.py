from __future__ import annotations

import argparse
import copy
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import DEFAULT_CONFIG_PATH, apply_overrides, get_run_dir, load_config
from datasets.xbd_oracle_instance_damage import (
    CLASS_NAMES,
    XBDOracleInstanceDamageDataset,
    oracle_instance_collate_fn,
)
from engine import (
    build_epoch_loss,
    compute_collapse_diagnostics,
    resolve_amp_settings,
    resolve_device,
    resolve_numerical_issue_log_path,
    run_epoch,
)
from models import build_model
from utils.io import append_jsonl, ensure_dir, save_checkpoint, write_json, write_text, write_yaml
from utils.metrics import compute_composite_score
from utils.seed import seed_worker, set_seed

TRAIN_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "ablations" / "lean_train.yaml"


class TeeStream:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8") if streams else "utf-8"

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the cleaned instance-level damage classification mainline.")
    parser.add_argument("--config", type=str, default=str(TRAIN_DEFAULT_CONFIG_PATH))
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--train_list", type=str, default=None)
    parser.add_argument("--val_list", type=str, default=None)
    parser.add_argument("--test_list", type=str, default=None)
    parser.add_argument("--instance_source", type=str, default=None)
    parser.add_argument("--allow_tier3", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--input_mode", type=str, choices=["rgb", "rgbm"], default=None)
    parser.add_argument("--feature_dim", type=int, default=None)
    parser.add_argument("--token_count", type=int, default=None)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--amp_dtype", type=str, choices=["auto", "fp16", "bf16"], default=None)
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.set_defaults(amp=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    return parser.parse_args()


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for key, value in ema_state.items():
            model_value = model_state[key].detach()
            if not torch.is_floating_point(value):
                value.copy_(model_value)
                continue
            value.mul_(self.decay).add_(model_value.to(dtype=value.dtype), alpha=1.0 - self.decay)


def build_dataloader(
    dataset: XBDOracleInstanceDamageDataset,
    *,
    config: dict[str, Any],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    sampler = None
    if shuffle and bool(config.get("sampler", {}).get("enable_class_boost", False)):
        boost_map = {
            0: float(config["sampler"].get("no_damage_boost", 1.0)),
            1: float(config["sampler"].get("minor_boost", 1.0)),
            2: float(config["sampler"].get("major_boost", 1.0)),
            3: float(config["sampler"].get("destroyed_boost", 1.0)),
        }
        sample_weights = torch.tensor(
            [boost_map.get(int(sample["label"]), 1.0) for sample in dataset.samples],
            dtype=torch.double,
        )
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=oracle_instance_collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_cfg = config["optimizer"]
    backbone_lr = float(optimizer_cfg.get("backbone_lr", optimizer_cfg["lr"]))
    new_module_lr = float(optimizer_cfg.get("new_module_lr", optimizer_cfg["lr"]))
    backbone_parameters = []
    new_module_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_parameters.append(parameter)
        else:
            new_module_parameters.append(parameter)
    parameter_groups = []
    if backbone_parameters:
        parameter_groups.append({"params": backbone_parameters, "lr": backbone_lr})
    if new_module_parameters:
        parameter_groups.append({"params": new_module_parameters, "lr": new_module_lr})
    return torch.optim.AdamW(
        parameter_groups,
        lr=float(optimizer_cfg["lr"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )


class WarmupCosineScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, config: dict[str, Any]) -> None:
        self.optimizer = optimizer
        self.total_epochs = int(config["training"]["epochs"])
        self.warmup_epochs = int(config["scheduler"]["warmup_epochs"])
        self.min_lr_ratio = float(config["scheduler"]["min_lr_ratio"])
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.current_epoch = 0

    def _lr_multiplier(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return float(epoch + 1) / max(self.warmup_epochs, 1)
        if self.total_epochs <= self.warmup_epochs:
            return 1.0
        progress = float(epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return self.min_lr_ratio + ((1.0 - self.min_lr_ratio) * cosine)

    def step(self, epoch: int) -> None:
        self.current_epoch = int(epoch)
        multiplier = self._lr_multiplier(self.current_epoch)
        for base_lr, param_group in zip(self.base_lrs, self.optimizer.param_groups):
            param_group["lr"] = base_lr * multiplier

    def state_dict(self) -> dict[str, Any]:
        return {"current_epoch": int(self.current_epoch)}


def compute_epoch_composite_score(metrics: dict[str, Any], config: dict[str, Any]) -> float:
    training_cfg = config["training"]
    return compute_composite_score(
        macro_f1=float(metrics["macro_f1"]),
        qwk=float(metrics["QWK"]),
        far_error_rate=float(metrics["far_error_rate"]),
        alpha_qwk=float(training_cfg["composite_alpha_qwk"]),
        beta_far_error=float(training_cfg["composite_beta_far_error"]),
    )


def clone_checkpoint_with_model_weights(state: dict[str, Any], model_state_dict: dict[str, Any]) -> dict[str, Any]:
    checkpoint = copy.deepcopy(state)
    checkpoint["model_state_dict"] = copy.deepcopy(model_state_dict)
    return checkpoint


def make_checkpoint_state(
    *,
    epoch: int,
    model: torch.nn.Module,
    ema_model: ModelEMA | None,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    scaler: torch.amp.GradScaler | None,
    config: dict[str, Any],
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    ema_val_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "ema_state_dict": None if ema_model is None else ema_model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": None if scaler is None else scaler.state_dict(),
        "config": copy.deepcopy(config),
        "class_names": CLASS_NAMES,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "ema_val_metrics": ema_val_metrics,
    }


def metric_improved(current: float, best: float, mode: str, atol: float = 1e-12) -> bool:
    if mode == "max":
        return float(current) > (float(best) + float(atol))
    if mode == "min":
        return float(current) < (float(best) - float(atol))
    raise ValueError(f"Unsupported mode '{mode}'.")


def format_per_class_f1(metrics: dict[str, Any]) -> str:
    per_class = metrics.get("per_class", {})
    ordered_names = ["no-damage", "minor-damage", "major-damage", "destroyed"]
    aliases = {
        "no-damage": "no",
        "minor-damage": "minor",
        "major-damage": "major",
        "destroyed": "destroyed",
    }
    values = []
    for name in ordered_names:
        score = float(per_class.get(name, {}).get("f1", 0.0))
        values.append(f"{aliases[name]}={score:.4f}")
    return " ".join(values)


def format_threshold_probabilities(metrics: dict[str, Any]) -> str:
    threshold_means = metrics.get("mean_threshold_probabilities", {})
    if not threshold_means:
        return "p0=n/a p1=n/a p2=n/a"
    return " ".join(f"{name}={float(value):.4f}" for name, value in threshold_means.items())


def format_optional_metric(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    run_dir = ensure_dir(get_run_dir(config))
    logs_dir = ensure_dir(run_dir / "logs")
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")

    log_path = logs_dir / "train.log"
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("w", encoding="utf-8") as log_file:
        tee = TeeStream(original_stdout, log_file)
        sys.stdout = tee
        sys.stderr = tee
        try:
            set_seed(int(config["seed"]))
            write_yaml(run_dir / "run_config.yaml", config)

            device = resolve_device()
            amp_settings = resolve_amp_settings(config, device)
            print(f"Device: {device}")
            print(f"Run dir: {run_dir}")
            print(f"Ablation run: {config['ablation']['run_name']}")
            print(f"Backbone: ConvNeXtV2 ({config['model']['backbone_name']})")
            print(f"Alignment module: {config['model']['alignment_type']}")
            print("Scale branches: tight/context/neighborhood")
            print(f"Fusion: {'cross-scale attention' if bool(config['model']['use_cross_scale_attention']) else 'token mean'}")
            print("Head: CORN ordinal head")
            print("Silhouette branch: removed")
            print("Neighbor-aware branch: removed")
            print(
                "Token counts: "
                f"tight={int(config['model']['tight_token_count'])} "
                f"context={int(config['model']['context_token_count'])} "
                f"neighborhood={int(config['model']['neighborhood_token_count'])}"
            )
            print(f"Classifier type: {config['model']['classifier_type']}")
            print(f"Multi-context model: {bool(config['model'].get('use_multi_context_model', False))}")
            print(f"Neighborhood graph: {bool(config['model'].get('enable_neighborhood_graph', False))}")
            print(
                "Change suppression: "
                f"enabled={bool(config['model']['use_change_suppression'])} "
                f"pseudo={bool(config['model']['enable_pseudo_suppression'])} "
                f"fuse_to_tokens={bool(config['model']['fuse_change_to_tokens'])}"
            )
            print(f"Train split: train ({config['dataset']['train_list']})")
            print(f"Eval split: {config['eval']['split_name']} ({config['dataset']['val_list']})")
            print("Bridge export: disabled during training entrypoint")
            print(
                "AMP: "
                f"enabled={bool(amp_settings['enabled'])} "
                f"dtype={amp_settings['dtype_name']}"
            )
            print(f"EMA: {'enabled' if bool(config['training'].get('ema_enabled', False)) else 'disabled'}")
            if amp_settings["fallback_reason"] is not None:
                print(f"AMP fallback: {amp_settings['fallback_reason']}")

            train_dataset = XBDOracleInstanceDamageDataset(
                config=config,
                split_name="train",
                list_path=config["dataset"]["train_list"],
                is_train=True,
            )
            val_dataset = XBDOracleInstanceDamageDataset(
                config=config,
                split_name=config["eval"]["split_name"],
                list_path=config["dataset"]["val_list"],
                is_train=False,
            )
            train_loader = build_dataloader(
                train_dataset,
                config=config,
                batch_size=int(config["training"]["batch_size"]),
                num_workers=int(config["training"]["num_workers"]),
                shuffle=True,
                seed=int(config["seed"]),
            )
            val_loader = build_dataloader(
                val_dataset,
                config=config,
                batch_size=int(config["eval"]["batch_size"]),
                num_workers=int(config["eval"]["num_workers"]),
                shuffle=False,
                seed=int(config["seed"]),
            )

            model = build_model(config).to(device)
            ema_model = ModelEMA(model, decay=float(config["training"].get("ema_decay", 0.999))) if bool(config["training"].get("ema_enabled", False)) else None
            criterion = build_epoch_loss(config).to(device)
            optimizer = build_optimizer(model, config)
            scheduler = WarmupCosineScheduler(optimizer, config)
            scaler = None
            if device.type == "cuda":
                scaler = torch.amp.GradScaler(
                    "cuda",
                    enabled=bool(amp_settings["scaler_enabled"]),
                    init_scale=float(config["training"]["grad_scaler_init_scale"]),
                    growth_interval=int(config["training"]["grad_scaler_growth_interval"]),
                )

            history_path = logs_dir / "history.jsonl"
            if history_path.exists():
                history_path.unlink()
            numerical_issue_log_path = resolve_numerical_issue_log_path(config)
            if numerical_issue_log_path.exists():
                numerical_issue_log_path.unlink()

            best_scores = {
                "macro_f1": float("-inf"),
                "macro_f1_ema": float("-inf"),
                "qwk": float("-inf"),
                "far_error": float("inf"),
                "composite_score": float("-inf"),
            }
            best_epochs = {name: -1 for name in best_scores}
            early_stop_metric = str(config["training"]["early_stop_metric"])
            patience = int(config["training"]["early_stop_patience"])
            epochs_without_improvement = 0
            total_epochs = int(config["training"]["epochs"])

            for epoch in range(total_epochs):
                scheduler.step(epoch)
                print(f"\nEpoch {epoch + 1}/{total_epochs}")
                lr_parts = [f"group{group_index}={param_group['lr']:.8f}" for group_index, param_group in enumerate(optimizer.param_groups)]
                print(f"LR: {' '.join(lr_parts)}")

                train_result = run_epoch(
                    model=model,
                    loader=train_loader,
                    criterion=criterion,
                    device=device,
                    config=config,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch_index=epoch,
                    post_step_callback=(None if ema_model is None else ema_model.update),
                )
                val_result = run_epoch(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    device=device,
                    config=config,
                    optimizer=None,
                    scaler=None,
                    epoch_index=epoch,
                    collect_predictions=False,
                )
                ema_val_result = None
                if ema_model is not None:
                    ema_val_result = run_epoch(
                        model=ema_model.module,
                        loader=val_loader,
                        criterion=criterion,
                        device=device,
                        config=config,
                        optimizer=None,
                        scaler=None,
                        epoch_index=epoch,
                        collect_predictions=False,
                    )

                train_metrics = train_result["metrics"]
                val_metrics = val_result["metrics"]
                ema_val_metrics = None if ema_val_result is None else ema_val_result["metrics"]
                train_metrics["composite_score"] = compute_epoch_composite_score(train_metrics, config)
                val_metrics["composite_score"] = compute_epoch_composite_score(val_metrics, config)
                if ema_val_metrics is not None:
                    ema_val_metrics["composite_score"] = compute_epoch_composite_score(ema_val_metrics, config)

                print(
                    "Train: "
                    f"loss_total={train_metrics['loss']:.4f} "
                    f"loss_corn={train_metrics['loss_terms']['loss_corn']:.4f} "
                    f"loss_damage_tight={train_metrics['loss_terms']['loss_damage_tight']:.4f} "
                    f"loss_damage_context={train_metrics['loss_terms']['loss_damage_context']:.4f} "
                    f"loss_damage_neighborhood={train_metrics['loss_terms']['loss_damage_neighborhood']:.4f} "
                    f"loss_severity_aux={train_metrics['loss_terms']['loss_severity_aux']:.4f} "
                    f"loss_unchanged={train_metrics['loss_terms']['loss_unchanged']:.4f} "
                    f"loss_graph_consistency={train_metrics['loss_terms']['loss_graph_consistency']:.4f} "
                    f"acc={train_metrics['overall_accuracy']:.4f} "
                    f"macro_f1={train_metrics['macro_f1']:.4f} "
                    f"qwk={train_metrics['QWK']:.4f}"
                )
                print(
                    "Val:   "
                    f"loss_total={val_metrics['loss']:.4f} "
                    f"loss_corn={val_metrics['loss_terms']['loss_corn']:.4f} "
                    f"loss_damage_tight={val_metrics['loss_terms']['loss_damage_tight']:.4f} "
                    f"loss_damage_context={val_metrics['loss_terms']['loss_damage_context']:.4f} "
                    f"loss_damage_neighborhood={val_metrics['loss_terms']['loss_damage_neighborhood']:.4f} "
                    f"loss_severity_aux={val_metrics['loss_terms']['loss_severity_aux']:.4f} "
                    f"loss_unchanged={val_metrics['loss_terms']['loss_unchanged']:.4f} "
                    f"loss_graph_consistency={val_metrics['loss_terms']['loss_graph_consistency']:.4f} "
                    f"acc={val_metrics['overall_accuracy']:.4f} "
                    f"macro_f1={val_metrics['macro_f1']:.4f} "
                    f"weighted_f1={val_metrics['weighted_f1']:.4f} "
                    f"balanced_acc={val_metrics['balanced_accuracy']:.4f} "
                    f"qwk={val_metrics['QWK']:.4f} "
                    f"far_error={val_metrics['far_error_rate']:.4f} "
                    f"composite={val_metrics['composite_score']:.4f}"
                )
                if ema_val_metrics is not None:
                    print(
                        "EMA:   "
                        f"macro_f1={ema_val_metrics['macro_f1']:.4f} "
                        f"qwk={ema_val_metrics['QWK']:.4f} "
                        f"far_error={ema_val_metrics['far_error_rate']:.4f} "
                        f"composite={ema_val_metrics['composite_score']:.4f}"
                    )
                print(f"Val per-class F1: {format_per_class_f1(val_metrics)}")
                print(f"Val confusion matrix: {val_metrics['confusion_matrix']}")
                if bool(config["logging"].get("log_cross_scale_attention", True)):
                    print(
                        "Cross-scale attention: "
                        f"context={format_optional_metric(val_metrics.get('cross_attn_to_context_mean'))} "
                        f"neighborhood={format_optional_metric(val_metrics.get('cross_attn_to_neighborhood_mean'))} "
                        f"entropy={format_optional_metric(val_metrics.get('cross_scale_attention_entropy'))}"
                    )
                if bool(config["logging"].get("log_gate_stats", True)):
                    print(
                        "Scale gate gaps: "
                        f"tight={format_optional_metric(val_metrics.get('tight_gate_gap'))} "
                        f"context={format_optional_metric(val_metrics.get('context_gate_gap'))} "
                        f"neighborhood={format_optional_metric(val_metrics.get('neighborhood_gate_gap'))}"
                    )
                if bool(config["logging"].get("log_scale_metrics", True)):
                    print(
                        "Damage aux metrics: "
                        f"tight_auc={format_optional_metric(val_metrics.get('damage_aux_auc_tight'))} "
                        f"context_auc={format_optional_metric(val_metrics.get('damage_aux_auc_context'))} "
                        f"neighborhood_auc={format_optional_metric(val_metrics.get('damage_aux_auc_neighborhood'))}"
                    )
                if bool(config["logging"].get("log_severity_aux_stats", True)):
                    print(
                        "Severity aux metrics: "
                        f"mae={format_optional_metric(val_metrics.get('severity_aux_mae'))} "
                        f"cls0={format_optional_metric(val_metrics.get('severity_score_mean_cls0'))} "
                        f"cls1={format_optional_metric(val_metrics.get('severity_score_mean_cls1'))} "
                        f"cls2={format_optional_metric(val_metrics.get('severity_score_mean_cls2'))} "
                        f"cls3={format_optional_metric(val_metrics.get('severity_score_mean_cls3'))}"
                    )
                if bool(config["logging"].get("log_graph_metrics", True)):
                    print(
                        "Dropout / graph: "
                        f"context_dropout={format_optional_metric(val_metrics.get('context_dropout_rate_actual'))} "
                        f"neighborhood_dropout={format_optional_metric(val_metrics.get('neighborhood_dropout_rate_actual'))} "
                        f"graph_gate={format_optional_metric(val_metrics.get('graph_gate_mean'))} "
                        f"graph_entropy={format_optional_metric(val_metrics.get('graph_attention_entropy'))}"
                    )
                    print(
                        "Neighborhood branch: "
                        f"neighbor_gate={format_optional_metric(val_metrics.get('neighborhood_branch_gate'))} "
                        f"neighbor_residual={format_optional_metric(val_metrics.get('neighborhood_residual_scale'))}"
                    )
                if bool(config["logging"].get("log_context_diagnostic", True)):
                    print(
                        "Minor diagnostics: "
                        f"precision={format_optional_metric(val_metrics.get('minor_precision'))} "
                        f"recall={format_optional_metric(val_metrics.get('minor_recall'))} "
                        f"f1={format_optional_metric(val_metrics.get('minor_f1'))} "
                        f"minor_to_no={format_optional_metric(val_metrics.get('gt_minor_pred_no_rate'))} "
                        f"minor_to_major={format_optional_metric(val_metrics.get('gt_minor_pred_major_rate'))} "
                        f"major_to_minor={format_optional_metric(val_metrics.get('gt_major_pred_minor_rate'))}"
                    )
                collapse_diagnostics = compute_collapse_diagnostics(val_metrics, epoch_index=epoch)
                if collapse_diagnostics["collapse_pred_dominant_class"] is not None:
                    print(
                        "CRITICAL WARNING: prediction collapse detected "
                        f"class={collapse_diagnostics['collapse_pred_dominant_class']} "
                        f"ratio={collapse_diagnostics['collapse_pred_dominant_ratio']:.4f}"
                    )
                feature_stats = val_metrics.get("feature_stats", {})
                feature_means = {
                    name: feature_stats.get(name, {}).get("mean")
                    for name in ("tight_feature_norm", "context_feature_norm", "neighborhood_feature_norm")
                }
                if feature_stats and collapse_diagnostics["collapse_feature_norm_warning"]:
                    feature_norm_critical = any(
                        value is not None and float(value) > 8000.0
                        for value in feature_means.values()
                    )
                    warning_level = "CRITICAL WARNING" if feature_norm_critical else "WARNING"
                    print(
                        f"{warning_level}: feature norm growth detected "
                        f"tight={format_optional_metric(feature_means['tight_feature_norm'])} "
                        f"context={format_optional_metric(feature_means['context_feature_norm'])} "
                        f"neighborhood={format_optional_metric(feature_means['neighborhood_feature_norm'])}"
                    )
                if feature_stats and collapse_diagnostics["collapse_instance_feature_std_warning"]:
                    final_std = feature_stats.get("final_instance_feature_norm", {}).get("std")
                    print(f"WARNING: final instance feature collapse detected std={format_optional_metric(final_std)}")
                if collapse_diagnostics["collapse_aux_auc_warning"]:
                    print(
                        "WARNING: tight damage auxiliary AUC degraded "
                        f"auc={format_optional_metric(val_metrics.get('damage_aux_auc_tight'))}"
                    )
                if collapse_diagnostics["collapse_severity_warning"]:
                    print(
                        "WARNING: severity auxiliary collapsed "
                        f"cls0={format_optional_metric(val_metrics.get('severity_score_mean_cls0'))} "
                        f"cls1={format_optional_metric(val_metrics.get('severity_score_mean_cls1'))} "
                        f"cls2={format_optional_metric(val_metrics.get('severity_score_mean_cls2'))} "
                        f"cls3={format_optional_metric(val_metrics.get('severity_score_mean_cls3'))}"
                    )
                neighborhood_gate_outside = val_metrics.get("neighborhood_gate_outside")
                if neighborhood_gate_outside is not None and float(neighborhood_gate_outside) > 0.8:
                    print(
                        "WARNING: neighborhood gate is overly open "
                        f"outside={float(neighborhood_gate_outside):.4f}"
                    )

                state = make_checkpoint_state(
                    epoch=epoch + 1,
                    model=model,
                    ema_model=ema_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config=config,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    ema_val_metrics=ema_val_metrics,
                )
                save_checkpoint(checkpoints_dir / "last.pth", state)
                ema_state = None if ema_model is None else clone_checkpoint_with_model_weights(state, ema_model.module.state_dict())

                improved_flags = {
                    "macro_f1": metric_improved(val_metrics["macro_f1"], best_scores["macro_f1"], "max"),
                    "macro_f1_ema": bool(
                        ema_val_metrics is not None and metric_improved(ema_val_metrics["macro_f1"], best_scores["macro_f1_ema"], "max")
                    ),
                    "qwk": metric_improved(val_metrics["QWK"], best_scores["qwk"], "max"),
                    "far_error": metric_improved(val_metrics["far_error_rate"], best_scores["far_error"], "min"),
                    "composite_score": metric_improved(val_metrics["composite_score"], best_scores["composite_score"], "max"),
                }
                if improved_flags["macro_f1"] and bool(config["training"]["save_best_macro_f1"]):
                    best_scores["macro_f1"] = float(val_metrics["macro_f1"])
                    best_epochs["macro_f1"] = epoch + 1
                    save_checkpoint(checkpoints_dir / "best_macro_f1.pth", state)
                    save_checkpoint(checkpoints_dir / "best_macro_f1_raw.pth", state)
                if improved_flags["macro_f1_ema"] and ema_val_metrics is not None and ema_state is not None:
                    best_scores["macro_f1_ema"] = float(ema_val_metrics["macro_f1"])
                    best_epochs["macro_f1_ema"] = epoch + 1
                    save_checkpoint(checkpoints_dir / "best_macro_f1_ema.pth", ema_state)
                if improved_flags["qwk"] and bool(config["training"]["save_best_qwk"]):
                    best_scores["qwk"] = float(val_metrics["QWK"])
                    best_epochs["qwk"] = epoch + 1
                    save_checkpoint(checkpoints_dir / "best_qwk.pth", state)
                if improved_flags["far_error"] and bool(config["training"]["save_best_far_error"]):
                    best_scores["far_error"] = float(val_metrics["far_error_rate"])
                    best_epochs["far_error"] = epoch + 1
                    save_checkpoint(checkpoints_dir / "best_far_error.pth", state)
                if improved_flags["composite_score"]:
                    best_scores["composite_score"] = float(val_metrics["composite_score"])
                    best_epochs["composite_score"] = epoch + 1
                    save_checkpoint(checkpoints_dir / "best_composite_score.pth", state)

                monitor_metrics = (ema_val_metrics if (ema_val_metrics is not None and early_stop_metric == "macro_f1") else val_metrics)
                monitor_value = (
                    float(monitor_metrics["macro_f1"]) if early_stop_metric == "macro_f1" else float(monitor_metrics["composite_score"])
                )
                best_monitor_value = (
                    best_scores["macro_f1_ema"] if (early_stop_metric == "macro_f1" and ema_val_metrics is not None) else (
                        best_scores["macro_f1"] if early_stop_metric == "macro_f1" else best_scores["composite_score"]
                    )
                )
                if (
                    (early_stop_metric == "macro_f1" and ((improved_flags["macro_f1_ema"] and ema_val_metrics is not None) or improved_flags["macro_f1"]))
                    or (early_stop_metric == "composite_score" and improved_flags["composite_score"])
                ):
                    epochs_without_improvement = 0
                    final_best_state = state
                    if early_stop_metric == "macro_f1" and improved_flags["macro_f1_ema"] and ema_state is not None:
                        final_best_state = ema_state
                    save_checkpoint(checkpoints_dir / "final_best.pth", final_best_state)
                else:
                    epochs_without_improvement += 1

                epoch_record = {
                    "epoch": int(epoch + 1),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "lr_groups": [float(param_group["lr"]) for param_group in optimizer.param_groups],
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "ema_val_metrics": ema_val_metrics,
                    "raw_val_macro_f1": float(val_metrics["macro_f1"]),
                    "raw_val_qwk": float(val_metrics["QWK"]),
                    "raw_val_far_error": float(val_metrics["far_error_rate"]),
                    "ema_val_macro_f1": None if ema_val_metrics is None else float(ema_val_metrics["macro_f1"]),
                    "ema_val_qwk": None if ema_val_metrics is None else float(ema_val_metrics["QWK"]),
                    "ema_val_far_error": None if ema_val_metrics is None else float(ema_val_metrics["far_error_rate"]),
                    "neighborhood_branch_gate": val_metrics.get("neighborhood_branch_gate"),
                    "neighborhood_residual_scale": val_metrics.get("neighborhood_residual_scale"),
                    "minor_precision": val_metrics.get("minor_precision"),
                    "minor_recall": val_metrics.get("minor_recall"),
                    "minor_f1": val_metrics.get("minor_f1"),
                    "gt_minor_pred_no_rate": val_metrics.get("gt_minor_pred_no_rate"),
                    "gt_minor_pred_major_rate": val_metrics.get("gt_minor_pred_major_rate"),
                    "gt_major_pred_minor_rate": val_metrics.get("gt_major_pred_minor_rate"),
                    "best_scores": best_scores,
                    "best_epochs": best_epochs,
                    "early_stop_metric": early_stop_metric,
                    "monitor_value": monitor_value,
                    "best_monitor_value": best_monitor_value,
                    "epochs_without_improvement": int(epochs_without_improvement),
                    **collapse_diagnostics,
                }
                append_jsonl(history_path, epoch_record)

                if epochs_without_improvement >= patience:
                    print(
                        f"Early stopping triggered at epoch {epoch + 1}: "
                        f"no {early_stop_metric} improvement for {epochs_without_improvement} epochs."
                    )
                    break

            summary = {
                "run_dir": str(run_dir),
                "best_scores": best_scores,
                "best_epochs": best_epochs,
                "early_stop_metric": early_stop_metric,
                "history_path": str(history_path),
                "recommended_checkpoint": str((checkpoints_dir / "final_best.pth").resolve()),
            }
            write_json(run_dir / "training_summary.json", summary)
            write_text(
                run_dir / "training_summary.txt",
                "\n".join(
                    [
                        "Training Summary",
                        f"- run_dir: {summary['run_dir']}",
                        f"- early_stop_metric: {summary['early_stop_metric']}",
                        f"- best_macro_f1_epoch: {summary['best_epochs']['macro_f1']}",
                        f"- best_macro_f1_ema_epoch: {summary['best_epochs']['macro_f1_ema']}",
                        f"- best_qwk_epoch: {summary['best_epochs']['qwk']}",
                        f"- best_far_error_epoch: {summary['best_epochs']['far_error']}",
                        f"- best_composite_epoch: {summary['best_epochs']['composite_score']}",
                        f"- recommended_checkpoint: {summary['recommended_checkpoint']}",
                    ]
                )
                + "\n",
            )

            final_best = checkpoints_dir / "final_best.pth"
            if final_best.exists():
                shutil.copy2(final_best, checkpoints_dir / "best.pth")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
