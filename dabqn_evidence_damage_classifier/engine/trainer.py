from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import XBDQueryDataset, xbd_query_collate_fn
from engine.evaluator import evaluate_model, move_batch_to_device
from losses import DABQNLoss
from models import build_dabqn_model
from models.query.matcher import HungarianMatcher
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.ema import ModelEMA
from utils.logger import setup_logger
from utils.misc import append_jsonl, ensure_dir, load_yaml, write_json, write_yaml
from utils.scheduler import WarmupCosineScheduler
from utils.seed import seed_worker, set_seed

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    tqdm = None


def _autocast_context(device: torch.device, enabled: bool, dtype_name: str):
    if device.type != "cuda" or not enabled:
        return nullcontext()
    amp_dtype = torch.bfloat16 if dtype_name.lower() == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_loader(dataset: XBDQueryDataset, *, batch_size: int, num_workers: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=xbd_query_collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def _build_grad_scaler(*, config: dict[str, Any], device: torch.device):
    enabled = bool(config["training"].get("amp", True) and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _evaluation_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("evaluation", config.get("eval", {}))


def _progress(sequence, *, total: int, desc: str, enabled: bool):
    if not enabled or tqdm is None:
        return sequence
    return tqdm(sequence, total=total, desc=desc, dynamic_ncols=True)


def _build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    training_cfg = config["training"]
    backbone_lr = float(training_cfg.get("backbone_lr", training_cfg.get("lr", 2e-4)))
    new_module_lr = float(training_cfg.get("new_module_lr", training_cfg.get("lr", 2e-4)))
    weight_decay = float(training_cfg.get("weight_decay", 0.05))
    groups = {
        "backbone_decay": {"params": [], "lr": backbone_lr, "weight_decay": weight_decay},
        "backbone_no_decay": {"params": [], "lr": backbone_lr, "weight_decay": 0.0},
        "new_decay": {"params": [], "lr": new_module_lr, "weight_decay": weight_decay},
        "new_no_decay": {"params": [], "lr": new_module_lr, "weight_decay": 0.0},
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_backbone = name.startswith("backbone.")
        no_decay = parameter.ndim == 1 or name.endswith(".bias")
        group_name = (
            "backbone_no_decay"
            if is_backbone and no_decay
            else "backbone_decay"
            if is_backbone
            else "new_no_decay"
            if no_decay
            else "new_decay"
        )
        groups[group_name]["params"].append(parameter)
    optimizer_groups = [group for group in groups.values() if group["params"]]
    return torch.optim.AdamW(optimizer_groups)


def _set_trainable_parameters(model: torch.nn.Module, stage: str, config: dict[str, Any]) -> None:
    stage_name = str(stage)
    training_cfg = config["training"]
    freeze_backbone = bool(training_cfg.get("freeze_backbone", stage_name == "damage"))
    freeze_localizer = bool(training_cfg.get("freeze_localizer", stage_name == "damage"))
    for name, parameter in model.named_parameters():
        requires_grad = True
        if freeze_backbone and name.startswith("backbone."):
            requires_grad = False
        if freeze_localizer and any(
            name.startswith(prefix)
            for prefix in ("fpn.", "pixel_decoder.", "query_decoder.", "class_head.", "box_head.", "mask_head.")
        ):
            requires_grad = False
        if stage_name == "localization" and name.startswith("damage_branch."):
            requires_grad = False
        parameter.requires_grad_(requires_grad)


def _monitor_value(result: dict[str, Any], metric_name: str) -> float:
    name = str(metric_name)
    if name == "localization_f1":
        return float(result["localization"]["localization_f1"])
    if name == "damage_macro_f1":
        return float(result["end_to_end_damage"]["macro_f1"])
    if name == "matched_damage_macro_f1":
        return float(result["matched_damage"]["macro_f1"])
    if name == "bridge_score":
        return float(result["pixel_bridge"]["xview2_overall_score"])
    raise KeyError(f"Unsupported monitor metric '{metric_name}'.")


def train_one_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    matcher: HungarianMatcher,
    criterion: DABQNLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: dict[str, Any],
    stage: str,
    epoch: int,
    ema: ModelEMA | None,
    logger: Any | None = None,
) -> dict[str, float]:
    model.train()
    loss_sums: dict[str, float] = {}
    progress_enabled = bool(config["training"].get("progress_bar", True))
    patch_stats = {
        "patch_count": 0.0,
        "gt_counts": [],
        "gt_over_query_count": 0.0,
        "gt_over_max_instances_per_patch_count": 0.0,
        "adaptive_shrink_count": 0.0,
        "fallback_keep_count": 0.0,
        "empty_patch_count": 0.0,
        "positive_patch_count": 0.0,
    }
    scaler = _build_grad_scaler(config=config, device=device)
    log_interval = max(1, int(config["training"].get("log_interval", 50)))
    num_batches = max(len(loader), 1)
    epoch_start = time.perf_counter()
    if logger is not None:
        logger.info("epoch=%d waiting for first batch (%d total batches)", epoch + 1, num_batches)
    batch_iterator = _progress(
        enumerate(loader, start=1),
        total=num_batches,
        desc=f"Train {epoch + 1}",
        enabled=progress_enabled,
    )
    for batch_index, batch in batch_iterator:
        for meta in batch["meta"]:
            num_instances = float(meta.get("num_instances_after_limit", meta.get("num_instances", 0)))
            patch_stats["patch_count"] += 1.0
            patch_stats["gt_counts"].append(num_instances)
            patch_stats["gt_over_query_count"] += float(bool(meta.get("gt_over_query", False)))
            patch_stats["gt_over_max_instances_per_patch_count"] += float(bool(meta.get("gt_over_max_instances_per_patch", False)))
            patch_stats["adaptive_shrink_count"] += float(bool(meta.get("adaptive_shrink_applied", False)))
            patch_stats["fallback_keep_count"] += float(bool(meta.get("fallback_keep_applied", False)))
            patch_stats["empty_patch_count"] += float(bool(meta.get("is_empty_patch", False)))
            patch_stats["positive_patch_count"] += float(bool(meta.get("is_positive_patch", False)))
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, bool(config["training"].get("amp", True)), str(config["training"].get("amp_dtype", "bf16"))):
            outputs = model.forward_localization(batch)
            matches = matcher(outputs, batch["targets"])
            outputs = model.forward_damage(batch, outputs, targets=batch["targets"], matches=matches, epoch=epoch, stage=stage)
            loss_terms = criterion(outputs, batch["targets"], matches, stage=stage)
            loss = loss_terms["loss_total"]
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"].get("grad_clip_norm", 1.0)))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"].get("grad_clip_norm", 1.0)))
            optimizer.step()
        if ema is not None:
            ema.update(model)
        for key, value in loss_terms.items():
            loss_sums[key] = loss_sums.get(key, 0.0) + float(value.detach().item())
        if logger is not None and (batch_index == 1 or batch_index % log_interval == 0 or batch_index == num_batches):
            avg_loss = loss_sums.get("loss_total", 0.0) / float(batch_index)
            last_loss = float(loss_terms.get("loss_total", loss).detach().item())
            elapsed = time.perf_counter() - epoch_start
            gt_array = np.asarray(patch_stats["gt_counts"], dtype=np.float32) if patch_stats["gt_counts"] else np.zeros((0,), dtype=np.float32)
            avg_gt = float(gt_array.mean()) if gt_array.size > 0 else 0.0
            max_gt = float(gt_array.max()) if gt_array.size > 0 else 0.0
            if progress_enabled and tqdm is not None and hasattr(batch_iterator, "set_postfix"):
                batch_iterator.set_postfix(
                    loss=f"{last_loss:.4f}",
                    avg_loss=f"{avg_loss:.4f}",
                    avg_gt=f"{avg_gt:.1f}",
                    max_gt=f"{max_gt:.0f}",
                    fallback=f"{patch_stats['fallback_keep_count']:.0f}",
                )
            logger.info(
                "epoch=%d batch=%d/%d loss=%.4f avg_loss=%.4f avg_gt=%.1f max_gt=%.0f gt_over_query=%.0f gt_over_patch_cap=%.0f adaptive_shrink=%.0f fallback_keep=%.0f elapsed=%.1fs",
                epoch + 1,
                batch_index,
                num_batches,
                last_loss,
                avg_loss,
                avg_gt,
                max_gt,
                patch_stats["gt_over_query_count"],
                patch_stats["gt_over_max_instances_per_patch_count"],
                patch_stats["adaptive_shrink_count"],
                patch_stats["fallback_keep_count"],
                elapsed,
            )
    stats = {key: value / num_batches for key, value in loss_sums.items()}
    patch_count = max(patch_stats["patch_count"], 1.0)
    gt_array = np.asarray(patch_stats["gt_counts"], dtype=np.float32) if patch_stats["gt_counts"] else np.zeros((0,), dtype=np.float32)
    stats["avg_gt_per_patch"] = float(gt_array.mean()) if gt_array.size > 0 else 0.0
    stats["p50_gt_per_patch"] = float(np.percentile(gt_array, 50)) if gt_array.size > 0 else 0.0
    stats["p90_gt_per_patch"] = float(np.percentile(gt_array, 90)) if gt_array.size > 0 else 0.0
    stats["p95_gt_per_patch"] = float(np.percentile(gt_array, 95)) if gt_array.size > 0 else 0.0
    stats["max_gt_per_patch"] = float(gt_array.max()) if gt_array.size > 0 else 0.0
    stats["patch_gt_over_query_count"] = patch_stats["gt_over_query_count"]
    stats["patch_gt_over_max_instances_per_patch_count"] = patch_stats["gt_over_max_instances_per_patch_count"]
    stats["adaptive_shrink_count"] = patch_stats["adaptive_shrink_count"]
    stats["fallback_keep_count"] = patch_stats["fallback_keep_count"]
    stats["empty_patch_count"] = patch_stats["empty_patch_count"]
    stats["positive_patch_count"] = patch_stats["positive_patch_count"]
    stats["fallback_keep_ratio"] = patch_stats["fallback_keep_count"] / patch_count
    return stats


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    stage = str(config["training"].get("stage", config.get("stage", "joint")))
    eval_cfg = _evaluation_config(config)
    output_dir = ensure_dir(config["project"]["output_dir"])
    logger = setup_logger(output_dir)
    write_yaml(Path(output_dir) / "config.yaml", config)
    set_seed(int(config["training"].get("seed", 42)))

    logger.info("building datasets for stage=%s", stage)
    train_dataset = XBDQueryDataset(config=config, split="train", is_train=True)
    val_split = str(eval_cfg.get("split", "val"))
    val_dataset = XBDQueryDataset(config=config, split=val_split, is_train=False)
    logger.info("datasets ready train_samples=%d val_samples=%d", len(train_dataset), len(val_dataset))
    train_loader = _build_loader(
        train_dataset,
        batch_size=int(config["training"].get("batch_size", 2)),
        num_workers=int(config["training"].get("num_workers", 4)),
        shuffle=True,
        seed=int(config["training"].get("seed", 42)),
    )
    val_loader = _build_loader(
        val_dataset,
        batch_size=int(eval_cfg.get("batch_size", 2)),
        num_workers=int(eval_cfg.get("num_workers", 2)),
        shuffle=False,
        seed=int(config["training"].get("seed", 42)),
    )

    device = _resolve_device()
    logger.info(
        "building model on device=%s batch_size=%d train_workers=%d val_workers=%d amp=%s amp_dtype=%s",
        device,
        int(config["training"].get("batch_size", 2)),
        int(config["training"].get("num_workers", 4)),
        int(eval_cfg.get("num_workers", 2)),
        bool(config["training"].get("amp", True)),
        str(config["training"].get("amp_dtype", "bf16")),
    )
    model = build_dabqn_model(config).to(device)
    init_checkpoint_path = str(config["training"].get("init_checkpoint", "")).strip()
    if init_checkpoint_path:
        checkpoint = load_checkpoint(init_checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("ema_state_dict") or checkpoint.get("model_state_dict")
        if state_dict is None:
            raise RuntimeError(f"Checkpoint {init_checkpoint_path} does not contain model_state_dict or ema_state_dict.")
        incompatible = model.load_state_dict(state_dict, strict=False)
        logger.info(
            "loaded init checkpoint=%s missing=%s unexpected=%s",
            init_checkpoint_path,
            list(incompatible.missing_keys),
            list(incompatible.unexpected_keys),
        )
    _set_trainable_parameters(model, stage, config)
    optimizer = _build_optimizer(model, config)
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_epochs=int(config["training"].get("epochs", 20)),
        warmup_epochs=int(config["training"].get("warmup_epochs", 2)),
        min_lr_ratio=float(config["training"].get("min_lr_ratio", 0.01)),
    )
    matcher = HungarianMatcher(
        cost_objectness=float(config["matcher"].get("cost_objectness", 1.0)),
        cost_box_l1=float(config["matcher"].get("cost_box_l1", 5.0)),
        cost_giou=float(config["matcher"].get("cost_giou", 2.0)),
        cost_mask=float(config["matcher"].get("cost_mask", 2.0)),
        cost_dice=float(config["matcher"].get("cost_dice", 5.0)),
        add_damage_cost=bool(config["matcher"].get("add_damage_cost", False)),
        cost_damage=float(config["matcher"].get("cost_damage", 0.0)),
    )
    criterion = DABQNLoss(config).to(device)
    ema = ModelEMA(model, decay=float(config["training"].get("ema_decay", 0.999))) if bool(config["training"].get("ema_enabled", True)) else None

    best_metric_name = str(eval_cfg.get("save_best_by", "bridge_score"))
    best_metric = float("-inf")
    best_epoch = -1
    history_path = Path(output_dir) / "history.jsonl"
    if history_path.exists():
        history_path.unlink()

    for epoch in range(int(config["training"].get("epochs", 20))):
        logger.info("starting epoch=%d/%d", epoch + 1, int(config["training"].get("epochs", 20)))
        scheduler.step(epoch)
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            matcher=matcher,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            config=config,
            stage=stage,
            epoch=epoch,
            ema=ema,
            logger=logger,
        )
        eval_model = ema.module if ema is not None else model
        logger.info("running validation for epoch=%d", epoch + 1)
        val_result = evaluate_model(
            model=eval_model,
            loader=val_loader,
            matcher=matcher,
            criterion=criterion,
            device=device,
            config=config,
            stage=stage,
            epoch=epoch,
        )
        monitor_value = _monitor_value(val_result, best_metric_name)
        record = {
            "epoch": epoch + 1,
            "train": train_stats,
            "val": {
                "loss": val_result["loss"],
                "localization": val_result["localization"],
                "matched_damage": val_result["matched_damage"],
                "end_to_end_damage": val_result["end_to_end_damage"],
                "pixel_bridge": val_result["pixel_bridge"],
            },
            "monitor_metric": best_metric_name,
            "monitor_value": monitor_value,
        }
        append_jsonl(history_path, record)
        logger.info(
            "epoch=%d train_loss=%.4f avg_gt=%.1f p50_gt=%.1f p90_gt=%.1f p95_gt=%.1f max_gt=%.0f gt_over_query=%.0f gt_over_patch_cap=%.0f adaptive_shrink=%.0f fallback_keep=%.0f empty_patch=%.0f positive_patch=%.0f before_nms=%d after_nms=%d loc_f1=%.4f matched_macro_f1=%.4f e2e_macro_f1=%.4f bridge=%.4f",
            epoch + 1,
            float(train_stats.get("loss_total", 0.0)),
            float(train_stats.get("avg_gt_per_patch", 0.0)),
            float(train_stats.get("p50_gt_per_patch", 0.0)),
            float(train_stats.get("p90_gt_per_patch", 0.0)),
            float(train_stats.get("p95_gt_per_patch", 0.0)),
            float(train_stats.get("max_gt_per_patch", 0.0)),
            float(train_stats.get("patch_gt_over_query_count", 0.0)),
            float(train_stats.get("patch_gt_over_max_instances_per_patch_count", 0.0)),
            float(train_stats.get("adaptive_shrink_count", 0.0)),
            float(train_stats.get("fallback_keep_count", 0.0)),
            float(train_stats.get("empty_patch_count", 0.0)),
            float(train_stats.get("positive_patch_count", 0.0)),
            int(val_result.get("merge_stats", {}).get("predictions_before_nms", 0)),
            int(val_result.get("merge_stats", {}).get("predictions_after_nms", 0)),
            float(val_result["localization"]["localization_f1"]),
            float(val_result["matched_damage"]["macro_f1"]),
            float(val_result["end_to_end_damage"]["macro_f1"]),
            float(val_result["pixel_bridge"]["xview2_overall_score"]),
        )
        if float(train_stats.get("patch_gt_over_query_count", 0.0)) > 0.0:
            logger.warning("epoch=%d gt_over_num_queries_count=%.0f; matcher protection is being used.", epoch + 1, float(train_stats["patch_gt_over_query_count"]))
        if float(train_stats.get("fallback_keep_ratio", 0.0)) > 0.005:
            logger.warning("epoch=%d fallback_keep_ratio=%.4f exceeds target threshold 0.005.", epoch + 1, float(train_stats["fallback_keep_ratio"]))
        checkpoint = {
            "epoch": epoch + 1,
            "config": config,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": None if ema is None else ema.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_metric_name": best_metric_name,
            "best_metric": best_metric,
            "stage": stage,
        }
        save_checkpoint(Path(output_dir) / "checkpoints" / "last.pth", checkpoint)
        if monitor_value > best_metric:
            best_metric = monitor_value
            best_epoch = epoch + 1
            checkpoint["best_metric"] = best_metric
            save_checkpoint(Path(output_dir) / "checkpoints" / f"best_{best_metric_name}.pth", checkpoint)
            write_json(Path(output_dir) / "best_metrics.json", record)

    summary = {"best_metric_name": best_metric_name, "best_metric": best_metric, "best_epoch": best_epoch}
    write_json(Path(output_dir) / "training_summary.json", summary)
    return summary
