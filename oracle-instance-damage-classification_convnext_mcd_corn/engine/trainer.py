from __future__ import annotations

import copy
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import WeightedRandomSampler
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from datasets.label_mapping import CLASS_NAMES
from engine.evaluator import evaluate_model, move_batch_to_device
from utils.checkpoint import save_checkpoint, save_json
from utils.config import dump_config


def build_weighted_sampler(dataset, config: dict[str, Any]) -> tuple[WeightedRandomSampler | None, dict[str, float]]:
    train_cfg = config["train"]
    if not bool(train_cfg.get("use_weighted_sampler", False)):
        return None, {name: 1.0 for name in CLASS_NAMES}

    counts = torch.tensor(dataset.class_counts, dtype=torch.float32).clamp_min(1.0)
    power = float(train_cfg.get("sampler_power", 0.1))
    inv = counts.max() / counts
    class_weights = inv.pow(power)
    scale_cfg = train_cfg.get("class_sample_weight_scale", {})
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_weights[class_idx] *= float(scale_cfg.get(class_name, 1.0))
    sample_weights = torch.tensor([float(class_weights[int(sample["label"])].item()) for sample in dataset.samples], dtype=torch.double)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler, {name: float(class_weights[idx].item()) for idx, name in enumerate(CLASS_NAMES)}


def build_ce_class_weights(dataset) -> torch.Tensor:
    counts = torch.tensor(dataset.class_counts, dtype=torch.float32).clamp_min(1.0)
    weights = counts.sum() / counts
    return weights / weights.mean()


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    train_cfg = config["train"]
    base_lr = float(train_cfg["lr"])
    backbone_lr_scale = float(train_cfg.get("backbone_lr_scale", 0.1))
    head_lr_scale = float(train_cfg.get("head_lr_scale", 1.0))
    weight_decay = float(train_cfg.get("weight_decay", 0.05))

    backbone_params = model.get_backbone_parameters()
    backbone_param_ids = {id(param) for param in backbone_params}
    head_params = [param for param in model.parameters() if param.requires_grad and id(param) not in backbone_param_ids]
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": base_lr * backbone_lr_scale, "weight_decay": weight_decay})
    if head_params:
        param_groups.append({"params": head_params, "lr": base_lr * head_lr_scale, "weight_decay": weight_decay})
    return torch.optim.AdamW(param_groups)


def set_tau_head_trainable(model: torch.nn.Module, trainable: bool) -> None:
    for parameter in model.tau_head.parameters():
        parameter.requires_grad = bool(trainable)


def _summarize_scalar_tensors(loss_dict: dict[str, Any], batch_size: int, running: dict[str, float]) -> None:
    for key, value in loss_dict.items():
        if key == "loss":
            scalar = float(value.detach().cpu().item())
        elif torch.is_tensor(value) and value.ndim == 0:
            scalar = float(value.detach().cpu().item())
        else:
            continue
        running[key] = running.get(key, 0.0) + (scalar * batch_size)


def _format_seconds(seconds: float) -> str:
    total_seconds = int(max(seconds, 0.0))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _build_progress(iterable, *, enabled: bool, desc: str, total: int | None = None):
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, leave=False, dynamic_ncols=True)


def _flatten_metrics(prefix: str, payload: dict[str, Any], flat: dict[str, Any]) -> None:
    for key, value in payload.items():
        current_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            _flatten_metrics(current_key, value, flat)
        elif isinstance(value, (list, tuple)):
            flat[current_key] = json.dumps(value, ensure_ascii=False)
        else:
            flat[current_key] = value


def _write_metrics_csv(history: list[dict[str, Any]], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    fieldnames: set[str] = set()
    for record in history:
        flat: dict[str, Any] = {}
        _flatten_metrics("", record, flat)
        rows.append(flat)
        fieldnames.update(flat.keys())
    ordered_fieldnames = sorted(fieldnames)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    criterion,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    config: dict[str, Any],
    *,
    show_progress: bool = True,
) -> dict[str, float]:
    model.train()
    running: dict[str, float] = {}
    total_samples = 0
    grad_clip = float(config["train"].get("grad_clip", 5.0))
    use_amp = bool(config["train"].get("amp", True) and device.type == "cuda")

    freeze_tau_epochs = int(config["model"].get("freeze_tau_epochs", 0))
    set_tau_head_trainable(model, trainable=epoch >= freeze_tau_epochs)

    start_time = time.perf_counter()
    num_batches = 0
    progress = _build_progress(
        loader,
        enabled=show_progress,
        desc=f"Train {epoch + 1}",
        total=len(loader) if hasattr(loader, "__len__") else None,
    )
    for num_batches, batch in enumerate(progress, start=1):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(batch["pre_image"], batch["post_image"], batch["instance_mask"])
            loss_dict = criterion(outputs, batch["label"])
            loss = loss_dict["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        batch_size = int(batch["label"].shape[0])
        total_samples += batch_size
        _summarize_scalar_tensors(loss_dict, batch_size, running)
        if show_progress and tqdm is not None:
            progress.set_postfix(
                loss=f"{float(loss.detach().cpu().item()):.4f}",
                ce=f"{float(loss_dict['loss_ce'].detach().cpu().item()):.4f}",
                corn=f"{float(loss_dict['loss_corn'].detach().cpu().item()):.4f}",
                tau=f"{float(outputs['tau'].detach().mean().cpu().item()):.3f}",
            )

    summary = {key: value / max(total_samples, 1) for key, value in running.items()}
    elapsed_seconds = time.perf_counter() - start_time
    summary["epoch_seconds"] = elapsed_seconds
    summary["samples_per_second"] = float(total_samples / max(elapsed_seconds, 1e-6))
    summary["num_samples"] = float(total_samples)
    summary["num_batches"] = float(num_batches)
    return summary


def _select_metric(report: dict[str, Any], metric_name: str) -> float:
    if metric_name.startswith("raw_"):
        metric_key = metric_name[len("raw_") :]
        return float(report["raw_metrics"][metric_key])
    if metric_name.startswith("calibrated_"):
        metric_key = metric_name[len("calibrated_") :]
        calibrated = report["calibrated_metrics"] or report["raw_metrics"]
        return float(calibrated[metric_key])
    if metric_name in report["avg_losses"]:
        return -float(report["avg_losses"][metric_name])
    raise KeyError(f"Unsupported metric selector: {metric_name}")


def fit(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    criterion,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    config: dict[str, Any],
    output_dir: str | Path,
    logger,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_config(copy.deepcopy(config), output_dir / "resolved_config.yaml")

    scaler = torch.amp.GradScaler(device.type, enabled=bool(config["train"].get("amp", True) and device.type == "cuda"))
    history: list[dict[str, Any]] = []
    best_metric_name = str(config["train"].get("save_best_metric", "calibrated_macro_f1"))
    best_metric_value = float("-inf")
    best_thresholds: list[float] | None = None
    best_epoch = -1
    patience = int(config["train"].get("early_stopping_patience", 8))
    stale_epochs = 0
    train_start_time = time.perf_counter()
    epoch_durations: list[float] = []
    metrics_dir = output_dir / "epoch_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(int(config["train"]["epochs"])):
        epoch_start_time = time.perf_counter()
        train_summary = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            config=config,
            show_progress=True,
        )
        val_report = evaluate_model(
            model=model,
            loader=val_loader,
            device=device,
            criterion=criterion,
            amp=bool(config["train"].get("amp", True)),
            threshold_candidates=list(config["model"].get("threshold_candidates", [0.5])),
            fit_thresholds=bool(config["eval"].get("use_threshold_calibration", True)),
            preset_thresholds=None,
            show_progress=True,
            progress_desc=f"Val {epoch + 1}",
        )
        if scheduler is not None:
            scheduler.step()
        current_lr = max(group["lr"] for group in optimizer.param_groups)

        selected_metric = _select_metric(val_report, best_metric_name)
        if selected_metric > best_metric_value:
            best_metric_value = selected_metric
            best_thresholds = list(val_report["best_thresholds"]) if val_report["best_thresholds"] is not None else None
            best_epoch = epoch
            stale_epochs = 0
            is_best = True
        else:
            stale_epochs += 1
            is_best = False

        epoch_seconds = time.perf_counter() - epoch_start_time
        epoch_durations.append(epoch_seconds)
        elapsed_seconds = time.perf_counter() - train_start_time
        remaining_epochs = max(int(config["train"]["epochs"]) - (epoch + 1), 0)
        avg_epoch_seconds = sum(epoch_durations) / max(len(epoch_durations), 1)
        eta_seconds = remaining_epochs * avg_epoch_seconds

        epoch_record = {
            "epoch": epoch + 1,
            "lr": current_lr,
            "train": train_summary,
            "val_raw_metrics": val_report["raw_metrics"],
            "val_calibrated_metrics": val_report["calibrated_metrics"],
            "val_avg_losses": val_report["avg_losses"],
            "val_timing": val_report.get("timing", {}),
            "best_thresholds": val_report["best_thresholds"],
            "selected_metric": selected_metric,
            "timing": {
                "epoch_seconds": epoch_seconds,
                "elapsed_seconds": elapsed_seconds,
                "eta_seconds": eta_seconds,
                "avg_epoch_seconds": avg_epoch_seconds,
                "epoch_hms": _format_seconds(epoch_seconds),
                "elapsed_hms": _format_seconds(elapsed_seconds),
                "eta_hms": _format_seconds(eta_seconds),
            },
        }
        history.append(epoch_record)
        save_json(history, output_dir / "metrics_history.json")
        _write_metrics_csv(history, output_dir / "metrics_history.csv")
        save_json(epoch_record, metrics_dir / f"epoch_{epoch + 1:03d}.json")
        save_json(epoch_record, output_dir / "latest_metrics.json")
        save_json(
            {
                "epoch": epoch + 1,
                "best_thresholds": val_report["best_thresholds"],
                "threshold_search": val_report["threshold_search"],
            },
            output_dir / "latest_thresholds.json",
        )

        checkpoint_payload = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
            "config": copy.deepcopy(config),
            "best_thresholds": best_thresholds,
            "latest_thresholds": val_report["best_thresholds"],
            "best_metric_name": best_metric_name,
            "best_metric_value": best_metric_value,
            "history": history,
            "pretrained_report": model.get_pretrained_report(),
        }
        save_checkpoint(checkpoint_payload, output_dir / "latest.pth")
        if is_best:
            save_checkpoint(checkpoint_payload, output_dir / "best.pth")
            save_json(epoch_record, output_dir / "best_metrics.json")
            save_json(
                {
                    "best_epoch": epoch + 1,
                    "best_metric_name": best_metric_name,
                    "best_metric_value": best_metric_value,
                    "best_thresholds": best_thresholds,
                },
                output_dir / "best_thresholds.json",
            )

        raw_macro_f1 = float(val_report["raw_metrics"]["macro_f1"])
        calibrated_macro_f1 = float((val_report["calibrated_metrics"] or val_report["raw_metrics"])["macro_f1"])
        raw_per_class = {name: round(stats["f1"], 4) for name, stats in val_report["raw_metrics"]["per_class"].items()}
        calibrated_payload = val_report["calibrated_metrics"] or val_report["raw_metrics"]
        calibrated_per_class = {name: round(stats["f1"], 4) for name, stats in calibrated_payload["per_class"].items()}
        logger.info(
            "Epoch %d/%d | train_loss=%.4f | loss_ce=%.4f | loss_corn=%.4f | loss_tau_reg=%.4f | lr=%.6g",
            epoch + 1,
            int(config["train"]["epochs"]),
            float(train_summary.get("loss", 0.0)),
            float(train_summary.get("loss_ce", 0.0)),
            float(train_summary.get("loss_corn", 0.0)),
            float(train_summary.get("loss_tau_reg", 0.0)),
            current_lr,
        )
        logger.info(
            "Epoch %d | raw_macro_f1=%.4f | calibrated_macro_f1=%.4f | best_thresholds=%s",
            epoch + 1,
            raw_macro_f1,
            calibrated_macro_f1,
            val_report["best_thresholds"],
        )
        logger.info("Epoch %d | raw_per_class_f1=%s", epoch + 1, raw_per_class)
        logger.info("Epoch %d | calibrated_per_class_f1=%s", epoch + 1, calibrated_per_class)
        logger.info(
            "Epoch %d | epoch_time=%s | elapsed=%s | eta=%s | train_samples_per_sec=%.2f | val_samples_per_sec=%.2f",
            epoch + 1,
            epoch_record["timing"]["epoch_hms"],
            epoch_record["timing"]["elapsed_hms"],
            epoch_record["timing"]["eta_hms"],
            float(train_summary.get("samples_per_second", 0.0)),
            float(val_report.get("timing", {}).get("samples_per_second", 0.0)),
        )

        if stale_epochs >= patience:
            logger.info("Early stopping triggered at epoch %d.", epoch + 1)
            break

    total_elapsed_seconds = time.perf_counter() - train_start_time
    return {
        "best_epoch": best_epoch + 1,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
        "best_thresholds": best_thresholds,
        "history": history,
        "output_dir": str(output_dir),
        "total_elapsed_seconds": total_elapsed_seconds,
        "total_elapsed_hms": _format_seconds(total_elapsed_seconds),
    }
