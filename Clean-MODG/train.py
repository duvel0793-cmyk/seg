"""Training entrypoint for Clean-MODG."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.xbd_dataset import XBDInstanceDataset
from losses import build_loss
from metrics.bridge_score import compute_bridge_score
from metrics.classification import accuracy_score_np, classification_report_dict, macro_f1_score_np, per_class_f1_score_np
from metrics.confusion import confusion_matrix, save_confusion_matrix_png
from metrics.ordinal import ordinal_mae, ordinal_rmse, severe_error_rate
from models import build_model
from utils.checkpoint import auto_resume_path, load_checkpoint, save_checkpoint
from utils.common import AverageMeter, CLASS_NAMES, ensure_dir, flatten_metrics_for_csv, format_metrics_line, move_batch_to_device, outputs_to_predictions, outputs_to_probs, save_predictions_csv
from utils.config import load_config, save_config
from utils.distributed import get_device
from utils.logger import setup_logger
from utils.seed import set_seed


def autocast_context(device: torch.device, enabled: bool):
    return torch.amp.autocast(device_type=device.type, enabled=enabled)


def build_dataloaders(config: Dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    manifest_path = Path(data_cfg["manifest_path"])
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest file was not found: {manifest_path}. "
            "Create it first with data/build_xbd_manifest.py or point config.data.manifest_path to an existing CSV."
        )
    train_split = data_cfg.get("split_train", "train")
    val_split = data_cfg.get("split_val", "val")
    if bool(data_cfg.get("eval_on_train_subset", False)):
        val_split = train_split
    train_dataset = XBDInstanceDataset(data_cfg=data_cfg, split=train_split)
    val_dataset = XBDInstanceDataset(data_cfg=data_cfg, split=val_split)
    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 4))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, val_loader


def build_eval_loader(config: Dict[str, Any], split: str) -> DataLoader:
    data_cfg = config["data"]
    dataset = XBDInstanceDataset(data_cfg=data_cfg, split=split)
    return DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def build_optimizer(model: torch.nn.Module, train_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    name = str(train_cfg.get("optimizer", "adamw")).lower()
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.05))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, train_cfg: Dict[str, Any]):
    name = str(train_cfg.get("scheduler", "cosine")).lower()
    epochs = int(train_cfg.get("epochs", 100))
    warmup_epochs = int(train_cfg.get("warmup_epochs", 0))
    if name == "cosine":
        def lr_lambda(epoch: int) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return float(epoch + 1) / float(max(warmup_epochs, 1))
            progress = float(epoch - warmup_epochs) / float(max(epochs - warmup_epochs, 1))
            return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 3, 1), gamma=0.1)
    raise ValueError(f"Unsupported scheduler: {name}")


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool,
    grad_clip: float | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> Dict[str, float]:
    model.train()
    meters = {
        "loss": AverageMeter(),
    }
    progress = tqdm(loader, desc="train", leave=False)
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, enabled=use_amp):
            outputs = model(batch)
            loss_dict = criterion(outputs, batch)
            loss = loss_dict["loss"]
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        meters["loss"].update(float(loss.item()), n=batch["label"].shape[0])
        progress.set_postfix({"loss": f"{meters['loss'].avg:.4f}"})
    return {"loss": meters["loss"].avg}


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    use_amp: bool = False,
) -> tuple[Dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    loss_meter = AverageMeter()
    all_preds: list[int] = []
    all_targets: list[int] = []
    all_areas: list[float] = []
    prediction_rows: list[dict[str, Any]] = []

    progress = tqdm(loader, desc="eval", leave=False)
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        with autocast_context(device, enabled=use_amp):
            outputs = model(batch)
            loss_dict = criterion(outputs, batch)
        probs = outputs_to_probs(outputs).detach().cpu()
        preds = outputs_to_predictions(outputs).detach().cpu()
        targets = batch["label"].detach().cpu()
        areas = batch["area"].detach().cpu().tolist() if isinstance(batch["area"], torch.Tensor) else list(batch["area"])

        loss_meter.update(float(loss_dict["loss"].item()), n=targets.shape[0])
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.tolist())
        all_areas.extend([float(x) for x in areas])

        for idx in range(targets.shape[0]):
            prediction_rows.append(
                {
                    "building_id": batch["building_id"][idx],
                    "disaster_id": batch["disaster_id"][idx],
                    "target": int(targets[idx].item()),
                    "pred": int(preds[idx].item()),
                    "prob_no": float(probs[idx, 0].item()),
                    "prob_minor": float(probs[idx, 1].item()),
                    "prob_major": float(probs[idx, 2].item()),
                    "prob_destroyed": float(probs[idx, 3].item()),
                    "area": float(areas[idx]),
                }
            )
        progress.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

    metrics = {
        "loss": float(loss_meter.avg),
        "accuracy": accuracy_score_np(all_preds, all_targets),
        "macro_f1": macro_f1_score_np(all_preds, all_targets),
        "per_class_f1": per_class_f1_score_np(all_preds, all_targets),
        "ordinal_mae": ordinal_mae(all_preds, all_targets),
        "ordinal_rmse": ordinal_rmse(all_preds, all_targets),
        "severe_error_rate": severe_error_rate(all_preds, all_targets),
        "bridge_score": compute_bridge_score(all_preds, all_targets, areas=all_areas),
        "classification_report": classification_report_dict(all_preds, all_targets),
        "confusion_matrix": confusion_matrix(all_preds, all_targets),
        "preds": all_preds,
        "targets": all_targets,
    }
    return metrics, prediction_rows


def append_metrics_csv(path: str | Path, row: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = flatten_metrics_for_csv(row)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(flat.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(flat)


def prepare_output_dirs(output_dir: str | Path) -> Dict[str, Path]:
    output_dir = ensure_dir(output_dir)
    return {
        "root": output_dir,
        "checkpoints": ensure_dir(output_dir / "checkpoints"),
        "confusion": ensure_dir(output_dir / "confusion"),
        "predictions": ensure_dir(output_dir / "predictions"),
    }


def run_training(config: Dict[str, Any], resume_path: str | None = None) -> None:
    output_dir = config["project"]["output_dir"]
    dirs = prepare_output_dirs(output_dir)
    save_config(config, dirs["root"] / "config.yaml")
    logger = setup_logger(dirs["root"])
    set_seed(int(config["train"].get("seed", 42)))

    device = get_device()
    logger.info("Using device: %s", device)
    train_loader, val_loader = build_dataloaders(config)
    model = build_model(config).to(device)
    criterion = build_loss(config)
    optimizer = build_optimizer(model, config["train"])
    scheduler = build_scheduler(optimizer, config["train"])
    use_amp = bool(config["train"].get("amp", False) and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    start_epoch = 0
    best_metrics = {"macro_f1": -1.0, "bridge_score": -1.0}
    if resume_path:
        checkpoint = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_metrics.update(checkpoint.get("best_metrics", {}))
        logger.info("Resumed from %s at epoch %d", resume_path, start_epoch)

    epochs = int(config["train"].get("epochs", 100))
    grad_clip = config["train"].get("grad_clip", None)
    save_every = int(config["train"].get("save_every", 10))
    patience = int(config["train"].get("early_stop_patience", 20))
    main_metric = str(config["eval"].get("main_metric", "macro_f1"))
    epochs_without_improve = 0

    for epoch in range(start_epoch, epochs):
        logger.info("Epoch %d / %d", epoch + 1, epochs)
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            use_amp=use_amp,
            grad_clip=grad_clip,
            scaler=scaler,
        )
        val_metrics, val_predictions = evaluate_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )
        scheduler.step()

        logger.info("Train | %s", format_metrics_line(train_metrics))
        logger.info("Val   | %s", format_metrics_line(val_metrics))
        logger.info("Val per-class F1: %s", [round(x, 4) for x in val_metrics["per_class_f1"]])

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "accuracy": val_metrics["accuracy"],
            "macro_f1": val_metrics["macro_f1"],
            "ordinal_mae": val_metrics["ordinal_mae"],
            "ordinal_rmse": val_metrics["ordinal_rmse"],
            "severe_error_rate": val_metrics["severe_error_rate"],
            "bridge_score": val_metrics["bridge_score"],
            "per_class_f1": val_metrics["per_class_f1"],
        }
        append_metrics_csv(dirs["root"] / "metrics.csv", row)

        save_predictions_csv(val_predictions, dirs["predictions"] / "val_predictions.csv")
        if bool(config["eval"].get("save_confusion", True)):
            save_confusion_matrix_png(
                val_metrics["confusion_matrix"],
                dirs["confusion"] / f"epoch_{epoch:03d}.png",
                class_names=CLASS_NAMES,
            )

        improved = val_metrics[main_metric] > best_metrics.get(main_metric, -1.0)
        if improved:
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if val_metrics["macro_f1"] > best_metrics.get("macro_f1", -1.0):
            best_metrics["macro_f1"] = val_metrics["macro_f1"]
            save_checkpoint(
                dirs["checkpoints"] / "best_macro_f1.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_metrics=best_metrics,
                config=config,
            )
        if val_metrics["bridge_score"] > best_metrics.get("bridge_score", -1.0):
            best_metrics["bridge_score"] = val_metrics["bridge_score"]
            save_checkpoint(
                dirs["checkpoints"] / "best_bridge_score.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_metrics=best_metrics,
                config=config,
            )

        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                dirs["checkpoints"] / f"epoch_{epoch:03d}.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_metrics=best_metrics,
                config=config,
            )

        save_checkpoint(
            dirs["checkpoints"] / "last.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_metrics=best_metrics,
            config=config,
        )

        if epochs_without_improve >= patience:
            logger.info("Early stopping triggered after %d epochs without improvement.", epochs_without_improve)
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Clean-MODG models from YAML config.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint path to resume from.")
    parser.add_argument("--auto-resume", action="store_true", help="Resume from outputs/.../checkpoints/last.pth if it exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    resume_path = args.resume
    if resume_path is None and args.auto_resume:
        resume = auto_resume_path(config["project"]["output_dir"])
        resume_path = str(resume) if resume is not None else None
    try:
        run_training(config=config, resume_path=resume_path)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
