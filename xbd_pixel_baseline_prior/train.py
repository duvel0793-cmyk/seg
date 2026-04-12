from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.datasets.xbd_dataset import XBDDataset, format_data_check
from src.engine.evaluator import evaluate
from src.engine.trainer import train_one_epoch
from src.losses.damage_losses import WeightedFocalCrossEntropyLoss
from src.losses.segmentation_losses import BCEWithDiceLoss
from src.models.resnet18_fpn_bda import build_resnet18_fpn_bda
from src.utils.io import (
    count_parameters,
    dump_yaml,
    ensure_dir,
    pretty_yaml,
    resolve_device,
    save_json,
    load_yaml_config,
    strip_meta,
)
from src.utils.logger import setup_logger
from src.utils.metrics import CLASS_NAMES, format_confusion_matrix
from src.utils.seed import seed_worker, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the lightweight xBD pixel baseline.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline_no_prior.yaml",
        help="Path to the yaml config.",
    )
    return parser.parse_args()


def build_dataloader(dataset, batch_size: int, shuffle: bool, config: dict[str, Any], seed: int):
    num_workers = int(config["data"]["num_workers"])
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=bool(config["data"]["pin_memory"]),
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def create_grad_scaler(enabled: bool, device: torch.device):
    try:
        return torch.amp.GradScaler(device=device.type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def validate_config(config: dict[str, Any]) -> None:
    prior_mode = config["data"]["prior_mode"]
    in_channels = int(config["model"]["in_channels"])
    expected_channels = 7 if prior_mode == "input_channel" else 6
    if in_channels != expected_channels:
        raise ValueError(
            f"model.in_channels={in_channels} does not match prior_mode={prior_mode}. "
            f"Expected {expected_channels} input channels."
        )


def log_epoch_summary(logger, epoch: int, epochs: int, train_metrics: dict[str, Any], val_metrics: dict[str, Any]) -> None:
    logger.info(
        "Epoch %d/%d | train loss %.4f | val loss %.4f | pixel acc %.4f | macro F1 %.4f | loc F1 %.4f",
        epoch,
        epochs,
        train_metrics["loss"],
        float(val_metrics["loss"] or 0.0),
        float(val_metrics["overall_pixel_accuracy"]),
        float(val_metrics["damage_macro_f1"]),
        float(val_metrics["building_localization_f1"]),
    )
    logger.info(
        "Epoch %d/%d | F1 no-damage %.4f | minor %.4f | major %.4f | destroyed %.4f",
        epoch,
        epochs,
        float(val_metrics["no_damage_f1"]),
        float(val_metrics["minor_damage_f1"]),
        float(val_metrics["major_damage_f1"]),
        float(val_metrics["destroyed_f1"]),
    )


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    validate_config(config)

    output_dir = ensure_dir(config["output_dir"])
    logger = setup_logger(log_dir=output_dir)
    logger.info("Starting xBD pixel baseline training")
    logger.info("Resolved config:\n%s", pretty_yaml(strip_meta(config)))

    requested_device = str(config["device"])
    device = resolve_device(requested_device)
    if device.type != requested_device.split(":")[0]:
        logger.warning("Requested device %s is unavailable, falling back to %s", requested_device, device)

    dump_yaml(strip_meta(config), output_dir / "config_backup.yaml")
    set_seed(int(config["seed"]))

    label_stats_files = config["data"].get("label_stats_files", 128)
    train_dataset = XBDDataset(
        list_path=config["data"]["train_list"],
        image_dir=config["data"]["train_image_dir"],
        target_dir=config["data"]["target_dir"],
        prior_dir=config["data"]["prior_dir"],
        prior_mode=config["data"]["prior_mode"],
        crop_size=config["data"]["crop_size"],
        is_train=True,
        strict_data_check=bool(config["data"]["strict_data_check"]),
        prior_filename_pattern=config["data"].get("prior_filename_pattern", "{sample_id}.png"),
    )
    val_dataset = XBDDataset(
        list_path=config["data"]["val_list"],
        image_dir=config["data"]["val_image_dir"],
        target_dir=config["data"]["target_dir"],
        prior_dir=config["data"]["prior_dir"],
        prior_mode=config["data"]["prior_mode"],
        crop_size=None,
        is_train=False,
        strict_data_check=bool(config["data"]["strict_data_check"]),
        prior_filename_pattern=config["data"].get("prior_filename_pattern", "{sample_id}.png"),
    )
    logger.info("Train data check:\n%s", format_data_check(train_dataset.self_check(label_stats_files)))
    logger.info("Val data check:\n%s", format_data_check(val_dataset.self_check(label_stats_files)))

    train_loader = build_dataloader(
        train_dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=True,
        config=config,
        seed=int(config["seed"]),
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=int(config["eval"].get("batch_size", 1)),
        shuffle=False,
        config=config,
        seed=int(config["seed"]) + 1,
    )

    model = build_resnet18_fpn_bda(config["model"]).to(device)
    logger.info(
        "Model parameters: total=%d trainable=%d",
        count_parameters(model, trainable_only=False),
        count_parameters(model, trainable_only=True),
    )

    damage_criterion = WeightedFocalCrossEntropyLoss(
        class_weights=config["loss"]["class_weights"],
        gamma=float(config["loss"]["focal_gamma"]),
    )
    localization_criterion = BCEWithDiceLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )

    logging_cfg = dict(config.get("logging", {}))
    train_cfg = dict(config["train"])
    train_cfg["print_freq"] = logging_cfg.get("print_freq", 10)

    use_amp = bool(config["train"].get("amp", False) and device.type == "cuda")
    scaler = create_grad_scaler(enabled=use_amp, device=device)

    epochs = int(config["train"]["epochs"])
    val_interval = max(1, int(config["train"].get("val_interval", 1)))
    best_metric_name = str(config["train"].get("save_best_metric", "damage_macro_f1"))
    best_metric_value = float("-inf")
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            damage_criterion=damage_criterion,
            localization_criterion=localization_criterion,
            epoch_index=epoch,
            total_epochs=epochs,
            train_config=train_cfg,
            loss_config={
                "lambda_loc": float(config["loss"]["lambda_loc"]),
                "prior_mode": config["data"]["prior_mode"],
                "prior_alpha": float(config["loss"]["prior_alpha"]),
            },
            scaler=scaler,
        )

        val_metrics: dict[str, Any] = {
            "loss": None,
            "overall_pixel_accuracy": 0.0,
            "damage_macro_f1": 0.0,
            "building_localization_f1": 0.0,
            "no_damage_f1": 0.0,
            "minor_damage_f1": 0.0,
            "major_damage_f1": 0.0,
            "destroyed_f1": 0.0,
            "confusion_matrix": [[0] * len(CLASS_NAMES) for _ in range(len(CLASS_NAMES))],
            "per_class": {},
        }

        if epoch % val_interval == 0:
            val_metrics = evaluate(
                model=model,
                dataloader=val_loader,
                device=device,
                damage_criterion=damage_criterion,
                localization_criterion=localization_criterion,
                lambda_loc=float(config["loss"]["lambda_loc"]),
                prior_mode=config["data"]["prior_mode"],
                prior_alpha=float(config["loss"]["prior_alpha"]),
                ignore_background_in_macro=bool(config["eval"]["ignore_background_in_macro"]),
                amp=bool(config["train"]["amp"]),
                save_pred_masks=False,
                save_overlay_images=False,
                output_dir=None,
                desc=f"Val {epoch}/{epochs}",
            )

        log_epoch_summary(logger, epoch, epochs, train_metrics, val_metrics)
        logger.info("Val confusion matrix:\n%s", format_confusion_matrix(val_metrics["confusion_matrix"]))

        current_metric = float(val_metrics.get(best_metric_name, 0.0) or 0.0)
        is_best = current_metric >= best_metric_value
        if is_best:
            best_metric_value = current_metric

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": strip_meta(config),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "best_metric_name": best_metric_name,
            "best_metric_value": best_metric_value,
        }
        torch.save(checkpoint, output_dir / "latest.pth")

        if is_best:
            torch.save(checkpoint, output_dir / "best_macro_f1.pth")

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
        )
        save_json(
            {
                "best_metric_name": best_metric_name,
                "best_metric_value": best_metric_value,
                "history": history,
            },
            output_dir / "metrics.json",
        )

    logger.info("Training finished. Best %s=%.4f", best_metric_name, best_metric_value)


if __name__ == "__main__":
    main()
