"""Train oracle instance-level xBD damage classification."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from datasets.build import build_dataloader
from engine.evaluator import Evaluator
from engine.trainer import Trainer
from models.build import build_model
from utils.checkpoint import save_checkpoint
from utils.config import load_config, save_config
from utils.logger import setup_logger
from utils.misc import count_parameters, make_run_dir
from utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ConvNeXt tau+CORN on oracle xBD instances.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to merged root config.")
    parser.add_argument("--run-name", default="", help="Optional run name suffix.")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config["runtime"]["seed"])

    run_name = args.run_name or config.get("experiment", {}).get("name", "oracle_instance_damage")
    run_dir = make_run_dir(config["runtime"]["output_dir"], run_name=run_name)
    logger = setup_logger("train", output_dir=run_dir, filename="train.log")
    save_config(config, Path(run_dir) / "merged_config.yaml")

    logger.info(
        "Model config pretrained_path=%s pretrained_url=%s auto_download_pretrained=%s "
        "load_pretrained=%s freeze_backbone=%s tau_mode=%s runtime.use_weighted_sampler=%s",
        config["model"].get("pretrained_path", ""),
        config["model"].get("pretrained_url", ""),
        config["model"].get("auto_download_pretrained", False),
        config["model"].get("load_pretrained", False),
        config["model"].get("freeze_backbone", False),
        config["model"].get("tau_mode", "per_threshold"),
        config["runtime"].get("use_weighted_sampler", False),
    )

    device = resolve_device(config["runtime"].get("device", "cuda"))
    train_loader = build_dataloader(config, split="train", shuffle=None, logger=logger)
    val_loader = build_dataloader(config, split="val", shuffle=False, logger=logger)

    model = build_model(config).to(device)
    logger.info("Backbone pretrained report: %s", model.get_backbone_pretrained_report())
    logger.info("Model parameters: %.2fM", count_parameters(model) / 1e6)
    logger.info("Train samples: %d | Val samples: %d", len(train_loader.dataset), len(val_loader.dataset))
    logger.info("Using device: %s", device)
    logger.info(
        "Train class counts: %s | weighted_sampler=%s | sampler_power=%s | weight_preview=%s",
        getattr(train_loader, "class_count_summary", {}),
        getattr(train_loader, "weighted_sampler_enabled", False),
        getattr(train_loader, "sampler_power", None),
        getattr(train_loader, "sample_weight_preview", []),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["runtime"]["lr"],
        weight_decay=config["runtime"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["runtime"].get("scheduler_t_max", config["runtime"]["epochs"]),
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        logger=logger,
    )
    evaluator = Evaluator(model=model, config=config, device=device, logger=logger)

    best_macro_f1 = -1.0
    epochs = config["runtime"]["epochs"]
    for epoch in range(1, epochs + 1):
        train_metrics = trainer.train_one_epoch(train_loader, epoch)
        val_metrics = evaluator.evaluate(val_loader, split="val")

        logger.info(
            "Epoch [%d/%d] train_loss=%.4f train_corn_loss=%.4f train_tau_center=%.6f train_tau_diff=%.6f "
            "train_acc=%.4f train_macro_f1=%.4f train_weighted_f1=%.4f train_tau_clamped=%s",
            epoch,
            epochs,
            train_metrics["loss"],
            train_metrics["corn_loss"],
            train_metrics["tau_center_reg"],
            train_metrics["tau_diff_reg"],
            train_metrics["accuracy"],
            train_metrics["macro_f1"],
            train_metrics["weighted_f1"],
            train_metrics["tau_clamped"],
        )
        logger.info(
            "Epoch [%d/%d] val_loss=%.4f val_corn_loss=%.4f val_tau_center=%.6f val_tau_diff=%.6f "
            "val_acc=%.4f val_macro_f1=%.4f val_weighted_f1=%.4f val_tau_clamped=%s",
            epoch,
            epochs,
            val_metrics["loss"],
            val_metrics["corn_loss"],
            val_metrics["tau_center_reg"],
            val_metrics["tau_diff_reg"],
            val_metrics["accuracy"],
            val_metrics["macro_f1"],
            val_metrics["weighted_f1"],
            val_metrics["tau_clamped"],
        )

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": config,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        save_checkpoint(checkpoint, Path(run_dir) / "latest.pth")

        if epoch % config["runtime"]["save_freq"] == 0:
            save_checkpoint(checkpoint, Path(run_dir) / f"epoch_{epoch:03d}.pth")

        if val_metrics["macro_f1"] >= best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            save_checkpoint(checkpoint, Path(run_dir) / "best.pth")

    logger.info("Training finished. Best val macro_f1=%.4f", best_macro_f1)


if __name__ == "__main__":
    main()
