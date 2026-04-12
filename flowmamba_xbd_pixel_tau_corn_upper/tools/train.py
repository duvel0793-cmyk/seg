#!/usr/bin/env python
"""Single-card training entry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "exp_fulltrain_vmamba.yaml"

from src.data import build_dataloader
from src.engine import Trainer
from src.losses import build_criterion
from src.models import VMambaTauCornUpper
from src.utils import (
    build_run_identity,
    dump_config,
    ensure_dir,
    load_config,
    log_kv,
    resolve_device,
    seed_everything,
    setup_logger,
    summarize_backend_metadata,
)


def build_scheduler(cfg, optimizer):
    if cfg["scheduler"]["name"].lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(cfg["train"]["epochs"]), 1),
            eta_min=float(cfg["scheduler"].get("min_lr", 1.0e-6)),
        )
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the standalone VMamba xBD pixel tau CORN upper project.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))

    output_dir = ensure_dir(cfg["output_dir"])
    log_dir = ensure_dir(output_dir / "logs")
    resolved_config_path = log_dir / "resolved_config.yaml"
    dump_config(cfg, resolved_config_path)
    logger = setup_logger("train", log_dir / "train.log")
    logger.info("======== Train startup ========")
    logger.info("config=%s", Path(args.config).expanduser().resolve())
    logger.info("resolved_config_path=%s", resolved_config_path)
    logger.info("output_dir=%s", output_dir)
    logger.info("resolving device and building dataloaders/model ...")
    log_kv(logger, "resolved_config", cfg)

    device = resolve_device(cfg["device"])
    logger.info("device=%s", device)

    logger.info("building train dataloader ...")
    train_dataset, train_loader = build_dataloader(cfg, mode="train")

    logger.info("building val dataloader ...")
    val_dataset, val_loader = build_dataloader(cfg, mode="val")

    logger.info("building model ...")
    model = VMambaTauCornUpper(cfg).to(device)
    log_kv(logger, "run_identity", build_run_identity(cfg, model, resolved_config_path=resolved_config_path))

    logger.info("building criterion / optimizer / scheduler ...")
    criterion = build_criterion(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scheduler = build_scheduler(cfg, optimizer)

    logger.info("experiment=%s device=%s", cfg.get("experiment", {}).get("name", "unnamed"), device)
    logger.info("train samples=%d val samples=%d", len(train_dataset), len(val_dataset))
    log_kv(logger, "backbone", summarize_backend_metadata(model.backbone.get_metadata()))

    trainer = Trainer(
        cfg=cfg,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        logger=logger,
        resolved_config_path=resolved_config_path,
    )
    trainer.train()


if __name__ == "__main__":
    main()
