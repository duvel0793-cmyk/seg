#!/usr/bin/env python
"""Run a minimal forward/backward/validation loop."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_dataloader
from src.engine import Evaluator
from src.losses import build_criterion
from src.models import VMambaTauCornUpper
from src.utils import (
    build_run_identity,
    dump_config,
    ensure_dir,
    load_config,
    log_kv,
    resolve_device,
    save_debug_visualization,
    seed_everything,
    setup_logger,
    summarize_backend_metadata,
)
from src.utils.misc import move_batch_to_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for the standalone VMamba xBD project.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))

    output_dir = ensure_dir(cfg["output_dir"])
    log_dir = ensure_dir(output_dir / "logs")
    debug_dir = ensure_dir(output_dir / "debug_vis")
    resolved_config_path = log_dir / "smoke_test_resolved_config.yaml"
    dump_config(cfg, resolved_config_path)
    logger = setup_logger("smoke_test", log_dir / "smoke_test.log")
    device = resolve_device(cfg["device"])
    logger.info("======== Smoke test startup ========")
    logger.info("config=%s", Path(args.config).expanduser().resolve())
    logger.info("resolved_config_path=%s", resolved_config_path)
    logger.info("output_dir=%s", output_dir)
    logger.info("device=%s", device)
    log_kv(logger, "resolved_config", cfg)

    _, train_loader = build_dataloader(cfg, mode="train")
    _, val_loader = build_dataloader(cfg, mode="val")

    model = VMambaTauCornUpper(cfg).to(device)
    criterion = build_criterion(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    log_kv(logger, "run_identity", build_run_identity(cfg, model, resolved_config_path=resolved_config_path))
    log_kv(logger, "backbone", summarize_backend_metadata(model.backbone.get_metadata()))

    model.train()
    last_batch = None
    last_outputs = None
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= int(cfg["smoke_test"]["num_train_batches"]):
            break
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch, epoch=0, enable_instance_aux=True)
        loss_dict = criterion(outputs, batch, epoch=0, is_train=True)
        loss_dict["total_loss"].backward()
        optimizer.step()

        logger.info(
            "train batch=%d pre_shape=%s post_shape=%s loc_shape=%s ord_shape=%s "
            "loc_loss=%.4f pixel_corn_main=%.4f pixel_corn_soft=%.4f tau_reg=%.6f instance_corn_aux=%.4f total_loss=%.4f "
            "tau_phase=%s tau_mean=%.4f instances=%d",
            batch_idx,
            tuple(batch["pre_image"].shape),
            tuple(batch["post_image"].shape),
            tuple(outputs["loc_logits"].shape),
            tuple(outputs["ordinal_logits"].shape),
            float(loss_dict["loc_loss"].item()),
            float(loss_dict["pixel_corn_main"].item()),
            float(loss_dict["pixel_corn_soft"].item()),
            float(loss_dict["tau_reg"].item()),
            float(loss_dict["instance_corn_aux"].item()),
            float(loss_dict["total_loss"].item()),
            loss_dict["tau_phase"],
            float(loss_dict["tau_mean"]),
            int(outputs["instance_pool"]["valid_instances"]),
        )

        last_batch = batch
        last_outputs = outputs

    if last_batch is not None and last_outputs is not None:
        debug_path = debug_dir / "smoke_test_debug.png"
        save_debug_visualization(debug_path, last_batch, last_outputs)
        logger.info("Saved debug visualization to %s", debug_path)

    evaluator = Evaluator(cfg, model, criterion, device, logger=logger)
    val_result = evaluator.evaluate(val_loader, max_batches=int(cfg["smoke_test"]["num_val_batches"]), epoch=0)
    logger.info(
        "val F1_oa=%.4f F1_loc=%.4f F1_bda=%.4f tau=%s valid_instances=%d",
        val_result["F1_oa"],
        val_result["F1_loc"],
        val_result["F1_bda"],
        val_result["tau"],
        val_result["instance_aux"]["valid_instances"],
    )
    logger.info("val confusion overall=%s", val_result["overall_confusion"])


if __name__ == "__main__":
    main()
