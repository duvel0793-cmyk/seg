#!/usr/bin/env python
"""Validation entry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_dataloader
from src.engine import Evaluator
from src.losses import build_criterion
from src.models import FlowMambaTauCornUpper
from src.utils import (
    build_checkpoint_metadata,
    ensure_dir,
    load_checkpoint,
    load_config,
    log_kv,
    resolve_device,
    setup_logger,
    summarize_backend_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the standalone FlowMamba xBD project.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="")
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = ensure_dir(cfg["output_dir"])
    log_dir = ensure_dir(output_dir / "logs")
    logger = setup_logger("validate", log_dir / "validate.log")
    device = resolve_device(cfg["device"])

    _, val_loader = build_dataloader(cfg, mode="val")
    model = FlowMambaTauCornUpper(cfg).to(device)
    criterion = build_criterion(cfg)
    state = load_checkpoint(
        args.checkpoint,
        model=model,
        map_location=str(device),
        expected_metadata=build_checkpoint_metadata(cfg, model),
        strict_metadata=True,
        logger=logger,
        load_weights_only=True,
    )

    checkpoint_epoch = int(state.get("epoch", 0))
    logger.info("Loaded checkpoint epoch=%d from %s", checkpoint_epoch, args.checkpoint)
    log_kv(logger, "backbone", summarize_backend_metadata(model.backbone.get_metadata()))

    evaluator = Evaluator(cfg, model, criterion, device=device, logger=logger)
    checkpoint_stem = Path(args.checkpoint).stem
    save_dir = args.save_dir or str(output_dir / "validate" / checkpoint_stem)
    result = evaluator.evaluate(
        val_loader,
        max_batches=args.max_batches if args.max_batches is not None else cfg["train"].get("max_val_batches"),
        epoch=checkpoint_epoch,
        save_dir=save_dir,
    )
    logger.info(
        "F1_oa=%.4f F1_loc=%.4f F1_bda=%.4f F1_subcls=%s",
        result["F1_oa"],
        result["F1_loc"],
        result["F1_bda"],
        result["F1_subcls"],
    )
    logger.info("Saved validation artifacts to %s", save_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

