"""Validation entrypoint for Clean-MODG."""

from __future__ import annotations

import argparse

from losses import build_loss
from models import build_model
from train import build_eval_loader, evaluate_model, prepare_output_dirs
from utils.checkpoint import load_checkpoint
from utils.common import CLASS_NAMES, format_metrics_line, save_predictions_csv
from utils.config import load_config
from utils.distributed import get_device
from utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Clean-MODG checkpoint.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dirs = prepare_output_dirs(config["project"]["output_dir"])
    logger = setup_logger(dirs["root"], name="Clean-MODG-validate")
    device = get_device()
    loader = build_eval_loader(config, split=config["data"].get("split_val", "val"))
    model = build_model(config).to(device)
    criterion = build_loss(config)
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    metrics, predictions = evaluate_model(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        use_amp=bool(config["train"].get("amp", False) and device.type == "cuda"),
    )
    logger.info("Validation | %s", format_metrics_line(metrics))
    logger.info("Per-class F1: %s", [round(x, 4) for x in metrics["per_class_f1"]])
    logger.info("Class order: %s", CLASS_NAMES)
    save_predictions_csv(predictions, dirs["predictions"] / "val_predictions.csv")


if __name__ == "__main__":
    main()
