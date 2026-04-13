"""Evaluate a trained checkpoint on val or test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from datasets.build import build_dataloader
from engine.evaluator import Evaluator
from models.build import build_model
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.logger import setup_logger
from utils.misc import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate oracle instance damage classifier.")
    parser.add_argument("--config", default="configs/default.yaml", help="Config path.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Evaluation split.")
    parser.add_argument("--output-dir", default="", help="Optional evaluation output directory.")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")

    device = resolve_device(config["runtime"].get("device", "cuda"))
    output_dir = args.output_dir or str(Path(args.checkpoint).resolve().parent / f"eval_{args.split}")
    logger = setup_logger("evaluate", output_dir=output_dir, filename="evaluate.log")

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    dataloader = build_dataloader(config, split=args.split, shuffle=False, logger=logger)

    evaluator = Evaluator(model=model, config=config, device=device, logger=logger)
    metrics = evaluator.evaluate(dataloader, split=args.split)

    logger.info("Split: %s", args.split)
    logger.info("Accuracy: %.4f", metrics["accuracy"])
    logger.info("Macro F1: %.4f", metrics["macro_f1"])
    logger.info("Weighted F1: %.4f", metrics["weighted_f1"])
    logger.info("Per-class F1: %s", metrics["per_class_f1"])
    logger.info("Tau clamped: %s", metrics["tau_clamped"])
    logger.info("Tau positive before clamp: %s", metrics["tau_positive_before_clamp"])
    logger.info("Decode mode: %s", metrics["decode_mode"])
    logger.info("Confusion Matrix: %s", metrics["confusion_matrix"])
    for item in metrics["per_class"]:
        logger.info(
            "Class %s precision=%.4f recall=%.4f f1=%.4f support=%d",
            item["class_name"],
            item["precision"],
            item["recall"],
            item["f1"],
            item["support"],
        )

    save_json(metrics, Path(output_dir) / f"metrics_{args.split}.json")
    with open(Path(output_dir) / f"metrics_{args.split}.txt", "w", encoding="utf-8") as handle:
        handle.write(f"split: {args.split}\n")
        handle.write(f"accuracy: {metrics['accuracy']:.6f}\n")
        handle.write(f"macro_f1: {metrics['macro_f1']:.6f}\n")
        handle.write(f"weighted_f1: {metrics['weighted_f1']:.6f}\n")
        handle.write(f"confusion_matrix: {metrics['confusion_matrix']}\n")
        for item in metrics["per_class"]:
            handle.write(
                f"{item['class_name']}: precision={item['precision']:.6f}, "
                f"recall={item['recall']:.6f}, f1={item['f1']:.6f}, support={item['support']}\n"
            )


if __name__ == "__main__":
    main()
