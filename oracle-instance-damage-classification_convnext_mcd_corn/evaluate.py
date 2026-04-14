from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import CLASS_NAMES, XBDInstanceDamageDataset, xbd_instance_collate_fn
from engine.evaluator import evaluate_model
from losses import build_loss
from models import build_model
from utils import load_checkpoint, save_json
from utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="")
    parser.add_argument("--fit-thresholds-on-current-split", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_name = args.split or str(config["data"]["test_split"])
    dataset = XBDInstanceDamageDataset(config, split_name=split_name, is_train=False)
    ce_counts = torch.tensor(dataset.class_counts, dtype=torch.float32).clamp_min(1.0)
    ce_class_weights = (ce_counts.sum() / ce_counts)
    ce_class_weights = ce_class_weights / ce_class_weights.mean()

    loader = DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        collate_fn=xbd_instance_collate_fn,
    )

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    criterion = build_loss(config, ce_class_weights=ce_class_weights.to(device))

    preset_thresholds = checkpoint.get("best_thresholds")
    report = evaluate_model(
        model=model,
        loader=loader,
        device=device,
        criterion=criterion,
        amp=bool(config["train"].get("amp", True)),
        threshold_candidates=list(config["model"].get("threshold_candidates", [0.5])),
        fit_thresholds=bool(args.fit_thresholds_on_current_split),
        preset_thresholds=None if args.fit_thresholds_on_current_split else preset_thresholds,
    )

    raw_metrics = report["raw_metrics"]
    calibrated_metrics = report["calibrated_metrics"]
    print(f"Split: {split_name}")
    print(f"Raw metrics: {raw_metrics}")
    print(f"Calibrated metrics: {calibrated_metrics}")
    print(f"Confusion matrix: {(calibrated_metrics or raw_metrics)['confusion_matrix']}")
    for class_name in CLASS_NAMES:
        payload = (calibrated_metrics or raw_metrics)["per_class"][class_name]
        print(
            f"{class_name}: precision={payload['precision']:.4f} "
            f"recall={payload['recall']:.4f} f1={payload['f1']:.4f} support={payload['support']}"
        )

    eval_output = Path(args.checkpoint).expanduser().resolve().parent / f"eval_{split_name}.json"
    save_json(
        {
            "split": split_name,
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "raw_metrics": raw_metrics,
            "calibrated_metrics": calibrated_metrics,
            "best_thresholds": report["best_thresholds"],
            "avg_losses": report["avg_losses"],
        },
        eval_output,
    )
    print(f"Saved evaluation report to {eval_output}")


if __name__ == "__main__":
    main()

