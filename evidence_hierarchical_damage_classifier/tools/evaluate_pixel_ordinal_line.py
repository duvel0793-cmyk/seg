from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import XBDInstanceDataset
from engine import build_epoch_loss, run_epoch
from evaluate import build_dataloader
from models import build_model
from utils.checkpoint import load_checkpoint
from utils.misc import append_jsonl, ensure_dir, get_split_list, load_config, merge_config, resolve_device, set_seed, write_csv_rows, write_json, write_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    return parser.parse_args()


def _resolve_state_dict(checkpoint: dict[str, Any], *, use_ema: bool) -> dict[str, Any]:
    if use_ema:
        state_dict = checkpoint.get("ema_state_dict") or checkpoint.get("model_state_dict") or checkpoint.get("raw_model_state_dict")
    else:
        state_dict = checkpoint.get("raw_model_state_dict") or checkpoint.get("model_state_dict") or checkpoint.get("ema_state_dict")
    if state_dict is None:
        raise RuntimeError("Checkpoint does not contain a usable model state dict.")
    return state_dict


def _collect_per_class_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    branch_map = {
        "main_instance": metrics.get("main_instance_per_class_f1") or {},
        "pixel_agg": metrics.get("pixel_agg_per_class_f1") or {},
        "final": metrics.get("final_per_class_f1") or {},
    }
    for branch_name, per_class in branch_map.items():
        for class_name, f1 in per_class.items():
            rows.append({"branch": branch_name, "class_name": class_name, "f1": float(f1)})
    return rows


def _collect_confusion_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    matrices = {
        "main_instance": metrics.get("main_instance_confusion_matrix") or [],
        "pixel_agg": metrics.get("pixel_agg_confusion_matrix") or [],
        "final": metrics.get("final_confusion_matrix") or metrics.get("confusion_matrix") or [],
    }
    class_names = ["no-damage", "minor-damage", "major-damage", "destroyed"]
    for branch_name, matrix in matrices.items():
        for gt_idx, row in enumerate(matrix):
            for pred_idx, value in enumerate(row):
                rows.append(
                    {
                        "branch": branch_name,
                        "gt_class": class_names[gt_idx],
                        "pred_class": class_names[pred_idx],
                        "count": int(value),
                    }
                )
    return rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    if isinstance(checkpoint.get("config"), dict):
        config = merge_config(checkpoint["config"])
    set_seed(int(config["training"]["seed"]))
    save_dir = ensure_dir(
        Path(args.save_dir)
        if args.save_dir
        else Path(config["project"]["output_dir"]) / "eval_pixel_ordinal_line" / Path(args.checkpoint).stem
    )
    write_yaml(save_dir / "eval_config.yaml", config)

    dataset = XBDInstanceDataset(config=config, split=args.split, list_path=get_split_list(config, args.split), is_train=False)
    if args.max_samples is not None:
        dataset.samples = dataset.samples[: max(1, int(args.max_samples))]
    loader = build_dataloader(
        dataset,
        batch_size=int(config["eval"]["batch_size"]),
        num_workers=int(config["eval"]["num_workers"]),
        seed=int(config["training"]["seed"]),
    )
    device = resolve_device()
    model = build_model(config).to(device)
    if bool(config["training"].get("channels_last", False)):
        model = model.to(memory_format=torch.channels_last)
    incompatible = model.load_state_dict(_resolve_state_dict(checkpoint, use_ema=args.use_ema), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            "checkpoint_load_warning "
            f"missing={list(incompatible.missing_keys)} unexpected={list(incompatible.unexpected_keys)}"
        )
    criterion = build_epoch_loss(config).to(device)
    result = run_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        config=config,
        optimizer=None,
        scaler=None,
        epoch_index=max(int(checkpoint.get("epoch", 1)) - 1, 0),
        collect_predictions=True,
    )

    write_json(save_dir / "metrics.json", result["metrics"])
    write_csv_rows(save_dir / "per_class_f1.csv", ["branch", "class_name", "f1"], _collect_per_class_rows(result["metrics"]))
    write_csv_rows(
        save_dir / "confusion_matrix.csv",
        ["branch", "gt_class", "pred_class", "count"],
        _collect_confusion_rows(result["metrics"]),
    )
    sample_predictions_path = save_dir / "sample_predictions.jsonl"
    if sample_predictions_path.exists():
        sample_predictions_path.unlink()
    for record in result["prediction_records"]:
        sample = dataset.samples[int(record["sample_index"])]
        append_jsonl(
            sample_predictions_path,
            {
                "image_id": record.get("image_id"),
                "disaster": sample.get("disaster_name"),
                "polygon_id": sample.get("building_idx"),
                "building_id": record.get("building_id"),
                "gt_label": record.get("gt_label"),
                "instance_pred": record.get("instance_pred_label", record.get("corn_pred_label")),
                "pixel_agg_pred": record.get("pixel_agg_pred_label"),
                "final_pred": record.get("final_pred_label"),
                "pixel_instance_probabilities": record.get("pixel_instance_probabilities"),
                "instance_probabilities": record.get("instance_probabilities", record.get("corn_class_probabilities")),
            },
        )

    metrics = result["metrics"]
    print(
        f"main_instance_macro_f1={metrics.get('main_instance_macro_f1', 0.0):.4f} "
        f"pixel_agg_macro_f1={0.0 if metrics.get('pixel_agg_macro_f1') is None else metrics.get('pixel_agg_macro_f1'):.4f} "
        f"final_macro_f1={metrics.get('final_macro_f1', 0.0):.4f}"
    )


if __name__ == "__main__":
    main()
