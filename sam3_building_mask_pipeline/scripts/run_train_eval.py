#!/usr/bin/env python3
"""Train on xBD train split, then evaluate on xBD test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam_compare.config import load_experiment_config
from sam_compare.infer import evaluate_model
from sam_compare.paths import CONFIGS_DIR
from sam_compare.trainer import train_model
from sam_compare.utils import to_jsonable, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None, help="Optional extra YAML config to merge.")
    parser.add_argument("--xbd-root", type=str, default=None, help="Path to xBD root.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to SAM3 base checkpoint.")
    parser.add_argument("--output-dir", type=str, default=None, help="Root output directory for the full run.")
    parser.add_argument("--device", type=str, default=None, help="Device, for example cuda or cpu.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for both train and eval unless overridden elsewhere.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers.")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from a saved experiment checkpoint.")
    parser.add_argument("--image-size", type=int, default=None, help="Resized square image size fed to SAM3.")
    parser.add_argument("--save-pred-masks", action="store_true", help="Export predicted PNG masks during evaluation.")
    parser.add_argument(
        "--save-visualizations",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Export prediction-vs-label comparisons and prediction overlays during evaluation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_paths = [CONFIGS_DIR / "default.yaml", CONFIGS_DIR / "train.yaml", CONFIGS_DIR / "eval.yaml"]
    if args.config:
        config_paths.append(Path(args.config))
    config = load_experiment_config(config_paths)

    if args.xbd_root is not None:
        config.paths.xbd_root = Path(args.xbd_root).expanduser().resolve()
    if args.checkpoint is not None:
        config.paths.checkpoint = Path(args.checkpoint).expanduser().resolve()
    if args.output_dir is None:
        output_dir = config.paths.output_root / "xbd_sam3_run"
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()
    if args.device is not None:
        config.system.device = args.device
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
        config.eval.batch_size = args.batch_size
    if args.lr is not None:
        config.train.lr = args.lr
    if args.num_workers is not None:
        config.system.num_workers = args.num_workers
    if args.resume is not None:
        config.train.resume = Path(args.resume).expanduser().resolve()
    if args.image_size is not None:
        config.data.image_size = args.image_size
    if args.save_pred_masks:
        config.eval.save_pred_masks = True
    if args.save_visualizations is not None:
        config.eval.save_visualizations = args.save_visualizations

    train_dir = output_dir / "train"
    eval_dir = output_dir / "eval"
    train_artifacts = train_model(config, output_dir=train_dir)
    eval_artifacts = evaluate_model(
        config,
        output_dir=eval_dir,
        checkpoint_path=train_artifacts.best_model_path,
    )

    summary = {
        "train_output_dir": str(train_artifacts.output_dir),
        "best_model_path": str(train_artifacts.best_model_path),
        "training_summary_json": str(train_artifacts.training_summary_path),
        "best_epoch": train_artifacts.best_epoch,
        "best_val_iou": train_artifacts.best_val_iou,
        "eval_output_dir": str(eval_artifacts.output_dir),
        "metrics_json": str(eval_artifacts.metrics_json_path),
        "metrics_breakdown_json": str(eval_artifacts.metrics_breakdown_path),
        "per_image_metrics_csv": str(eval_artifacts.per_image_csv_path),
        "eval_config_used_json": str(eval_artifacts.eval_config_used_path),
        "pred_masks_dir": str(eval_artifacts.pred_masks_dir) if eval_artifacts.pred_masks_dir else None,
        "comparison_vis_dir": str(eval_artifacts.comparison_vis_dir) if eval_artifacts.comparison_vis_dir else None,
        "overlay_vis_dir": str(eval_artifacts.overlay_vis_dir) if eval_artifacts.overlay_vis_dir else None,
        "used_threshold": eval_artifacts.used_threshold,
    }
    write_json(summary, output_dir / "summary.json")
    print(json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
