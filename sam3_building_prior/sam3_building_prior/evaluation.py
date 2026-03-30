"""Batch evaluation helpers for binary building masks."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import csv
import logging

import numpy as np

from .config import (
    DEFAULT_XBD_ROOT,
    IMAGE_EXTENSIONS,
    SUPPORTED_SPLITS,
    resolve_metrics_base_dir,
    resolve_metrics_summary_path,
    resolve_per_image_metrics_csv_path,
    resolve_split_gt_dir,
)
from .io_utils import resolve_matching_image_path, write_json
from .prompting import read_binary_mask


def add_evaluation_arguments(parser: ArgumentParser) -> ArgumentParser:
    """Attach evaluation-related CLI arguments to a parser."""
    parser.add_argument(
        "--pred-dir",
        "--pred-mask-dir",
        dest="pred_dir",
        type=Path,
        required=True,
        help="Prediction mask directory or a single prediction mask file.",
    )
    parser.add_argument(
        "--gt-dir",
        "--gt-mask-dir",
        dest="gt_dir",
        type=Path,
        default=None,
        help="Ground-truth mask directory or a single GT mask file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=SUPPORTED_SPLITS,
        default=None,
        help="xBD split used to infer the default GT directory when --gt-dir is omitted.",
    )
    parser.add_argument(
        "--xbd-root",
        type=Path,
        default=DEFAULT_XBD_ROOT,
        help="xBD dataset root used with --split.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip the first N prediction files after sorting.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate at most N prediction files. Use 0 for all.",
    )
    parser.add_argument(
        "--per-image-csv",
        type=Path,
        default=None,
        help="Output CSV for per-image metrics.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Output JSON for overall metrics summary.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print evaluated file names in addition to aggregate metrics.",
    )
    return parser


def build_parser(add_help: bool = True) -> ArgumentParser:
    """Build the evaluation parser."""
    parser = ArgumentParser(
        add_help=add_help,
        description="Evaluate predicted building masks against GT masks.",
    )
    return add_evaluation_arguments(parser)


def parse_args(argv=None) -> Namespace:
    """Parse CLI arguments for evaluation."""
    return build_parser().parse_args(argv)


def prepare_evaluation_args(args: Namespace) -> Namespace:
    """Resolve derived evaluation paths and validate argument combinations."""
    if getattr(args, "_sam3_eval_prepared", False):
        return args

    args.xbd_root = Path(args.xbd_root).resolve()
    args.pred_dir = Path(args.pred_dir).resolve()
    if not args.pred_dir.exists():
        raise FileNotFoundError(f"Prediction path does not exist: {args.pred_dir}")

    if args.gt_dir is None:
        if args.split is None:
            raise ValueError("Either --gt-dir or --split must be provided for evaluation")
        args.gt_dir = resolve_split_gt_dir(args.split, args.xbd_root)
    args.gt_dir = Path(args.gt_dir).resolve()
    if not args.gt_dir.exists():
        raise FileNotFoundError(f"GT path does not exist: {args.gt_dir}")

    if args.start_index < 0:
        raise ValueError("start_index must be >= 0")
    if args.limit < 0:
        raise ValueError("limit must be >= 0")

    base_dir = resolve_metrics_base_dir(args.pred_dir)
    if args.summary_json is None:
        args.summary_json = resolve_metrics_summary_path(base_dir)
    else:
        args.summary_json = Path(args.summary_json).resolve()
    if args.per_image_csv is None:
        args.per_image_csv = resolve_per_image_metrics_csv_path(base_dir)
    else:
        args.per_image_csv = Path(args.per_image_csv).resolve()

    args._sam3_eval_prepared = True
    return args


def iter_prediction_paths(pred_dir: Path) -> list[Path]:
    """Collect prediction mask files from a directory or a single file path."""
    if pred_dir.is_file():
        return [pred_dir] if pred_dir.suffix.lower() in IMAGE_EXTENSIONS else []
    return sorted(
        path
        for path in pred_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def resolve_gt_path(pred_path: Path, gt_dir: Path) -> Path | None:
    """Resolve the matching GT path for a prediction file."""
    return resolve_matching_image_path(
        search_root=gt_dir,
        candidate_names=[pred_path.name],
        candidate_stems=[
            pred_path.stem,
            pred_path.stem.replace("_pre_disaster", "_pre_disaster_target"),
        ],
        suffixes=IMAGE_EXTENSIONS,
    )


def compute_mask_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict[str, float]:
    """Compute binary IoU / F1 / Precision / Recall for a single mask pair."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gt).sum())

    iou_den = tp + fp + fn
    iou = 1.0 if iou_den == 0 else tp / iou_den

    precision_den = tp + fp
    recall_den = tp + fn
    precision = 1.0 if precision_den == 0 and recall_den == 0 else 0.0
    if precision_den > 0:
        precision = tp / precision_den
    recall = 1.0 if recall_den == 0 else tp / recall_den

    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "iou": float(iou),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


def evaluate_predictions(args: Namespace) -> tuple[dict, list[dict[str, float | str]]]:
    """Evaluate prediction masks against GT and return summary plus per-image rows."""
    args = prepare_evaluation_args(args)
    pred_paths = iter_prediction_paths(args.pred_dir)
    pred_paths = pred_paths[args.start_index :]
    if args.limit > 0:
        pred_paths = pred_paths[: args.limit]

    if not pred_paths:
        raise FileNotFoundError(f"No prediction masks found in {args.pred_dir}")

    rows: list[dict[str, float | str]] = []
    skipped_files: list[str] = []
    failed_files: list[dict[str, str | None]] = []

    for pred_path in pred_paths:
        gt_path = resolve_gt_path(pred_path, args.gt_dir)
        if gt_path is None:
            skipped_files.append(pred_path.name)
            continue

        try:
            pred_mask = read_binary_mask(pred_path)
            gt_mask = read_binary_mask(gt_path)
            if pred_mask.shape != gt_mask.shape:
                raise ValueError(
                    f"Shape mismatch: pred={pred_mask.shape}, gt={gt_mask.shape}"
                )
            metrics = compute_mask_metrics(pred_mask, gt_mask)
        except Exception as exc:
            logging.exception(
                "Failed evaluating %s against %s: %s",
                pred_path,
                gt_path,
                exc,
            )
            failed_files.append(
                {
                    "pred_file": pred_path.name,
                    "gt_file": gt_path.name,
                    "error": str(exc),
                }
            )
            continue

        rows.append(
            {
                "pred_file": pred_path.name,
                "gt_file": gt_path.name,
                "iou": metrics["iou"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
            }
        )

    if not rows:
        raise RuntimeError("No matched prediction/GT mask pairs were successfully evaluated")

    summary = {
        "pred_dir": str(args.pred_dir),
        "gt_dir": str(args.gt_dir),
        "start_index": args.start_index,
        "limit": args.limit,
        "prediction_files_scanned": len(pred_paths),
        "matched_images": len(rows),
        "skipped_images": len(skipped_files),
        "failed_images": len(failed_files),
        "evaluated_files": [str(row["pred_file"]) for row in rows],
        "skipped_files": skipped_files,
        "failed_files": failed_files,
        "mean_iou": float(np.mean([row["iou"] for row in rows])),
        "mean_f1": float(np.mean([row["f1"] for row in rows])),
        "mean_precision": float(np.mean([row["precision"] for row in rows])),
        "mean_recall": float(np.mean([row["recall"] for row in rows])),
    }
    return summary, rows


def write_per_image_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    """Write per-image evaluation rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["pred_file", "gt_file", "iou", "f1", "precision", "recall"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_evaluation_outputs(
    summary_json: Path,
    per_image_csv: Path,
    summary: dict,
    rows: list[dict[str, float | str]],
) -> None:
    """Persist evaluation outputs to JSON and CSV."""
    write_json(summary_json, summary)
    write_per_image_csv(per_image_csv, rows)


def run(args: Namespace) -> tuple[dict, list[dict[str, float | str]]]:
    """Run evaluation and persist JSON/CSV outputs."""
    args = prepare_evaluation_args(args)
    summary, rows = evaluate_predictions(args)
    write_evaluation_outputs(args.summary_json, args.per_image_csv, summary, rows)
    return summary, rows


def main(argv=None) -> int:
    """Evaluation CLI entrypoint."""
    try:
        args = prepare_evaluation_args(parse_args(argv))
        summary, _ = run(args)
    except Exception as exc:
        print(f"Evaluation failed: {exc}")
        return 1

    print(f"Prediction input: {args.pred_dir}")
    print(f"GT input: {args.gt_dir}")
    print(f"Matched images: {summary['matched_images']}")
    print(f"Skipped images: {summary['skipped_images']}")
    print(f"Failed images: {summary['failed_images']}")
    if args.verbose and summary["evaluated_files"]:
        print("Evaluated files:")
        for name in summary["evaluated_files"]:
            print(f"- {name}")
    print(f"mean IoU: {summary['mean_iou']:.4f}")
    print(f"mean F1: {summary['mean_f1']:.4f}")
    print(f"mean Precision: {summary['mean_precision']:.4f}")
    print(f"mean Recall: {summary['mean_recall']:.4f}")
    print(f"Summary JSON: {args.summary_json}")
    print(f"Per-image CSV: {args.per_image_csv}")
    return 0 if summary["failed_images"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
