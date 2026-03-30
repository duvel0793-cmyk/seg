"""One-command orchestration for xBD inference plus evaluation."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import logging
import sys

from .cli import add_run_arguments, default_vscode_argv, prepare_run_args, run_pipeline
from .config import resolve_experiment_summary_path, resolve_run_summary_path
from .evaluation import (
    evaluate_predictions,
    prepare_evaluation_args,
    write_evaluation_outputs,
)
from .io_utils import configure_logging, write_json


def build_parser(add_help: bool = True) -> ArgumentParser:
    """Build the full-experiment parser."""
    parser = ArgumentParser(
        add_help=add_help,
        description="Run xBD batch inference and batch evaluation in one command.",
    )
    add_run_arguments(parser)
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=None,
        help="Optional GT directory. Defaults to <xbd-root>/<split>/targets.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional evaluation summary JSON path.",
    )
    parser.add_argument(
        "--per-image-csv",
        type=Path,
        default=None,
        help="Optional per-image metrics CSV path.",
    )
    return parser


def _build_evaluation_args(run_args: Namespace) -> Namespace:
    """Construct evaluation arguments from a prepared run namespace."""
    return prepare_evaluation_args(
        Namespace(
            pred_dir=run_args.output_dir / "masks",
            gt_dir=run_args.gt_dir,
            split=run_args.split,
            xbd_root=run_args.xbd_root,
            start_index=run_args.start_index,
            limit=run_args.limit,
            per_image_csv=run_args.per_image_csv,
            summary_json=run_args.summary_json,
            verbose=run_args.verbose,
        )
    )


def run_full_experiment(args: Namespace) -> dict:
    """Run inference, then evaluate generated masks, and persist a joint summary."""
    args = prepare_run_args(args)
    experiment_summary_path = resolve_experiment_summary_path(args.output_dir)
    run_summary = run_pipeline(args)

    if run_summary["success_count"] == 0 and run_summary["skipped_count"] == 0:
        message = "No successful or reusable predictions were produced; skipping evaluation."
        logging.warning(message)
        experiment_summary = {"run": run_summary, "evaluation": None, "message": message}
        write_json(experiment_summary_path, experiment_summary)
        return experiment_summary

    evaluation_summary = None
    try:
        eval_args = _build_evaluation_args(args)
        evaluation_summary, rows = evaluate_predictions(eval_args)
        write_evaluation_outputs(
            eval_args.summary_json,
            eval_args.per_image_csv,
            evaluation_summary,
            rows,
        )
    except Exception as exc:
        logging.exception("Batch evaluation failed: %s", exc)
        evaluation_summary = {"error": str(exc)}

    experiment_summary = {
        "mode": "text" if args.prompt_mode == "text" else "refine",
        "run": run_summary,
        "evaluation": evaluation_summary,
    }
    write_json(experiment_summary_path, experiment_summary)
    return experiment_summary


def main(argv=None) -> int:
    """Full-experiment CLI entrypoint."""
    if argv is None and len(sys.argv) == 1:
        argv = default_vscode_argv()
        print(
            "No CLI arguments provided; using VSCode default preset: "
            "--split train --prompt-mode text --skip-existing"
        )
    try:
        args = prepare_run_args(build_parser().parse_args(argv))
    except Exception as exc:
        print(f"Argument error: {exc}")
        return 1

    configure_logging(args.log_file, verbose=args.verbose)
    experiment_summary = run_full_experiment(args)
    run_summary = experiment_summary["run"]
    evaluation_summary = experiment_summary["evaluation"]
    experiment_summary_path = resolve_experiment_summary_path(args.output_dir)

    print(f"Run summary: {resolve_run_summary_path(args.output_dir)}")
    if evaluation_summary is None:
        print(experiment_summary["message"])
        print(f"Experiment summary: {experiment_summary_path}")
        return 1
    if "error" in evaluation_summary:
        print(f"Experiment finished with evaluation error: {evaluation_summary['error']}")
        print(f"Experiment summary: {experiment_summary_path}")
        return 1

    print(f"Experiment summary: {experiment_summary_path}")
    print(f"Matched images: {evaluation_summary['matched_images']}")
    print(f"Skipped images: {evaluation_summary['skipped_images']}")
    print(f"Failed images: {evaluation_summary['failed_images']}")
    print(f"mean IoU: {evaluation_summary['mean_iou']:.4f}")
    print(f"mean F1: {evaluation_summary['mean_f1']:.4f}")
    print(f"mean Precision: {evaluation_summary['mean_precision']:.4f}")
    print(f"mean Recall: {evaluation_summary['mean_recall']:.4f}")
    return 0 if run_summary["failed_count"] == 0 and evaluation_summary["failed_images"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
