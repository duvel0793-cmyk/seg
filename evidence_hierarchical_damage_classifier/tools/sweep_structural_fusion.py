from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_STRUCTURAL_WEIGHTS = "0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5"
DEFAULT_ROUTER_WEIGHTS = "0.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-file", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--structural-weights", type=str, default=DEFAULT_STRUCTURAL_WEIGHTS)
    parser.add_argument("--router-weights", type=str, default=DEFAULT_ROUTER_WEIGHTS)
    return parser.parse_args()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_csv_rows(path: str | Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_weight_list(text: str) -> list[float]:
    weights = []
    for part in str(text).split(","):
        stripped = part.strip()
        if not stripped:
            continue
        value = float(stripped)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Fusion weights must be in [0, 1], got {value}.")
        weights.append(value)
    if not weights:
        raise ValueError("At least one fusion weight must be provided.")
    return weights


def _normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    sums = probabilities.sum(axis=1, keepdims=True)
    return probabilities / np.clip(sums, 1e-8, None)


def _fuse_probabilities(
    corn_probabilities: np.ndarray,
    structural_probabilities: np.ndarray,
    router_probabilities: np.ndarray | None,
    structural_weight: float,
    router_weight: float,
) -> np.ndarray:
    final_probabilities = ((1.0 - structural_weight) * corn_probabilities) + (structural_weight * structural_probabilities)
    if router_probabilities is not None and router_weight > 0.0:
        final_probabilities = ((1.0 - router_weight) * final_probabilities) + (router_weight * router_probabilities)
    return _normalize_probabilities(final_probabilities)


def _resolve_predictions_file(path: Path) -> Path:
    if path.is_dir():
        candidates = [
            path / "predictions_best.jsonl",
            path / "predictions_best.csv",
            path / "predictions_last.jsonl",
            path / "predictions_last.csv",
        ]
    elif path.name.startswith("diagnostics"):
        base = path.parent
        candidates = [
            base / "predictions_best.jsonl",
            base / "predictions_best.csv",
            base / "predictions_last.jsonl",
            base / "predictions_last.csv",
        ]
    else:
        candidates = [path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve predictions file from: {path}")


def _row_probabilities(row: dict[str, str], prefix: str) -> list[float] | None:
    values = [row.get(f"{prefix}{suffix}") for suffix in ("no", "minor", "major", "destroyed")]
    if all(value in {"", None} for value in values):
        return None
    parsed = []
    for value in values:
        if value in {"", None}:
            raise RuntimeError(f"CSV predictions are missing a probability for prefix '{prefix}'.")
        parsed.append(float(value))
    return parsed


def _load_prediction_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    if path.suffix.lower() == ".csv":
        records = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(
                    {
                        "gt_label": int(row["gt_label"]),
                        "final_class_probabilities": _row_probabilities(row, "prob_"),
                        "corn_class_probabilities": _row_probabilities(row, "corn_prob_"),
                        "structural_class_probabilities": _row_probabilities(row, "struct_prob_"),
                        "scale_router_probabilities": _row_probabilities(row, "router_prob_"),
                    }
                )
        return records
    raise ValueError(f"Unsupported predictions file: {path}")


def build_dataloader(dataset: Any, *, batch_size: int, num_workers: int, seed: int) -> Any:
    import torch
    from torch.utils.data import DataLoader

    from datasets import xbd_instance_collate_fn
    from utils.misc import seed_worker

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    data_cfg = dataset.config.get("data", {})
    pin_memory = bool(data_cfg.get("pin_memory", torch.cuda.is_available()))
    persistent_workers = bool(data_cfg.get("persistent_workers", True)) and num_workers > 0
    prefetch_factor = int(data_cfg.get("prefetch_factor", 2))
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": xbd_instance_collate_fn,
        "worker_init_fn": seed_worker,
        "generator": generator,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def _load_from_checkpoint(config_path: str, checkpoint_path: str, split: str) -> tuple[list[dict[str, Any]], Path]:
    import torch

    from datasets import XBDInstanceDataset
    from engine import build_epoch_loss, run_epoch
    from models import build_model
    from utils.checkpoint import load_checkpoint, resolve_checkpoint_model_state
    from utils.misc import get_split_list, load_config, merge_config, resolve_device, set_seed

    config = load_config(config_path)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if "config" in checkpoint and isinstance(checkpoint["config"], dict):
        config = merge_config(checkpoint["config"])
    set_seed(int(config["training"]["seed"]))
    dataset = XBDInstanceDataset(config=config, split=split, list_path=get_split_list(config, split), is_train=False)
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
    state_dict = resolve_checkpoint_model_state(checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
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
        apply_best_damage_threshold="none",
    )
    return result["prediction_records"], ensure_dir(Path(config["project"]["output_dir"]))


def _extract_probability_arrays(
    prediction_records: list[dict[str, Any]],
    *,
    require_router: bool,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray | None]:
    y_true: list[int] = []
    corn_probabilities: list[list[float]] = []
    structural_probabilities: list[list[float]] = []
    router_probabilities: list[list[float]] = []

    for idx, record in enumerate(prediction_records):
        y_true.append(int(record["gt_label"]))

        corn_probs = record.get("corn_class_probabilities") or record.get("class_probabilities")
        if corn_probs is None or len(corn_probs) != 4:
            raise RuntimeError(
                "Prediction records must contain 4-way CORN class probabilities in "
                "`corn_class_probabilities` or `class_probabilities`."
            )
        corn_probabilities.append([float(value) for value in corn_probs])

        structural_probs = record.get("structural_class_probabilities")
        if structural_probs is None or len(structural_probs) != 4:
            raise RuntimeError(
                "Structural fusion sweep requires `structural_class_probabilities` for every prediction record. "
                f"Missing or invalid entry at row {idx}."
            )
        structural_probabilities.append([float(value) for value in structural_probs])

        router_probs = record.get("scale_router_probabilities")
        if router_probs is not None:
            if len(router_probs) != 4:
                raise RuntimeError(f"Invalid `scale_router_probabilities` length at row {idx}.")
            router_probabilities.append([float(value) for value in router_probs])
        elif require_router:
            raise RuntimeError(
                "Router fusion weights were requested, but prediction records do not contain `scale_router_probabilities`."
            )

    router_array = None
    if router_probabilities:
        if len(router_probabilities) != len(y_true):
            raise RuntimeError("Router probabilities are only present for a subset of records.")
        router_array = np.asarray(router_probabilities, dtype=np.float64)
    return (
        y_true,
        np.asarray(corn_probabilities, dtype=np.float64),
        np.asarray(structural_probabilities, dtype=np.float64),
        router_array,
    )


def _candidate_priority(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> tuple[float, float, int, int, float, float, float]:
    destroyed_ok = int(candidate["destroyed_f1"] >= baseline["destroyed_f1"] - 1e-8)
    far_error_ok = int(candidate["far_error"] <= baseline["far_error"] + 1e-8)
    return (
        float(candidate["macro_f1"]),
        float(candidate["minor_f1"]),
        destroyed_ok,
        far_error_ok,
        float(candidate["destroyed_f1"]),
        -float(candidate["far_error"]),
        float(candidate["qwk"]),
    )


def main() -> None:
    from utils.metrics import compute_classification_metrics

    args = parse_args()
    if args.predictions_file is None and (args.config is None or args.checkpoint is None):
        raise SystemExit("Provide either --predictions-file or both --config and --checkpoint.")

    structural_weights = _parse_weight_list(args.structural_weights)
    router_weights = _parse_weight_list(args.router_weights)
    require_router = any(weight > 0.0 for weight in router_weights)

    if args.predictions_file is not None:
        predictions_path = _resolve_predictions_file(Path(args.predictions_file))
        prediction_records = _load_prediction_records(predictions_path)
        default_output_dir = ensure_dir(predictions_path.parent)
    else:
        prediction_records, default_output_dir = _load_from_checkpoint(args.config, args.checkpoint, args.split)
    output_dir = ensure_dir(Path(args.save_dir) if args.save_dir else default_output_dir)

    y_true, corn_probs, structural_probs, router_probs = _extract_probability_arrays(
        prediction_records,
        require_router=require_router,
    )

    table_rows: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    best_candidate: dict[str, Any] | None = None
    baseline_candidate: dict[str, Any] | None = None

    for structural_weight in structural_weights:
        for router_weight in router_weights:
            final_probabilities = _fuse_probabilities(
                corn_probabilities=corn_probs,
                structural_probabilities=structural_probs,
                router_probabilities=router_probs,
                structural_weight=float(structural_weight),
                router_weight=float(router_weight),
            )
            predictions = final_probabilities.argmax(axis=1).astype(np.int64).tolist()
            metrics = compute_classification_metrics(y_true, predictions, final_probabilities)
            candidate = {
                "structural_weight": float(structural_weight),
                "router_weight": float(router_weight),
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "qwk": float(metrics["QWK"]),
                "far_error": float(metrics["far_error"]),
                "no_f1": float(metrics["no_f1"]),
                "minor_f1": float(metrics["minor_f1"]),
                "major_f1": float(metrics["major_f1"]),
                "destroyed_f1": float(metrics["destroyed_f1"]),
                "gt_minor_pred_no_rate": float(metrics["gt_minor_pred_no_rate"]),
                "gt_minor_pred_major_rate": float(metrics["gt_minor_pred_major_rate"]),
                "gt_destroyed_pred_major_rate": float(metrics["gt_destroyed_pred_major_rate"]),
                "per_class_f1": {
                    "no-damage": float(metrics["no_f1"]),
                    "minor-damage": float(metrics["minor_f1"]),
                    "major-damage": float(metrics["major_f1"]),
                    "destroyed": float(metrics["destroyed_f1"]),
                },
                "confusion_matrix": metrics["confusion_matrix"],
            }
            if structural_weight == 0.0 and router_weight == 0.0:
                baseline_candidate = candidate
            candidate_records.append(candidate)
            table_rows.append(
                {
                    "structural_weight": candidate["structural_weight"],
                    "router_weight": candidate["router_weight"],
                    "accuracy": candidate["accuracy"],
                    "macro_f1": candidate["macro_f1"],
                    "qwk": candidate["qwk"],
                    "far_error": candidate["far_error"],
                    "no_f1": candidate["no_f1"],
                    "minor_f1": candidate["minor_f1"],
                    "major_f1": candidate["major_f1"],
                    "destroyed_f1": candidate["destroyed_f1"],
                    "gt_minor_pred_no_rate": candidate["gt_minor_pred_no_rate"],
                    "gt_minor_pred_major_rate": candidate["gt_minor_pred_major_rate"],
                    "gt_destroyed_pred_major_rate": candidate["gt_destroyed_pred_major_rate"],
                    "confusion_matrix_json": json.dumps(candidate["confusion_matrix"], ensure_ascii=False),
                }
            )

    if baseline_candidate is None:
        raise RuntimeError(
            "The sweep must include structural_weight=0.0 and router_weight=0.0 so the CORN baseline can be compared."
        )

    for candidate in table_rows:
        candidate["destroyed_f1_delta_vs_base"] = float(candidate["destroyed_f1"] - baseline_candidate["destroyed_f1"])
        candidate["far_error_delta_vs_base"] = float(candidate["far_error"] - baseline_candidate["far_error"])

    for candidate in candidate_records:
        if best_candidate is None or _candidate_priority(candidate, baseline_candidate) > _candidate_priority(best_candidate, baseline_candidate):
            best_candidate = candidate

    if best_candidate is None:
        raise RuntimeError("Structural fusion sweep produced no candidates.")

    for row in table_rows:
        row["selected_by_priority"] = int(
            row["structural_weight"] == best_candidate["structural_weight"]
            and row["router_weight"] == best_candidate["router_weight"]
        )

    best_record = {
        "selection_priority": [
            "macro_f1",
            "minor_f1",
            "destroyed_f1_not_lower_than_baseline",
            "far_error_not_higher_than_baseline",
            "destroyed_f1",
            "far_error",
            "qwk",
        ],
        "baseline": {
            **baseline_candidate,
            "destroyed_f1_delta_vs_base": 0.0,
            "far_error_delta_vs_base": 0.0,
        },
        "best": {
            **best_candidate,
            "destroyed_f1_delta_vs_base": float(best_candidate["destroyed_f1"] - baseline_candidate["destroyed_f1"]),
            "far_error_delta_vs_base": float(best_candidate["far_error"] - baseline_candidate["far_error"]),
        },
        "num_candidates": len(table_rows),
    }

    write_csv_rows(
        output_dir / "structural_fusion_sweep.csv",
        [
            "structural_weight",
            "router_weight",
            "accuracy",
            "macro_f1",
            "qwk",
            "far_error",
            "no_f1",
            "minor_f1",
            "major_f1",
            "destroyed_f1",
            "gt_minor_pred_no_rate",
            "gt_minor_pred_major_rate",
            "gt_destroyed_pred_major_rate",
            "destroyed_f1_delta_vs_base",
            "far_error_delta_vs_base",
            "selected_by_priority",
            "confusion_matrix_json",
        ],
        table_rows,
    )
    write_json(output_dir / "structural_fusion_sweep_best.json", best_record)
    print(
        "structural_fusion_sweep "
        f"best_structural_weight={best_candidate['structural_weight']:.4f} "
        f"best_router_weight={best_candidate['router_weight']:.4f} "
        f"macro_f1={best_candidate['macro_f1']:.4f} "
        f"minor_f1={best_candidate['minor_f1']:.4f} "
        f"destroyed_f1={best_candidate['destroyed_f1']:.4f} "
        f"far_error={best_candidate['far_error']:.4f}"
    )


if __name__ == "__main__":
    main()
