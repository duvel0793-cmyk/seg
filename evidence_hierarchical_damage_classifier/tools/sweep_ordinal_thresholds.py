from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-file", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--t-min", type=float, default=0.30)
    parser.add_argument("--t-max", type=float, default=0.70)
    parser.add_argument("--t-step", type=float, default=0.02)
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


def _threshold_grid(t_min: float, t_max: float, t_step: float) -> list[float]:
    import numpy as np

    values = np.arange(t_min, t_max + (0.5 * t_step), t_step, dtype=np.float64)
    return [round(float(value), 4) for value in values.tolist()]


def _decode_with_thresholds(probabilities: np.ndarray, t0: float, t1: float, t2: float) -> np.ndarray:
    import numpy as np

    p_damage = 1.0 - probabilities[:, 0]
    p_major_or_higher = probabilities[:, 2] + probabilities[:, 3]
    p_destroyed = probabilities[:, 3]
    return (
        (p_damage >= float(t0)).astype(np.int64)
        + (p_major_or_higher >= float(t1)).astype(np.int64)
        + (p_destroyed >= float(t2)).astype(np.int64)
    )


def _resolve_predictions_file(path: Path) -> Path:
    if path.is_dir():
        candidates = [
            path / "predictions_best.csv",
            path / "predictions_best.jsonl",
            path / "predictions_last.csv",
            path / "predictions_last.jsonl",
        ]
    elif path.name.startswith("diagnostics"):
        base = path.parent
        candidates = [
            base / "predictions_best.csv",
            base / "predictions_best.jsonl",
            base / "predictions_last.csv",
            base / "predictions_last.jsonl",
        ]
    else:
        candidates = [path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve predictions file from: {path}")


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
        import csv

        records = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                final_probs = [row.get("prob_no"), row.get("prob_minor"), row.get("prob_major"), row.get("prob_destroyed")]
                corn_probs = [row.get("corn_prob_no"), row.get("corn_prob_minor"), row.get("corn_prob_major"), row.get("corn_prob_destroyed")]
                records.append(
                    {
                        "gt_label": int(row["gt_label"]),
                        "class_probabilities": [float(v) for v in final_probs if v not in {"", None}],
                        "corn_class_probabilities": [float(v) for v in corn_probs if v not in {"", None}],
                    }
                )
        return records
    raise ValueError(f"Unsupported predictions file: {path}")


def _load_from_checkpoint(config_path: str, checkpoint_path: str, split: str) -> tuple[list[dict[str, Any]], Path]:
    import torch

    from datasets import XBDInstanceDataset
    from engine import build_epoch_loss, run_epoch
    from models import build_model
    from utils.checkpoint import load_checkpoint
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
    state_dict = checkpoint.get("ema_state_dict") or checkpoint.get("model_state_dict") or checkpoint.get("raw_model_state_dict")
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


def main() -> None:
    import numpy as np

    from utils.metrics import compute_classification_metrics

    args = parse_args()
    if args.predictions_file is None and (args.config is None or args.checkpoint is None):
        raise SystemExit("Provide either --predictions-file or both --config and --checkpoint.")

    if args.predictions_file is not None:
        predictions_path = _resolve_predictions_file(Path(args.predictions_file))
        prediction_records = _load_prediction_records(predictions_path)
        output_dir = ensure_dir(predictions_path.parent)
    else:
        prediction_records, output_dir = _load_from_checkpoint(args.config, args.checkpoint, args.split)

    y_true = [int(record["gt_label"]) for record in prediction_records]
    probabilities = []
    for record in prediction_records:
        corn_probs = record.get("corn_class_probabilities") or record.get("class_probabilities")
        if not corn_probs or len(corn_probs) != 4:
            raise RuntimeError("Prediction records must contain 4-way CORN class probabilities.")
        probabilities.append([float(value) for value in corn_probs])
    probabilities_np = np.asarray(probabilities, dtype=np.float64)

    threshold_values = _threshold_grid(args.t_min, args.t_max, args.t_step)
    table_rows: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None
    for t0, t1, t2 in itertools.product(threshold_values, threshold_values, threshold_values):
        predictions = _decode_with_thresholds(probabilities_np, t0, t1, t2).tolist()
        metrics = compute_classification_metrics(y_true, predictions, probabilities_np)
        row = {
            "t0": float(t0),
            "t1": float(t1),
            "t2": float(t2),
            "macro_f1": float(metrics["macro_f1"]),
            "qwk": float(metrics["QWK"]),
            "far_error": float(metrics["far_error"]),
            "no_f1": float(metrics["no_f1"]),
            "minor_f1": float(metrics["minor_f1"]),
            "major_f1": float(metrics["major_f1"]),
            "destroyed_f1": float(metrics["destroyed_f1"]),
            "gt_minor_pred_no_rate": float(metrics["gt_minor_pred_no_rate"]),
            "gt_minor_pred_major_rate": float(metrics["gt_minor_pred_major_rate"]),
            "gt_major_pred_minor_rate": float(metrics["gt_major_pred_minor_rate"]),
        }
        table_rows.append(row)
        candidate = {"thresholds": {"t0": float(t0), "t1": float(t1), "t2": float(t2)}, "qwk": float(metrics["QWK"]), **metrics}
        if best_record is None or candidate["macro_f1"] > best_record["macro_f1"] or (
            candidate["macro_f1"] == best_record["macro_f1"] and candidate["QWK"] > best_record["QWK"]
        ):
            best_record = candidate

    if best_record is None:
        raise RuntimeError("Threshold sweep produced no results.")

    write_json(output_dir / "threshold_sweep_best.json", best_record)
    write_csv_rows(
        output_dir / "threshold_sweep_table.csv",
        [
            "t0",
            "t1",
            "t2",
            "macro_f1",
            "qwk",
            "far_error",
            "no_f1",
            "minor_f1",
            "major_f1",
            "destroyed_f1",
            "gt_minor_pred_no_rate",
            "gt_minor_pred_major_rate",
            "gt_major_pred_minor_rate",
        ],
        table_rows,
    )
    thresholds = best_record["thresholds"]
    print(
        f"best thresholds t0={thresholds['t0']:.2f} t1={thresholds['t1']:.2f} t2={thresholds['t2']:.2f} "
        f"macro_f1={best_record['macro_f1']:.4f} qwk={best_record['QWK']:.4f} far_error={best_record['far_error']:.4f}"
    )


if __name__ == "__main__":
    main()
