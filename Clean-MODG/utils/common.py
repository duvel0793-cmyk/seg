"""Common helpers shared across scripts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
import torch

from losses.corn_loss import corn_class_probs, corn_predict

CLASS_NAMES = ["no_damage", "minor_damage", "major_damage", "destroyed"]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


class AverageMeter:
    """Track running averages for scalar metrics."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.val = float(value)
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def maybe_detach(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def outputs_to_probs(outputs: Mapping[str, Any]) -> torch.Tensor:
    if outputs.get("logits") is not None:
        return torch.softmax(outputs["logits"], dim=1)
    if outputs.get("corn_logits") is not None:
        return corn_class_probs(outputs["corn_logits"])
    raise KeyError("Model outputs must contain either 'logits' or 'corn_logits'.")


def outputs_to_predictions(outputs: Mapping[str, Any]) -> torch.Tensor:
    if outputs.get("logits") is not None:
        return torch.argmax(outputs["logits"], dim=1)
    if outputs.get("corn_logits") is not None:
        return corn_predict(outputs["corn_logits"])
    raise KeyError("Model outputs must contain either 'logits' or 'corn_logits'.")


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    if tensor.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(tensor.shape)}.")
    array = tensor.detach().cpu().float().numpy()
    if array.shape[0] in (1, 3, 4):
        array = np.transpose(array[:3], (1, 2, 0))
    array = np.clip(array, 0.0, 1.0)
    return (array * 255.0).astype(np.uint8)


def save_predictions_csv(records: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    records = list(records)
    if not records:
        with path.open("w", newline="", encoding="utf-8") as fp:
            fp.write("")
        return path
    fieldnames = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return path


def flatten_metrics_for_csv(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            for idx, item in enumerate(value):
                flat[f"{key}_{idx}"] = item
        else:
            flat[key] = value
    return flat


def format_metrics_line(metrics: Mapping[str, Any]) -> str:
    parts: List[str] = []
    for key in [
        "loss",
        "accuracy",
        "macro_f1",
        "ordinal_mae",
        "severe_error_rate",
        "bridge_score",
    ]:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    return " | ".join(parts)
