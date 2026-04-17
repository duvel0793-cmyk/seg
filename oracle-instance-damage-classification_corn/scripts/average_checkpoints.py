from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.io import load_checkpoint, read_yaml, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average checkpoints for oracle instance damage classification.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def _average_state_dicts(state_dicts: list[dict[str, Any]], name: str) -> dict[str, Any]:
    if not state_dicts:
        raise ValueError(f"No state dicts were provided for '{name}'.")

    reference_keys = list(state_dicts[0].keys())
    reference_key_set = set(reference_keys)
    for index, state_dict in enumerate(state_dicts[1:], start=1):
        if set(state_dict.keys()) != reference_key_set:
            raise ValueError(
                f"State dict keys for '{name}' do not match between checkpoint 0 and checkpoint {index}."
            )

    averaged: dict[str, Any] = {}
    for key in reference_keys:
        values = [state_dict[key] for state_dict in state_dicts]
        first_value = values[0]
        if torch.is_tensor(first_value):
            for index, value in enumerate(values[1:], start=1):
                if not torch.is_tensor(value):
                    raise ValueError(f"Key '{key}' in '{name}' is not consistently a tensor across checkpoints.")
                if value.shape != first_value.shape or value.dtype != first_value.dtype:
                    raise ValueError(
                        f"Key '{key}' in '{name}' has mismatched shape/dtype between checkpoint 0 "
                        f"and checkpoint {index}: {tuple(first_value.shape)}/{first_value.dtype} vs "
                        f"{tuple(value.shape)}/{value.dtype}."
                    )
            if torch.is_floating_point(first_value):
                stacked = torch.stack([value.detach().cpu().to(torch.float32) for value in values], dim=0)
                averaged[key] = stacked.mean(dim=0).to(dtype=first_value.dtype)
            else:
                averaged[key] = first_value.detach().cpu().clone()
        else:
            averaged[key] = copy.deepcopy(first_value)
    return averaged


def main() -> None:
    args = parse_args()
    checkpoint_paths = [Path(path) for path in args.checkpoints]
    if len(checkpoint_paths) < 2:
        raise ValueError("Checkpoint averaging requires at least two checkpoints.")

    checkpoints = [load_checkpoint(path, map_location="cpu") for path in checkpoint_paths]
    for path, checkpoint in zip(checkpoint_paths, checkpoints):
        if "model_state_dict" not in checkpoint:
            raise KeyError(f"Checkpoint does not contain 'model_state_dict': {path}")

    output_path = Path(args.output)
    config = checkpoints[0].get("config", read_yaml(args.config))
    averaged_checkpoint: dict[str, Any] = {
        "epoch": max(int(checkpoint.get("epoch", 0)) for checkpoint in checkpoints),
        "stage_name": "averaged",
        "config": config,
        "model_state_dict": _average_state_dicts(
            [checkpoint["model_state_dict"] for checkpoint in checkpoints],
            "model_state_dict",
        ),
        "averaged_from": [str(path) for path in checkpoint_paths],
        "num_averaged_checkpoints": len(checkpoint_paths),
    }

    if all("loss_state_dict" in checkpoint for checkpoint in checkpoints):
        averaged_checkpoint["loss_state_dict"] = _average_state_dicts(
            [checkpoint["loss_state_dict"] for checkpoint in checkpoints],
            "loss_state_dict",
        )

    for key in [
        "class_names",
        "class_weights",
        "class_counts",
        "best_macro_f1",
        "ordinal_state",
        "tau_statistics",
        "difficulty_statistics",
        "corr_tau_difficulty",
        "corr_raw_tau_difficulty",
        "tau_phase",
        "metrics",
    ]:
        if key in checkpoints[0]:
            averaged_checkpoint[key] = copy.deepcopy(checkpoints[0][key])

    save_checkpoint(output_path, averaged_checkpoint)
    print(f"Averaged {len(checkpoint_paths)} checkpoints into {output_path}")
    print(f"Source checkpoints: {[str(path) for path in checkpoint_paths]}")


if __name__ == "__main__":
    main()
