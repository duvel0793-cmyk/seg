from __future__ import annotations

import argparse
from pathlib import Path

import torch

from datasets import XBDQueryDataset, xbd_query_collate_fn
from engine.evaluator import evaluate_model
from losses import DABQNLoss
from models import build_dabqn_model
from models.query.matcher import HungarianMatcher
from torch.utils.data import DataLoader
from utils.checkpoint import load_checkpoint
from utils.misc import ensure_dir, load_yaml, write_json, write_yaml
from utils.seed import seed_worker, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--stage", type=str, choices=["localization", "damage", "joint"], default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if args.stage is not None:
        config.setdefault("training", {})["stage"] = args.stage
    stage = str(config["training"].get("stage", "joint"))
    set_seed(int(config["training"].get("seed", 42)))

    save_dir = ensure_dir(args.save_dir or str(Path(config["project"]["output_dir"]) / "eval" / args.split))
    write_yaml(Path(save_dir) / "eval_config.yaml", config)

    dataset = XBDQueryDataset(config=config, split=args.split, is_train=False)
    loader = DataLoader(
        dataset,
        batch_size=int(config["eval"].get("batch_size", 2)),
        shuffle=False,
        num_workers=int(config["eval"].get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=xbd_query_collate_fn,
        worker_init_fn=seed_worker,
    )
    device = _resolve_device()
    model = build_dabqn_model(config).to(device)
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("ema_state_dict") or checkpoint.get("model_state_dict")
    if state_dict is None:
        raise RuntimeError("Checkpoint does not contain model_state_dict or ema_state_dict.")
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(f"checkpoint_load_warning missing={list(incompatible.missing_keys)} unexpected={list(incompatible.unexpected_keys)}")

    matcher = HungarianMatcher(
        cost_objectness=float(config["matcher"].get("cost_objectness", 1.0)),
        cost_box_l1=float(config["matcher"].get("cost_box_l1", 5.0)),
        cost_giou=float(config["matcher"].get("cost_giou", 2.0)),
        cost_mask=float(config["matcher"].get("cost_mask", 2.0)),
        cost_dice=float(config["matcher"].get("cost_dice", 5.0)),
        add_damage_cost=bool(config["matcher"].get("add_damage_cost", False)),
        cost_damage=float(config["matcher"].get("cost_damage", 0.0)),
    )
    criterion = DABQNLoss(config).to(device)
    result = evaluate_model(
        model=model,
        loader=loader,
        matcher=matcher,
        criterion=criterion,
        device=device,
        config=config,
        stage=stage,
        epoch=max(int(checkpoint.get("epoch", 1)) - 1, 0),
        max_batches=args.max_batches,
    )
    write_json(Path(save_dir) / "metrics_localization.json", result["localization"])
    write_json(Path(save_dir) / "metrics_matched_damage.json", result["matched_damage"])
    write_json(Path(save_dir) / "metrics_end_to_end_damage.json", result["end_to_end_damage"])
    write_json(Path(save_dir) / "metrics_pixel_bridge.json", result["pixel_bridge"])
    write_json(Path(save_dir) / "prediction_records.json", result["prediction_records"])
    print(f"localization_f1={result['localization']['localization_f1']:.4f}")
    print(f"matched_damage_macro_f1={result['matched_damage']['macro_f1']:.4f}")
    print(f"end_to_end_macro_f1={result['end_to_end_damage']['macro_f1']:.4f}")
    print(f"bridge_score={result['pixel_bridge']['xview2_overall_score']:.4f}")


if __name__ == "__main__":
    main()
