from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import XBDInstanceDataset, xbd_instance_collate_fn
from engine import build_epoch_loss, run_epoch
from models import build_model
from utils.bridge_eval import run_bridge_evaluation
from utils.checkpoint import load_checkpoint, resolve_checkpoint_model_state
from utils.misc import (
    append_jsonl,
    ensure_dir,
    get_split_list,
    load_config,
    merge_config,
    prediction_record_to_table_row,
    resolve_device,
    seed_worker,
    set_seed,
    write_csv_rows,
    write_json,
    write_text,
    write_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["instance", "export", "bridge"], default="instance")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--save-evidence-preview", action="store_true")
    parser.add_argument("--apply-best-threshold", type=str, choices=["none", "macro_f1", "qwk"], default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def build_dataloader(dataset: XBDInstanceDataset, *, batch_size: int, num_workers: int, seed: int) -> DataLoader:
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
    return DataLoader(
        dataset,
        **loader_kwargs,
    )


def write_prediction_exports(save_dir: Path, records: list[dict[str, object]]) -> None:
    predictions_path = save_dir / "predictions.jsonl"
    if predictions_path.exists():
        predictions_path.unlink()
    for record in records:
        append_jsonl(predictions_path, record)
    rows = [prediction_record_to_table_row(record) for record in records]
    write_csv_rows(
        save_dir / "predictions.csv",
        [
            "sample_id",
            "tile_id",
            "building_id",
            "gt_label",
            "corn_pred",
            "final_pred",
            "structural_pred",
            "scale_router_pred",
            "prob_no",
            "prob_minor",
            "prob_major",
            "prob_destroyed",
            "corn_prob_no",
            "corn_prob_minor",
            "corn_prob_major",
            "corn_prob_destroyed",
            "struct_prob_no",
            "struct_prob_minor",
            "struct_prob_major",
            "struct_prob_destroyed",
            "router_prob_no",
            "router_prob_minor",
            "router_prob_major",
            "router_prob_destroyed",
        ],
        rows,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    if "config" in checkpoint:
        checkpoint_config = checkpoint["config"]
        if isinstance(checkpoint_config, dict):
            config = merge_config(checkpoint_config)
    if args.apply_best_threshold is not None:
        config["eval"]["apply_best_damage_threshold"] = args.apply_best_threshold
    if str(config["eval"].get("apply_best_damage_threshold", "none")).lower() != "none":
        config["eval"]["damage_threshold_sweep"] = True
    set_seed(int(config["training"]["seed"]))
    save_dir = ensure_dir(Path(args.save_dir) if args.save_dir else Path(config["project"]["output_dir"]) / "eval" / args.mode)
    write_yaml(save_dir / "eval_config.yaml", config)

    dataset = XBDInstanceDataset(config=config, split=args.split, list_path=get_split_list(config, args.split), is_train=False)
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
        apply_best_damage_threshold=str(config["eval"].get("apply_best_damage_threshold", "none")),
        max_batches=args.max_batches,
    )
    write_json(save_dir / "metrics.json", result["metrics"])
    if result.get("threshold_sweep") is not None:
        write_json(save_dir / "threshold_sweep.json", result["threshold_sweep"])
    if result.get("diagnostics") is not None:
        write_json(save_dir / "diagnostics.json", result["diagnostics"])
    write_text(save_dir / "classification_report.txt", result["report"] + "\n")
    if args.mode in {"instance", "export", "bridge"}:
        write_prediction_exports(save_dir, result["prediction_records"])
    if args.mode in {"instance", "export"}:
        print(
            f"instance metrics acc={result['metrics']['accuracy']:.4f} "
            f"macro_f1={result['metrics']['macro_f1']:.4f} "
            f"qwk={result['metrics']['QWK']:.4f} "
            f"applied_threshold={result.get('applied_damage_threshold')}"
        )
    if args.mode == "bridge":
        bridge_summary, _ = run_bridge_evaluation(
            dataset,
            result["prediction_records"],
            config=config,
            save_dir=save_dir / "bridge",
        )
        global_metrics = bridge_summary["global"]
        print("instance:")
        print(
            f"  acc={result['metrics']['accuracy']:.4f} "
            f"macro_f1={result['metrics']['macro_f1']:.4f} "
            f"qwk={result['metrics']['QWK']:.4f} "
            f"far_error={result['metrics'].get('far_error', 0.0):.4f}"
        )
        print("bridge:")
        print(f"  localization_f1={global_metrics['localization_f1']:.4f}")
        print(f"  damage_f1_hmean={global_metrics['damage_f1_hmean']:.4f}")
        print(f"  xview2_overall_score={global_metrics['xview2_overall_score']:.4f}")
        print(f"  damage_macro_f1={global_metrics['damage_macro_f1']:.4f}")
        print(f"  damage_weighted_f1={global_metrics['damage_weighted_f1']:.4f}")
        print(f"  pixel_accuracy_gt_building={global_metrics['damage_pixel_accuracy_on_building_gt']:.4f}")
    if args.save_evidence_preview:
        print("save-evidence-preview requested, but preview export is minimal in this smoke-test implementation.")


if __name__ == "__main__":
    main()
