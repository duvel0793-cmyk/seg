from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from bridge import run_bridge_evaluation
from config import DEFAULT_CONFIG_PATH, apply_overrides, get_run_dir, get_split_list_path, load_config
from datasets.xbd_oracle_instance_damage import (
    CLASS_NAMES,
    XBDOracleInstanceDamageDataset,
    oracle_instance_collate_fn,
)
from engine import build_epoch_loss, resolve_amp_settings, resolve_device, run_epoch
from models import build_model
from utils.io import append_jsonl, ensure_dir, load_checkpoint, write_json, write_text, write_yaml
from utils.metrics import save_confusion_matrix_plot
from utils.seed import seed_worker, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the cleaned instance damage classification mainline.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["instance", "export", "bridge", "diagnostic_context"], default="instance")
    parser.add_argument("--split_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--train_list", type=str, default=None)
    parser.add_argument("--val_list", type=str, default=None)
    parser.add_argument("--test_list", type=str, default=None)
    parser.add_argument("--instance_source", type=str, default=None)
    parser.add_argument("--allow_tier3", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--input_mode", type=str, choices=["rgb", "rgbm"], default=None)
    parser.add_argument("--feature_dim", type=int, default=None)
    parser.add_argument("--token_count", type=int, default=None)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--amp_dtype", type=str, choices=["auto", "fp16", "bf16"], default=None)
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.set_defaults(amp=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    return parser.parse_args()


def _clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            cloned[key] = value.clone()
        elif isinstance(value, list):
            cloned[key] = list(value)
        elif isinstance(value, dict):
            cloned[key] = dict(value)
        else:
            cloned[key] = value
    return cloned


def build_diagnostic_batch_transform(mode_name: str):
    mode_name = str(mode_name)

    def _transform(_: int, batch: dict[str, Any]) -> dict[str, Any]:
        transformed = _clone_batch(batch)
        overrides: dict[str, bool] = {}
        if mode_name == "tight_only":
            overrides = {"use_context_branch": False, "use_neighborhood_scale": False}
        elif mode_name == "no_context":
            overrides = {"use_context_branch": False}
        elif mode_name == "no_neighborhood":
            overrides = {"use_neighborhood_scale": False}
        elif mode_name == "context_shuffle":
            batch_size = int(transformed["label"].size(0))
            permutation = torch.randperm(batch_size)
            keys = ["pre_context", "post_context", "mask_context"]
            for key in keys:
                if key in transformed and torch.is_tensor(transformed[key]):
                    transformed[key] = transformed[key][permutation]
        transformed["__forward_overrides__"] = overrides
        return transformed

    return _transform


def build_dataloader(
    dataset: XBDOracleInstanceDamageDataset,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=oracle_instance_collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def resolve_eval_config(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is not None:
        config = apply_overrides(copy.deepcopy(checkpoint_config), args, is_eval=True)
    else:
        config = apply_overrides(load_config(args.config), args, is_eval=True)
    return config


def save_instance_outputs(save_dir: Path, result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    report = result["report"]
    prediction_records = result["prediction_records"]

    write_json(save_dir / "metrics.json", metrics)
    write_text(save_dir / "classification_report.txt", report + "\n")
    write_json(save_dir / "confusion_matrix.json", metrics["confusion_matrix"])
    save_confusion_matrix_plot(metrics["confusion_matrix"], CLASS_NAMES, save_dir / "confusion_matrix.png")

    predictions_path = save_dir / "predictions.jsonl"
    if predictions_path.exists():
        predictions_path.unlink()
    for record in prediction_records:
        append_jsonl(predictions_path, record)


def main() -> None:
    args = parse_args()
    config = resolve_eval_config(args)
    set_seed(int(config["seed"]))

    run_dir = get_run_dir(config)
    save_dir = ensure_dir(Path(args.save_dir) if args.save_dir else run_dir / "eval" / args.mode)
    write_yaml(save_dir / "eval_config.yaml", config)

    split_name = args.split_name or config["eval"]["split_name"]
    list_path = get_split_list_path(config, split_name)
    dataset = XBDOracleInstanceDamageDataset(
        config=config,
        split_name=split_name,
        list_path=list_path,
        is_train=False,
    )
    loader = build_dataloader(
        dataset,
        batch_size=int(config["eval"]["batch_size"]),
        num_workers=int(config["eval"]["num_workers"]),
        seed=int(config["seed"]),
    )

    device = resolve_device()
    amp_settings = resolve_amp_settings(config, device)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Mode: {args.mode}")
    print(f"Save dir: {save_dir}")
    print(f"Ablation run: {config['ablation']['run_name']}")
    print(f"Backbone: ConvNeXtV2 ({config['model']['backbone_name']})")
    print(f"Alignment module: {config['model']['alignment_type']}")
    print("Scale branches: tight/context/neighborhood")
    print(f"Fusion: {'cross-scale attention' if bool(config['model']['use_cross_scale_attention']) else 'token mean'}")
    print("Head: CORN ordinal head")
    print("Silhouette branch: removed")
    print("Neighbor-aware branch: removed")
    print(
        "Token counts: "
        f"tight={int(config['model']['tight_token_count'])} "
        f"context={int(config['model']['context_token_count'])} "
        f"neighborhood={int(config['model']['neighborhood_token_count'])}"
    )
    print(f"Classifier type: {config['model']['classifier_type']}")
    print(f"Multi-context model: {bool(config['model'].get('use_multi_context_model', False))}")
    print(f"Neighborhood graph: {bool(config['model'].get('enable_neighborhood_graph', False))}")
    print(
        "Change suppression: "
        f"enabled={bool(config['model']['use_change_suppression'])} "
        f"pseudo={bool(config['model']['enable_pseudo_suppression'])} "
        f"fuse_to_tokens={bool(config['model']['fuse_change_to_tokens'])}"
    )
    print(f"Split: {split_name}")
    print(
        "AMP: "
        f"enabled={bool(amp_settings['enabled'])} "
        f"dtype={amp_settings['dtype_name']}"
    )

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    model = build_model(config).to(device)
    incompatible_keys = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
        raise RuntimeError(
            "Checkpoint is incompatible with the current model. "
            f"Missing keys: {list(incompatible_keys.missing_keys)}. "
            f"Unexpected keys: {list(incompatible_keys.unexpected_keys)}."
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

    save_instance_outputs(save_dir, result)
    metrics = result["metrics"]
    print(
        f"Instance metrics: "
        f"acc={metrics['overall_accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} "
        f"qwk={metrics['QWK']:.4f}"
    )
    print(
        "Aux metrics: "
        f"damage_aux_auc_tight={metrics.get('damage_aux_auc_tight') if metrics.get('damage_aux_auc_tight') is not None else 'n/a'} "
        f"damage_aux_auc_context={metrics.get('damage_aux_auc_context') if metrics.get('damage_aux_auc_context') is not None else 'n/a'} "
        f"damage_aux_auc_neighborhood={metrics.get('damage_aux_auc_neighborhood') if metrics.get('damage_aux_auc_neighborhood') is not None else 'n/a'} "
        f"severity_aux_mae={metrics.get('severity_aux_mae') if metrics.get('severity_aux_mae') is not None else 'n/a'}"
    )
    print(
        "Gate metrics: "
        f"tight_gap={metrics.get('tight_gate_gap') if metrics.get('tight_gate_gap') is not None else 'n/a'} "
        f"context_gap={metrics.get('context_gate_gap') if metrics.get('context_gate_gap') is not None else 'n/a'} "
        f"neighborhood_gap={metrics.get('neighborhood_gate_gap') if metrics.get('neighborhood_gate_gap') is not None else 'n/a'}"
    )

    if args.mode == "diagnostic_context":
        diagnostic_modes = [
            "full",
            "tight_only",
            "no_context",
            "no_neighborhood",
            "context_shuffle",
        ]
        diagnostics: dict[str, Any] = {}
        diagnostics["full"] = {
            "macro_f1": float(metrics["macro_f1"]),
            "per_class_f1": {name: float(values["f1"]) for name, values in metrics["per_class"].items()},
            "QWK": float(metrics["QWK"]),
            "far_error_rate": float(metrics["far_error_rate"]),
            "prediction_distribution": metrics["prediction_distribution"],
        }
        for mode_name in diagnostic_modes[1:]:
            diagnostic_result = run_epoch(
                model=model,
                loader=loader,
                criterion=criterion,
                device=device,
                config=config,
                optimizer=None,
                scaler=None,
                epoch_index=max(int(checkpoint.get("epoch", 1)) - 1, 0),
                collect_predictions=False,
                batch_transform=build_diagnostic_batch_transform(mode_name),
            )
            mode_metrics = diagnostic_result["metrics"]
            diagnostics[mode_name] = {
                "macro_f1": float(mode_metrics["macro_f1"]),
                "per_class_f1": {name: float(values["f1"]) for name, values in mode_metrics["per_class"].items()},
                "QWK": float(mode_metrics["QWK"]),
                "far_error_rate": float(mode_metrics["far_error_rate"]),
                "prediction_distribution": mode_metrics["prediction_distribution"],
            }
        diagnostics_dir = ensure_dir(Path(config["logging"]["output_root"]) / "diagnostics")
        write_json(diagnostics_dir / "context_ablation_metrics.json", diagnostics)
        print(f"Saved diagnostic context metrics to: {diagnostics_dir / 'context_ablation_metrics.json'}")
        return

    if args.mode == "instance":
        return

    export_dir = ensure_dir(save_dir / "exports")
    predictions_json = export_dir / "predictions.json"
    write_json(predictions_json, result["prediction_records"])
    write_text(
        export_dir / "README.txt",
        "\n".join(
            [
                "Instance prediction export",
                f"- split: {split_name}",
                f"- checkpoint: {Path(args.checkpoint).resolve()}",
                f"- predictions_jsonl: {(save_dir / 'predictions.jsonl').resolve()}",
                f"- predictions_json: {predictions_json.resolve()}",
                "- bridge consumes dataset GT polygons plus these per-instance predictions.",
            ]
        )
        + "\n",
    )
    print(f"Exported instance predictions to: {export_dir}")

    if args.mode == "export":
        return

    if not bool(config["ablation"]["keep_pixel_bridge_eval"]):
        raise RuntimeError("Bridge evaluation is disabled by config.ablation.keep_pixel_bridge_eval=false.")

    bridge_dir = ensure_dir(save_dir / config["bridge"]["output_subdir"])
    bridge_metrics, _ = run_bridge_evaluation(
        dataset,
        result["prediction_records"],
        config=config,
        save_dir=bridge_dir,
    )
    write_json(save_dir / "bridge_metrics.json", bridge_metrics)
    print(
        f"Bridge metrics: "
        f"localization_f1={bridge_metrics['localization_f1']:.4f} "
        f"damage_f1_hmean={bridge_metrics['damage_f1_hmean']:.4f} "
        f"overall={bridge_metrics['overall_xview2_style_score']:.4f}"
    )


if __name__ == "__main__":
    main()
