from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.xbd_oracle_instance_damage import (
    CLASS_NAMES,
    XBDOracleInstanceDamageDataset,
    oracle_instance_collate_fn,
)
from models import build_model
from utils.io import ensure_dir, load_checkpoint, read_yaml, write_json, write_text, write_yaml
from utils.losses import CORNLoss, DamageLossModule, build_loss_function
from utils.metrics import (
    compute_classification_metrics,
    compute_emd_from_probabilities,
    compute_ordinal_error_profile,
    save_confusion_matrix_plot,
    save_matrix_heatmap_plot,
)
from utils.seed import seed_worker, set_seed
from utils.visualize import save_prediction_visual

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_TYPE_CHOICES = ["post_only", "siamese_simple", "oracle_mcd", "oracle_mcd_corn"]
HEAD_TYPE_CHOICES = ["standard", "corn"]
LOSS_MODE_CHOICES = [
    "weighted_ce",
    "fixed_cda",
    "learnable_cda",
    "adaptive_ucl_cda",
    "adaptive_ucl_cda_v2",
    "adaptive_ucl_cda_v3",
    "corn",
]
STANDARD_ORACLE_LOSS_MODES = [
    "weighted_ce",
    "fixed_cda",
    "learnable_cda",
    "adaptive_ucl_cda",
    "adaptive_ucl_cda_v2",
    "adaptive_ucl_cda_v3",
]


def resolve_default_config_path() -> Path:
    preferred = PROJECT_ROOT / "configs" / "vscode_run.yaml"
    if preferred.exists():
        return preferred
    return PROJECT_ROOT / "configs" / "default.yaml"


DEFAULT_CONFIG_PATH = resolve_default_config_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate xBD Oracle Instance Damage Classification")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--checkpoint", type=str, default=None)

    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--val_list", type=str, default=None)
    parser.add_argument("--instance_source", type=str, default=None)
    parser.add_argument("--allow_tier3", action="store_true")
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--context_ratio", type=float, default=None)
    parser.add_argument("--min_polygon_area", type=float, default=None)
    parser.add_argument("--min_mask_pixels", type=int, default=None)
    parser.add_argument("--max_out_of_bound_ratio", type=float, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)

    parser.add_argument("--model_type", type=str, choices=MODEL_TYPE_CHOICES, default=None)
    parser.add_argument("--head_type", type=str, choices=HEAD_TYPE_CHOICES, default=None)
    parser.add_argument("--loss_mode", type=str, choices=LOSS_MODE_CHOICES, default=None)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--num_visuals_per_class", type=int, default=None)
    return parser.parse_args()


def ensure_training_defaults(config: dict[str, Any], default_loss_mode: str = "adaptive_ucl_cda_v3") -> dict[str, Any]:
    model_cfg = config.setdefault("model", {})
    model_cfg.setdefault("head_type", "standard")
    model_cfg.setdefault("ambiguity_hidden_features", 256)

    train_cfg = config.setdefault("training", {})
    train_cfg.setdefault("loss_mode", default_loss_mode)
    train_cfg.setdefault("lambda_ord", 0.20)
    train_cfg.setdefault("lambda_uni", 0.10)
    train_cfg.setdefault("lambda_conc", 0.01)
    train_cfg.setdefault("lambda_gap_reg", 1e-3)
    train_cfg.setdefault("lambda_tau", 0.01)
    train_cfg.setdefault("lambda_tau_mean", 0.005)
    train_cfg.setdefault("lambda_tau_diff", 0.03)
    train_cfg.setdefault("fixed_cda_alpha", 0.3)
    train_cfg.setdefault("tau_init", 0.22)
    train_cfg.setdefault("tau_min", 0.12)
    train_cfg.setdefault("tau_max", 0.45)
    train_cfg.setdefault("tau_base", 0.22)
    train_cfg.setdefault("delta_scale", 0.12)
    train_cfg.setdefault("tau_target", 0.22)
    train_cfg.setdefault("tau_easy", 0.16)
    train_cfg.setdefault("tau_hard", 0.32)
    train_cfg.setdefault("tau_freeze_epochs", 5)
    train_cfg.setdefault("tau_warmup_value", 0.22)
    train_cfg.setdefault("concentration_margin", 0.05)
    train_cfg.setdefault("lambda_aux", 0.2)
    train_cfg.setdefault("label_smoothing", 0.05)
    train_cfg.setdefault("use_focal", False)
    train_cfg.setdefault("focal_gamma", 2.0)
    return config


def harmonize_model_and_loss_config(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = config["model"]
    train_cfg = config["training"]

    model_type = str(model_cfg["model_type"])
    head_type = str(model_cfg.get("head_type", "standard"))
    loss_mode = str(train_cfg["loss_mode"])

    if model_type in {"post_only", "siamese_simple"}:
        if loss_mode in {"adaptive_ucl_cda", "adaptive_ucl_cda_v2", "adaptive_ucl_cda_v3", "corn"} or head_type == "corn":
            raise ValueError(f"{model_type} does not support loss_mode={loss_mode} / head_type={head_type}.")
        model_cfg["head_type"] = "standard"
        return config

    if model_type not in {"oracle_mcd", "oracle_mcd_corn"}:
        raise ValueError(f"Unsupported model_type='{model_type}'.")

    if model_type == "oracle_mcd_corn" or head_type == "corn" or loss_mode == "corn":
        model_cfg["model_type"] = "oracle_mcd_corn"
        model_cfg["head_type"] = "corn"
        train_cfg["loss_mode"] = "corn"
        return config

    model_cfg["model_type"] = "oracle_mcd"
    model_cfg["head_type"] = "standard"
    if loss_mode not in {
        "weighted_ce",
        "fixed_cda",
        "learnable_cda",
        "adaptive_ucl_cda",
        "adaptive_ucl_cda_v2",
        "adaptive_ucl_cda_v3",
    }:
        raise ValueError(f"Unsupported loss_mode='{loss_mode}' for oracle_mcd.")
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    return copy.deepcopy(read_yaml(path))


def apply_overrides(
    config: dict[str, Any],
    args: argparse.Namespace,
    default_loss_mode: str = "adaptive_ucl_cda_v3",
) -> dict[str, Any]:
    config = ensure_training_defaults(config, default_loss_mode=default_loss_mode)

    if args.seed is not None:
        config["seed"] = args.seed

    data_cfg = config["data"]
    if args.root_dir is not None:
        data_cfg["root_dir"] = args.root_dir
    if args.val_list is not None:
        data_cfg["val_list"] = args.val_list
    if args.instance_source is not None:
        data_cfg["instance_source"] = args.instance_source
    if args.allow_tier3:
        data_cfg["allow_tier3"] = True
    if args.image_size is not None:
        data_cfg["image_size"] = args.image_size
    if args.context_ratio is not None:
        data_cfg["context_ratio"] = args.context_ratio
    if args.min_polygon_area is not None:
        data_cfg["min_polygon_area"] = args.min_polygon_area
    if args.min_mask_pixels is not None:
        data_cfg["min_mask_pixels"] = args.min_mask_pixels
    if args.max_out_of_bound_ratio is not None:
        data_cfg["max_out_of_bound_ratio"] = args.max_out_of_bound_ratio
    if args.cache_dir is not None:
        data_cfg["cache_dir"] = args.cache_dir

    model_cfg = config["model"]
    if args.model_type is not None:
        model_cfg["model_type"] = args.model_type
    if args.head_type is not None:
        model_cfg["head_type"] = args.head_type
    if args.no_pretrained:
        model_cfg["pretrained"] = False

    train_cfg = config["training"]
    if args.loss_mode is not None:
        train_cfg["loss_mode"] = args.loss_mode
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        train_cfg["num_workers"] = args.num_workers
    if args.no_amp:
        train_cfg["amp"] = False

    output_cfg = config["output"]
    if args.output_root is not None:
        output_cfg["output_root"] = args.output_root
    if args.exp_name is not None:
        output_cfg["exp_name"] = args.exp_name

    if args.num_visuals_per_class is not None:
        config["evaluation"]["num_visuals_per_class"] = args.num_visuals_per_class
    return harmonize_model_and_loss_config(config)


def make_run_dir(config: dict[str, Any]) -> Path:
    return (
        Path(config["output"]["output_root"])
        / config["output"]["exp_name"]
        / config["model"]["model_type"]
        / config["training"]["loss_mode"]
    )


def resolve_checkpoint_path(config: dict[str, Any], explicit_checkpoint: str | None) -> Path:
    if explicit_checkpoint is not None:
        return Path(explicit_checkpoint)
    return make_run_dir(config) / "checkpoints" / "best_macro_f1.pth"


def make_dataloader(config: dict[str, Any]) -> tuple[DataLoader, XBDOracleInstanceDamageDataset]:
    dataset = XBDOracleInstanceDamageDataset(
        config=config,
        split_name=config["evaluation"]["split_name"],
        list_path=config["data"]["val_list"],
        is_train=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        collate_fn=oracle_instance_collate_fn,
        worker_init_fn=seed_worker,
        persistent_workers=int(config["training"]["num_workers"]) > 0,
    )
    return loader, dataset


def build_loss_module(config: dict[str, Any], checkpoint: dict[str, Any], device: torch.device) -> DamageLossModule:
    checkpoint_weights = checkpoint.get("class_weights")
    class_weights = None
    if checkpoint_weights is not None:
        if isinstance(checkpoint_weights, torch.Tensor):
            class_weights = checkpoint_weights.float()
        else:
            class_weights = torch.tensor(checkpoint_weights, dtype=torch.float32)

    criterion = build_loss_function(
        class_weights=class_weights,
        loss_mode=str(config["training"]["loss_mode"]),
        label_smoothing=float(config["training"]["label_smoothing"]),
        lambda_ord=float(config["training"]["lambda_ord"]),
        lambda_uni=float(config["training"].get("lambda_uni", 0.10)),
        lambda_conc=float(config["training"].get("lambda_conc", 0.01)),
        lambda_gap_reg=float(config["training"]["lambda_gap_reg"]),
        lambda_tau=float(config["training"].get("lambda_tau", 0.01)),
        lambda_tau_mean=float(config["training"].get("lambda_tau_mean", 0.005)),
        lambda_tau_diff=float(config["training"].get("lambda_tau_diff", 0.03)),
        fixed_cda_alpha=float(config["training"]["fixed_cda_alpha"]),
        tau_init=float(config["training"]["tau_init"]),
        tau_min=float(config["training"].get("tau_min", 0.10)),
        tau_max=float(config["training"].get("tau_max", 0.60)),
        tau_base=float(config["training"].get("tau_base", config["training"].get("tau_target", 0.22))),
        delta_scale=float(config["training"].get("delta_scale", 0.12)),
        tau_target=float(config["training"].get("tau_target", 0.22)),
        tau_easy=float(config["training"].get("tau_easy", 0.16)),
        tau_hard=float(config["training"].get("tau_hard", 0.32)),
        concentration_margin=float(config["training"].get("concentration_margin", 0.05)),
        lambda_aux=float(config["training"].get("lambda_aux", 0.2)),
        use_focal=bool(config["training"].get("use_focal", False)),
        focal_gamma=float(config["training"].get("focal_gamma", 2.0)),
        num_classes=len(CLASS_NAMES),
    )
    if "loss_state_dict" in checkpoint:
        criterion.load_state_dict(checkpoint["loss_state_dict"], strict=False)
    criterion = criterion.to(device)
    criterion.eval()
    return criterion


def unpack_model_outputs(model_output: Any) -> dict[str, Any]:
    if isinstance(model_output, torch.Tensor):
        return {
            "logits": model_output,
            "aux_logits": None,
            "pooled_feature": None,
            "tau": None,
            "raw_tau": None,
            "raw_delta_tau": None,
            "head_type": "standard",
        }
    if not isinstance(model_output, dict) or "logits" not in model_output:
        raise TypeError("Model forward must return a logits tensor or a dict containing 'logits'.")
    return {
        "logits": model_output["logits"],
        "aux_logits": model_output.get("aux_logits"),
        "pooled_feature": model_output.get("pooled_feature"),
        "tau": model_output.get("tau"),
        "raw_tau": model_output.get("raw_tau"),
        "raw_delta_tau": model_output.get("raw_delta_tau"),
        "head_type": model_output.get("head_type", "standard"),
    }


def summarize_distribution_values(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    value_array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(value_array)),
        "std": float(np.std(value_array)),
        "min": float(np.min(value_array)),
        "max": float(np.max(value_array)),
        "p10": float(np.percentile(value_array, 10)),
        "p50": float(np.percentile(value_array, 50)),
        "p90": float(np.percentile(value_array, 90)),
        "support": int(value_array.size),
    }


def summarize_tau_values(values: list[float]) -> dict[str, float] | None:
    return summarize_distribution_values(values)


def compute_scalar_correlation(first: list[float], second: list[float]) -> float | None:
    if not first or not second or len(first) != len(second):
        return None
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if first_array.size < 2:
        return 0.0
    first_centered = first_array - np.mean(first_array)
    second_centered = second_array - np.mean(second_array)
    denominator = float(np.sqrt(np.sum(first_centered ** 2) * np.sum(second_centered ** 2)))
    if denominator <= 1e-12:
        return 0.0
    return float(np.sum(first_centered * second_centered) / denominator)


def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: DamageLossModule,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[list[int], list[int], list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    criterion.eval()
    all_targets: list[int] = []
    all_predictions: list[int] = []
    prediction_records: list[dict[str, Any]] = []
    tau_values: list[float] = []
    difficulty_values: list[float] = []

    positions = criterion.get_current_positions().to(device=device, dtype=torch.float32)

    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc="eval"):
            pre_image = batch["pre_image"].to(device, non_blocking=True)
            post_image = batch["post_image"].to(device, non_blocking=True)
            instance_mask = batch["instance_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            context = torch.autocast(device_type=device.type, dtype=torch.float16) if amp_enabled else nullcontext()
            with context:
                model_outputs = unpack_model_outputs(model(pre_image, post_image, instance_mask))
                logits = model_outputs["logits"]
                if criterion.loss_mode == "corn" or model_outputs["head_type"] == "corn":
                    threshold_probabilities = CORNLoss.logits_to_threshold_probabilities(logits)
                    probabilities = CORNLoss.logits_to_class_probabilities(logits)
                else:
                    threshold_probabilities = None
                    probabilities = logits.softmax(dim=1)

            predictions = probabilities.argmax(dim=1)
            confidences = probabilities.max(dim=1).values
            true_probs = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
            difficulty = 1.0 - true_probs
            expected_severity = (probabilities.float() * positions.unsqueeze(0)).sum(dim=1)
            gt_severity = positions[labels]
            severity_abs_error = (expected_severity - gt_severity).abs()

            labels_cpu = labels.cpu().tolist()
            predictions_cpu = predictions.cpu().tolist()
            confidences_cpu = confidences.cpu().tolist()
            true_probs_cpu = true_probs.cpu().tolist()
            difficulty_cpu = difficulty.cpu().tolist()
            expected_severity_cpu = expected_severity.cpu().tolist()
            gt_severity_cpu = gt_severity.cpu().tolist()
            severity_abs_error_cpu = severity_abs_error.cpu().tolist()
            probabilities_cpu = probabilities.cpu().tolist()
            threshold_probs_cpu = None if threshold_probabilities is None else threshold_probabilities.cpu().tolist()
            tau_cpu = None if model_outputs["tau"] is None else model_outputs["tau"].detach().cpu().tolist()
            raw_delta_tau_cpu = (
                None
                if model_outputs["raw_delta_tau"] is None
                else model_outputs["raw_delta_tau"].detach().cpu().tolist()
            )
            sample_indices = batch["sample_index"].cpu().tolist()
            meta_list = batch["meta"]

            if tau_cpu is not None:
                tau_values.extend(float(value) for value in tau_cpu)
                difficulty_values.extend(float(value) for value in difficulty_cpu)

            all_targets.extend(labels_cpu)
            all_predictions.extend(predictions_cpu)
            for row_idx, (
                sample_index,
                gt,
                pred,
                conf,
                true_prob,
                sample_difficulty,
                expected_value,
                gt_value,
                abs_error,
                probs,
                meta,
            ) in enumerate(
                zip(
                    sample_indices,
                    labels_cpu,
                    predictions_cpu,
                    confidences_cpu,
                    true_probs_cpu,
                    difficulty_cpu,
                    expected_severity_cpu,
                    gt_severity_cpu,
                    severity_abs_error_cpu,
                    probabilities_cpu,
                    meta_list,
                )
            ):
                threshold_record = None if threshold_probs_cpu is None else [float(value) for value in threshold_probs_cpu[row_idx]]
                tau_value = None if tau_cpu is None else float(tau_cpu[row_idx])
                raw_delta_tau_value = None if raw_delta_tau_cpu is None else float(raw_delta_tau_cpu[row_idx])
                prediction_records.append(
                    {
                        "sample_index": int(sample_index),
                        "gt": int(gt),
                        "pred": int(pred),
                        "confidence": float(conf),
                        "true_probability": float(true_prob),
                        "difficulty": float(sample_difficulty),
                        "probabilities": [float(value) for value in probs],
                        "threshold_probabilities": threshold_record,
                        "tau": tau_value,
                        "raw_delta_tau": raw_delta_tau_value,
                        "expected_severity": float(expected_value),
                        "gt_severity": float(gt_value),
                        "severity_abs_error": float(abs_error),
                        "meta": meta,
                    }
                )
    return all_targets, all_predictions, prediction_records, {
        "tau_stats": summarize_tau_values(tau_values),
        "difficulty_stats": summarize_distribution_values(difficulty_values),
        "corr_tau_difficulty": compute_scalar_correlation(tau_values, difficulty_values),
    }


def save_prediction_records(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def select_top_records(records: list[dict[str, Any]], class_idx: int, correct: bool, top_k: int) -> list[dict[str, Any]]:
    selected = [record for record in records if record["gt"] == class_idx and ((record["pred"] == class_idx) == correct)]
    if correct:
        selected.sort(key=lambda item: item["confidence"], reverse=True)
    else:
        selected.sort(key=lambda item: (item["confidence"], 1.0 - item["true_probability"]), reverse=True)
    return selected[:top_k]


def save_visual_examples(
    dataset: XBDOracleInstanceDamageDataset,
    records: list[dict[str, Any]],
    eval_dir: Path,
    num_visuals_per_class: int,
) -> None:
    correct_dir = ensure_dir(eval_dir / "visuals" / "correct")
    wrong_dir = ensure_dir(eval_dir / "visuals" / "wrong")

    for class_idx, class_name in enumerate(CLASS_NAMES):
        correct_records = select_top_records(records, class_idx, correct=True, top_k=num_visuals_per_class)
        wrong_records = select_top_records(records, class_idx, correct=False, top_k=num_visuals_per_class)

        for rank, record in enumerate(correct_records, start=1):
            sample = dataset[record["sample_index"]]
            save_prediction_visual(
                save_path=correct_dir / class_name / f"{rank:02d}_{record['meta']['tile_id']}_{record['meta']['building_idx']}.png",
                pre_image=sample["pre_image"],
                post_image=sample["post_image"],
                mask=sample["instance_mask"],
                gt_label=CLASS_NAMES[record["gt"]],
                pred_label=CLASS_NAMES[record["pred"]],
                confidence=record["confidence"],
                meta=record["meta"],
                extra_lines=[f"expected={record['expected_severity']:.4f} gt_severity={record['gt_severity']:.4f}"],
            )

        for rank, record in enumerate(wrong_records, start=1):
            sample = dataset[record["sample_index"]]
            extra_lines = [
                f"expected={record['expected_severity']:.4f} gt_severity={record['gt_severity']:.4f}",
                f"severity_abs_error={record['severity_abs_error']:.4f}",
                f"difficulty={record['difficulty']:.4f}",
            ]
            if record["tau"] is not None:
                extra_lines.append(f"tau={record['tau']:.4f}")
            save_prediction_visual(
                save_path=wrong_dir / class_name / f"{rank:02d}_{record['meta']['tile_id']}_{record['meta']['building_idx']}.png",
                pre_image=sample["pre_image"],
                post_image=sample["post_image"],
                mask=sample["instance_mask"],
                gt_label=CLASS_NAMES[record["gt"]],
                pred_label=CLASS_NAMES[record["pred"]],
                confidence=record["confidence"],
                meta=record["meta"],
                extra_lines=extra_lines,
            )


def save_hardest_minor_major_visuals(
    dataset: XBDOracleInstanceDamageDataset,
    records: list[dict[str, Any]],
    eval_dir: Path,
    top_k: int,
) -> None:
    candidates = [
        record
        for record in records
        if record["gt"] != record["pred"] and {record["gt"], record["pred"]} == {1, 2}
    ]
    candidates.sort(
        key=lambda item: (
            item["confidence"],
            1.0 - item["true_probability"],
            item["severity_abs_error"],
        ),
        reverse=True,
    )
    selected = candidates[:top_k]
    output_dir = ensure_dir(eval_dir / "visuals" / "hardest_minor_major")
    summary_rows = []
    for rank, record in enumerate(selected, start=1):
        sample = dataset[record["sample_index"]]
        threshold_text = (
            "threshold_probs=" + ",".join(f"{value:.3f}" for value in record["threshold_probabilities"])
            if record["threshold_probabilities"] is not None
            else None
        )
        extra_lines = [
            f"expected={record['expected_severity']:.4f} gt_severity={record['gt_severity']:.4f}",
            f"severity_abs_error={record['severity_abs_error']:.4f}",
            f"difficulty={record['difficulty']:.4f}",
        ]
        if record["tau"] is not None:
            extra_lines.append(f"tau={record['tau']:.4f}")
        if threshold_text is not None:
            extra_lines.append(threshold_text)
        save_prediction_visual(
            save_path=output_dir / f"{rank:02d}_{record['meta']['tile_id']}_{record['meta']['building_idx']}.png",
            pre_image=sample["pre_image"],
            post_image=sample["post_image"],
            mask=sample["instance_mask"],
            gt_label=CLASS_NAMES[record["gt"]],
            pred_label=CLASS_NAMES[record["pred"]],
            confidence=record["confidence"],
            meta=record["meta"],
            extra_lines=extra_lines,
        )
        summary_rows.append(
            {
                "rank": rank,
                "tile_id": record["meta"]["tile_id"],
                "building_idx": record["meta"]["building_idx"],
                "gt": CLASS_NAMES[record["gt"]],
                "pred": CLASS_NAMES[record["pred"]],
                "confidence": record["confidence"],
                "true_probability": record["true_probability"],
                "severity_abs_error": record["severity_abs_error"],
                "tau": record["tau"],
            }
        )

    write_json(eval_dir / "hardest_minor_major_confusions.json", summary_rows)
    lines = ["Hardest minor-major confusions"]
    for row in summary_rows:
        lines.append(
            f"- rank={row['rank']} tile={row['tile_id']} building={row['building_idx']} "
            f"gt={row['gt']} pred={row['pred']} conf={row['confidence']:.4f} "
            f"true_prob={row['true_probability']:.4f} severity_abs_error={row['severity_abs_error']:.4f} "
            f"tau={row['tau'] if row['tau'] is not None else 'n/a'}"
        )
    write_text(eval_dir / "hardest_minor_major_confusions.txt", "\n".join(lines) + "\n")


def find_top_confusions(confusion: list[list[int]]) -> tuple[str, int]:
    cm = np.asarray(confusion)
    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)
    if off_diag.sum() == 0:
        return "No cross-class confusion observed.", 0
    index = np.unravel_index(np.argmax(off_diag), off_diag.shape)
    return f"{CLASS_NAMES[index[0]]} -> {CLASS_NAMES[index[1]]}", int(off_diag[index])


def save_ordinal_exports(eval_dir: Path, criterion: DamageLossModule) -> dict[str, Any]:
    ordinal_state = criterion.export_state(CLASS_NAMES)
    write_json(
        eval_dir / "ordinal_positions.json",
        {
            "class_names": CLASS_NAMES,
            "loss_mode": ordinal_state["loss_mode"],
            "positions": ordinal_state["positions_by_class"],
            "positions_list": ordinal_state["positions"],
            "gaps": ordinal_state["gaps"],
            "tau": ordinal_state["tau"],
            "tau_statistics": ordinal_state["tau_statistics"],
            "difficulty_statistics": ordinal_state.get("difficulty_statistics"),
            "corr_tau_difficulty": ordinal_state.get("corr_tau_difficulty"),
            "tau_bounds": ordinal_state["tau_bounds"],
            "fixed_cda_alpha": ordinal_state["fixed_cda_alpha"],
            "tau_regularizer": ordinal_state.get("tau_regularizer"),
        },
    )

    csv_path = eval_dir / "ordinal_soft_targets.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gt_class", *CLASS_NAMES])
        for class_name, row in zip(CLASS_NAMES, ordinal_state["soft_target_matrix"]):
            writer.writerow([class_name, *[f"{float(value):.8f}" for value in row]])

    save_matrix_heatmap_plot(
        matrix=np.asarray(ordinal_state["soft_target_matrix"], dtype=np.float32),
        row_labels=CLASS_NAMES,
        col_labels=CLASS_NAMES,
        save_path=eval_dir / "ordinal_soft_targets.png",
        title="Ordinal Soft Targets (Reference Matrix)",
        cmap="magma",
        value_format=".3f",
        xlabel="Prediction Class",
        ylabel="Ground Truth Class",
    )
    return ordinal_state


def build_adjacency_confusion_report(metrics: dict[str, Any]) -> dict[str, Any]:
    cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    supports = cm.sum(axis=1)
    pair_specs = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]
    pairs: dict[str, dict[str, float | int]] = {}
    for src, dst in pair_specs:
        key = f"{CLASS_NAMES[src]}->{CLASS_NAMES[dst]}"
        support = int(supports[src])
        count = int(cm[src, dst])
        pairs[key] = {
            "count": count,
            "support": support,
            "rate_within_gt_class": float(count / support) if support > 0 else 0.0,
        }

    return {
        "pairs": pairs,
        "summary": {
            "minor_major_bidirectional": int(cm[1, 2] + cm[2, 1]),
            "no_minor_bidirectional": int(cm[0, 1] + cm[1, 0]),
            "major_destroyed_bidirectional": int(cm[2, 3] + cm[3, 2]),
            "adjacent_bidirectional_total": int(sum(pairs[key]["count"] for key in pairs)),
        },
    }


def write_adjacency_confusion_report(eval_dir: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    report = build_adjacency_confusion_report(metrics)
    write_json(eval_dir / "adjacency_confusion.json", report)

    lines = ["Adjacency confusion report"]
    for key in [
        "no-damage->minor-damage",
        "minor-damage->no-damage",
        "minor-damage->major-damage",
        "major-damage->minor-damage",
        "major-damage->destroyed",
        "destroyed->major-damage",
    ]:
        payload = report["pairs"][key]
        lines.append(
            f"- {key}: count={payload['count']} support={payload['support']} rate={payload['rate_within_gt_class']:.4f}"
        )
    lines.append("")
    lines.append(f"- minor_major_bidirectional={report['summary']['minor_major_bidirectional']}")
    lines.append(f"- major_destroyed_bidirectional={report['summary']['major_destroyed_bidirectional']}")
    write_text(eval_dir / "adjacency_confusion.txt", "\n".join(lines) + "\n")
    return report


def build_severity_error_report(records: list[dict[str, Any]], ordinal_state: dict[str, Any]) -> dict[str, Any]:
    overall_errors = [float(record["severity_abs_error"]) for record in records]
    per_class: dict[str, dict[str, Any]] = {}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_records = [record for record in records if record["gt"] == class_idx]
        class_errors = [float(record["severity_abs_error"]) for record in class_records]
        class_expected = [float(record["expected_severity"]) for record in class_records]
        class_gt = [float(record["gt_severity"]) for record in class_records]
        per_class[class_name] = {
            "mae": float(np.mean(class_errors)) if class_errors else 0.0,
            "mean_expected_severity": float(np.mean(class_expected)) if class_expected else 0.0,
            "mean_gt_severity": float(np.mean(class_gt)) if class_gt else 0.0,
            "support": len(class_records),
        }

    return {
        "loss_mode": ordinal_state["loss_mode"],
        "mean_absolute_severity_error": float(np.mean(overall_errors)) if overall_errors else 0.0,
        "positions": ordinal_state["positions_by_class"],
        "per_class": per_class,
    }


def write_qwk_report(eval_dir: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "quadratic_weighted_kappa": float(metrics["quadratic_weighted_kappa"]),
        "num_instances": int(metrics["num_instances"]),
        "model_type": metrics["model_type"],
        "loss_mode": metrics["loss_mode"],
    }
    write_json(eval_dir / "qwk.json", payload)
    return payload


def write_emd_report(
    eval_dir: Path,
    records: list[dict[str, Any]],
    positions: list[float],
) -> dict[str, Any]:
    probabilities = np.asarray([record["probabilities"] for record in records], dtype=np.float64)
    y_true = np.asarray([record["gt"] for record in records], dtype=np.int64)
    mean_emd, sample_emd = compute_emd_from_probabilities(probabilities, y_true, positions)

    for record, emd_value in zip(records, sample_emd.tolist()):
        record["emd_severity"] = float(emd_value)

    per_class: dict[str, dict[str, Any]] = {}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        mask = y_true == class_idx
        values = sample_emd[mask]
        per_class[class_name] = {
            "support": int(mask.sum()),
            "mean_emd": float(values.mean()) if values.size else 0.0,
        }

    payload = {
        "emd_severity": float(mean_emd),
        "positions": {class_name: float(position) for class_name, position in zip(CLASS_NAMES, positions)},
        "per_class": per_class,
    }
    write_json(eval_dir / "emd.json", payload)
    return payload


def write_ordinal_error_profile(eval_dir: Path, y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    profile = compute_ordinal_error_profile(y_true, y_pred, CLASS_NAMES)
    write_json(eval_dir / "ordinal_error_profile.json", profile)
    lines = [
        "Ordinal error profile",
        f"- exact_match_rate={profile['exact_match_rate']:.4f}",
        f"- adjacent_error_rate={profile['adjacent_error_rate']:.4f}",
        f"- far_error_rate={profile['far_error_rate']:.4f}",
        f"- adjacent_error_share_among_errors={profile['adjacent_error_share_among_errors']:.4f}",
        f"- far_error_share_among_errors={profile['far_error_share_among_errors']:.4f}",
        f"- mean_absolute_class_distance={profile['mean_absolute_class_distance']:.4f}",
        "",
        "Distance histogram",
    ]
    for distance, count in profile["distance_histogram"].items():
        lines.append(f"- |pred-gt|={distance}: count={count}")
    lines.append("")
    lines.append("Per-class ordinal error")
    for class_name in CLASS_NAMES:
        payload = profile["per_class"][class_name]
        lines.append(
            f"- {class_name}: support={payload['support']} adjacent_rate={payload['adjacent_error_rate']:.4f} "
            f"far_rate={payload['far_error_rate']:.4f} mean_abs_distance={payload['mean_absolute_distance']:.4f}"
        )
    write_text(eval_dir / "ordinal_error_profile.txt", "\n".join(lines) + "\n")
    return profile


def write_tau_statistics(eval_dir: Path, inference_summary: dict[str, Any], ordinal_state: dict[str, Any]) -> dict[str, Any]:
    tau_stats = inference_summary.get("tau_stats")
    payload = {
        "inference_tau_statistics": tau_stats,
        "inference_difficulty_statistics": inference_summary.get("difficulty_stats"),
        "corr_tau_difficulty": inference_summary.get("corr_tau_difficulty"),
        "criterion_tau_statistics": ordinal_state.get("tau_statistics"),
        "criterion_difficulty_statistics": ordinal_state.get("difficulty_statistics"),
        "criterion_corr_tau_difficulty": ordinal_state.get("corr_tau_difficulty"),
        "tau_reference": ordinal_state.get("tau"),
        "tau_bounds": ordinal_state.get("tau_bounds"),
        "collapsed_near_constant": bool(tau_stats is not None and tau_stats["std"] < 1e-4),
    }
    write_json(eval_dir / "tau_stats.json", payload)
    return payload


def save_empty_tau_plot(save_path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_tau_histogram(eval_dir: Path, tau_values: list[float]) -> None:
    save_path = eval_dir / "tau_histogram.png"
    if not tau_values:
        save_empty_tau_plot(save_path, "Tau Histogram", "No tau values available")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.asarray(tau_values, dtype=np.float64), bins=24, color="#2b6cb0", alpha=0.85, edgecolor="white")
    ax.set_title("Adaptive Tau Histogram")
    ax.set_xlabel("tau")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_tau_boxplot(eval_dir: Path, tau_groups: dict[str, list[float]]) -> None:
    save_path = eval_dir / "tau_boxplot.png"
    valid_groups = [(name, values) for name, values in tau_groups.items() if values]
    if not valid_groups:
        save_empty_tau_plot(save_path, "Tau Boxplot", "No tau values available")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot([values for _, values in valid_groups], labels=[name for name, _ in valid_groups], patch_artist=True)
    ax.set_title("Adaptive Tau by GT Class")
    ax.set_xlabel("GT class")
    ax.set_ylabel("tau")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_tau_difficulty_scatter(eval_dir: Path, tau_values: list[float], difficulty_values: list[float]) -> None:
    save_path = eval_dir / "tau_vs_difficulty_scatter.png"
    if not tau_values or not difficulty_values or len(tau_values) != len(difficulty_values):
        save_empty_tau_plot(save_path, "Tau vs Difficulty", "No tau/difficulty pairs available")
        return

    x = np.asarray(difficulty_values, dtype=np.float64)
    y = np.asarray(tau_values, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.scatter(x, y, s=14, alpha=0.35, color="#c05621", edgecolors="none")
    if x.size >= 2 and np.std(x) > 1e-8:
        coeffs = np.polyfit(x, y, deg=1)
        trend_x = np.linspace(float(x.min()), float(x.max()), 100)
        trend_y = coeffs[0] * trend_x + coeffs[1]
        ax.plot(trend_x, trend_y, color="#1a202c", linewidth=2.0)
    ax.set_title("Adaptive Tau vs Difficulty")
    ax.set_xlabel("difficulty = 1 - p(gt)")
    ax.set_ylabel("tau")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_tau_group_payload(records: list[dict[str, Any]], key_fn) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[float]] = {}
    for record in records:
        tau = record.get("tau")
        if tau is None:
            continue
        group_key = str(key_fn(record))
        groups.setdefault(group_key, []).append(float(tau))

    payload: dict[str, dict[str, Any]] = {}
    for key, values in groups.items():
        payload[key] = summarize_tau_values(values) or {"support": 0}
    return payload


def export_tau_diagnostics(eval_dir: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    tau_values = [float(record["tau"]) for record in records if record.get("tau") is not None]
    difficulty_values = [float(record["difficulty"]) for record in records if record.get("tau") is not None]
    tau_by_gt_class = {
        class_name: summarize_tau_values(
            [float(record["tau"]) for record in records if record.get("tau") is not None and record["gt"] == class_idx]
        ) or {"support": 0}
        for class_idx, class_name in enumerate(CLASS_NAMES)
    }
    tau_by_correctness = {
        "correct": summarize_tau_values(
            [float(record["tau"]) for record in records if record.get("tau") is not None and record["gt"] == record["pred"]]
        ) or {"support": 0},
        "wrong": summarize_tau_values(
            [float(record["tau"]) for record in records if record.get("tau") is not None and record["gt"] != record["pred"]]
        ) or {"support": 0},
    }

    write_json(eval_dir / "tau_by_gt_class.json", tau_by_gt_class)
    write_json(eval_dir / "tau_by_correctness.json", tau_by_correctness)
    save_tau_histogram(eval_dir, tau_values)
    save_tau_boxplot(
        eval_dir,
        {
            class_name: [float(record["tau"]) for record in records if record.get("tau") is not None and record["gt"] == class_idx]
            for class_idx, class_name in enumerate(CLASS_NAMES)
        },
    )
    tau_difficulty_points = [
        {
            "sample_index": int(record["sample_index"]),
            "gt": int(record["gt"]),
            "pred": int(record["pred"]),
            "correct": bool(record["gt"] == record["pred"]),
            "tau": float(record["tau"]),
            "difficulty": float(record["difficulty"]),
        }
        for record in records
        if record.get("tau") is not None
    ]
    tau_vs_difficulty_payload = {
        "summary": {
            "tau_stats": summarize_tau_values(tau_values),
            "difficulty_stats": summarize_distribution_values(difficulty_values),
            "corr_tau_difficulty": compute_scalar_correlation(tau_values, difficulty_values),
            "support": len(tau_difficulty_points),
        },
        "points": tau_difficulty_points,
    }
    write_json(eval_dir / "tau_vs_difficulty.json", tau_vs_difficulty_payload)
    save_tau_difficulty_scatter(eval_dir, tau_values, difficulty_values)
    return {
        "all_tau_values": tau_values,
        "all_difficulty_values": difficulty_values,
        "tau_by_gt_class": tau_by_gt_class,
        "tau_by_correctness": tau_by_correctness,
        "tau_vs_difficulty": tau_vs_difficulty_payload,
    }


def write_adaptive_tau_analysis(
    eval_dir: Path,
    loss_mode: str,
    tau_report: dict[str, Any],
    tau_diag: dict[str, Any],
) -> None:
    overall = tau_report.get("inference_tau_statistics")
    lines = ["# Adaptive Tau Analysis", ""]

    if overall is None or loss_mode not in {"adaptive_ucl_cda", "adaptive_ucl_cda_v2", "adaptive_ucl_cda_v3"}:
        lines.append("- Current run does not expose adaptive sample-level tau diagnostics.")
        write_text(eval_dir / "adaptive_tau_analysis.md", "\n".join(lines) + "\n")
        return

    tau_difficulty_summary = tau_diag.get("tau_vs_difficulty", {}).get("summary", {})
    lines.extend(
        [
            "## Overall",
            (
                f"- mean={overall['mean']:.4f}, std={overall['std']:.6f}, "
                f"min={overall['min']:.4f}, p10={overall['p10']:.4f}, "
                f"p50={overall['p50']:.4f}, p90={overall['p90']:.4f}, max={overall['max']:.4f}"
            ),
        ]
    )
    if overall["std"] < 1e-4:
        lines.append("- adaptive tau remains near-constant and is effectively collapsed")
    else:
        lines.append("- adaptive tau remains non-constant across evaluation samples")

    corr_tau_difficulty = tau_difficulty_summary.get("corr_tau_difficulty")
    if corr_tau_difficulty is None:
        lines.append("- tau vs difficulty correlation: unavailable")
    elif corr_tau_difficulty > 0.10:
        lines.append(f"- tau is positively correlated with difficulty (corr={corr_tau_difficulty:.4f})")
    elif corr_tau_difficulty < -0.10:
        lines.append(f"- tau is negatively correlated with difficulty (corr={corr_tau_difficulty:.4f})")
    else:
        lines.append(f"- tau shows only weak correlation with difficulty (corr={corr_tau_difficulty:.4f})")

    lines.extend(["", "## By Correctness"])
    tau_by_correctness = tau_diag["tau_by_correctness"]
    for key in ["correct", "wrong"]:
        payload = tau_by_correctness.get(key, {"support": 0})
        if payload.get("support", 0) <= 0:
            lines.append(f"- {key}: no samples")
            continue
        lines.append(
            f"- {key}: support={payload['support']} mean={payload['mean']:.4f} std={payload['std']:.4f} "
            f"p10={payload['p10']:.4f} p50={payload['p50']:.4f} p90={payload['p90']:.4f}"
        )

    correct_payload = tau_by_correctness.get("correct")
    wrong_payload = tau_by_correctness.get("wrong")
    if correct_payload and wrong_payload and correct_payload.get("support", 0) > 0 and wrong_payload.get("support", 0) > 0:
        mean_gap = float(wrong_payload["mean"]) - float(correct_payload["mean"])
        abs_gap = abs(mean_gap)
        if mean_gap > 0.02:
            lines.append(f"- wrong samples have clearly larger tau on average (mean gap={mean_gap:.4f})")
        elif mean_gap > 0.0:
            lines.append(f"- wrong samples have slightly larger tau on average (mean gap={mean_gap:.4f})")
        elif abs_gap > 0.02:
            lines.append(f"- wrong samples are not larger; tau separates in the opposite direction (mean gap={mean_gap:.4f})")
        else:
            lines.append(f"- tau shows only weak correctness-dependent separation (mean gap={mean_gap:.4f})")

    lines.extend(["", "## By GT Class"])
    tau_by_gt_class = tau_diag["tau_by_gt_class"]
    gt_means: list[float] = []
    for class_name in CLASS_NAMES:
        payload = tau_by_gt_class.get(class_name, {"support": 0})
        if payload.get("support", 0) <= 0:
            lines.append(f"- {class_name}: no samples")
            continue
        gt_means.append(float(payload["mean"]))
        lines.append(
            f"- {class_name}: support={payload['support']} mean={payload['mean']:.4f} std={payload['std']:.4f} "
            f"p10={payload['p10']:.4f} p50={payload['p50']:.4f} p90={payload['p90']:.4f}"
        )
    if gt_means:
        class_spread = max(gt_means) - min(gt_means)
        if class_spread > 0.02:
            lines.append(f"- tau varies meaningfully across GT classes (mean spread={class_spread:.4f})")
        else:
            lines.append(f"- tau varies only weakly across GT classes (mean spread={class_spread:.4f})")

    write_text(eval_dir / "adaptive_tau_analysis.md", "\n".join(lines) + "\n")


def load_loss_mode_reports(current_run_dir: Path) -> dict[str, dict[str, Any]]:
    comparisons: dict[str, dict[str, Any]] = {}
    exp_root = current_run_dir.parent.parent
    candidates = [
        *((loss_mode, exp_root / "oracle_mcd" / loss_mode / "eval") for loss_mode in STANDARD_ORACLE_LOSS_MODES),
        ("corn", exp_root / "oracle_mcd_corn" / "corn" / "eval"),
    ]

    for loss_mode, eval_dir in candidates:
        metrics_path = eval_dir / "metrics.json"
        if not metrics_path.exists():
            continue

        payload: dict[str, Any] = {}
        with metrics_path.open("r", encoding="utf-8") as f:
            payload["metrics"] = json.load(f)

        for key, filename in [
            ("adjacency", "adjacency_confusion.json"),
            ("severity", "severity_error.json"),
            ("ordinal", "ordinal_positions.json"),
            ("qwk", "qwk.json"),
            ("emd", "emd.json"),
            ("ordinal_error", "ordinal_error_profile.json"),
            ("tau", "tau_stats.json"),
        ]:
            file_path = eval_dir / filename
            if file_path.exists():
                with file_path.open("r", encoding="utf-8") as f:
                    payload[key] = json.load(f)
        comparisons[loss_mode] = payload
    return comparisons


def summarize_comparison(before_name: str, after_name: str, comparisons: dict[str, dict[str, Any]]) -> str:
    before = comparisons.get(before_name)
    after = comparisons.get(after_name)
    if before is None or after is None:
        return f"- {after_name} vs {before_name}: comparison unavailable until both evaluations exist."

    before_adj = before["adjacency"]["summary"]["minor_major_bidirectional"]
    after_adj = after["adjacency"]["summary"]["minor_major_bidirectional"]
    before_f1 = before["metrics"]["macro_f1"]
    after_f1 = after["metrics"]["macro_f1"]
    before_qwk = before.get("qwk", {}).get("quadratic_weighted_kappa", before["metrics"].get("quadratic_weighted_kappa", 0.0))
    after_qwk = after.get("qwk", {}).get("quadratic_weighted_kappa", after["metrics"].get("quadratic_weighted_kappa", 0.0))

    if after_adj < before_adj:
        verdict = "improved"
    elif after_adj > before_adj:
        verdict = "worsened"
    else:
        verdict = "held flat"

    return (
        f"- {after_name} vs {before_name}: minor/major bidirectional confusion {verdict} "
        f"({before_adj} -> {after_adj}), macro_f1 {before_f1:.4f} -> {after_f1:.4f}, "
        f"QWK {before_qwk:.4f} -> {after_qwk:.4f}."
    )


def summarize_learned_gaps(comparisons: dict[str, dict[str, Any]]) -> list[str]:
    for key in ["adaptive_ucl_cda_v3", "adaptive_ucl_cda_v2", "adaptive_ucl_cda", "learnable_cda"]:
        payload = comparisons.get(key)
        if payload is None or "ordinal" not in payload:
            continue
        ordinal = payload["ordinal"]
        gaps = ordinal["gaps"]
        adjacent_gaps = {
            "no-damage <-> minor-damage": float(gaps["gap_01"]),
            "minor-damage <-> major-damage": float(gaps["gap_12"]),
            "major-damage <-> destroyed": float(gaps["gap_23"]),
        }
        smallest_pair = min(adjacent_gaps, key=adjacent_gaps.get)
        largest_pair = max(adjacent_gaps, key=adjacent_gaps.get)
        tau_stats = ordinal.get("tau_statistics")
        lines = [
            (
                f"- {key} gaps: gap_01={float(gaps['gap_01']):.4f}, "
                f"gap_12={float(gaps['gap_12']):.4f}, gap_23={float(gaps['gap_23']):.4f}."
            ),
            f"- Smallest adjacent gap: {smallest_pair}.",
            f"- Largest adjacent gap: {largest_pair}.",
        ]
        if tau_stats is not None:
            lines.append(
                f"- Tau stats: mean={tau_stats['mean']:.4f}, std={tau_stats['std']:.4f}, "
                f"p10={tau_stats['p10']:.4f}, p50={tau_stats['p50']:.4f}, p90={tau_stats['p90']:.4f}."
            )
        return lines
    return ["- Learnable ordinal gaps: not available yet."]


def write_analysis_report(
    eval_dir: Path,
    metrics: dict[str, Any],
    adjacency_report: dict[str, Any],
    severity_report: dict[str, Any],
    ordinal_state: dict[str, Any],
    comparisons: dict[str, dict[str, Any]],
    emd_report: dict[str, Any],
) -> None:
    top_confusion, confusion_count = find_top_confusions(metrics["confusion_matrix"])
    lines = [
        "# Oracle Upper-Bound Analysis",
        "",
        "## Core Reading",
        f"- Strongest confusion pair: {top_confusion} ({confusion_count} samples).",
        f"- minor-damage vs major-damage cross-confusions: {adjacency_report['summary']['minor_major_bidirectional']}.",
        f"- Mean absolute severity error: {severity_report['mean_absolute_severity_error']:.4f}.",
        f"- Quadratic weighted kappa: {metrics['quadratic_weighted_kappa']:.4f}.",
        f"- EMD / Wasserstein-1 severity error: {emd_report['emd_severity']:.4f}.",
        "",
        "## Learned Severity Axis",
    ]
    for class_name, position in ordinal_state["positions_by_class"].items():
        lines.append(f"- {class_name}: {position:.4f}")
    if ordinal_state["tau"] is not None:
        lines.append(f"- tau reference: {ordinal_state['tau']:.4f}")
    if ordinal_state.get("tau_statistics") is not None:
        tau_stats = ordinal_state["tau_statistics"]
        lines.append(
            f"- tau stats: mean={tau_stats['mean']:.4f}, std={tau_stats['std']:.4f}, "
            f"p10={tau_stats['p10']:.4f}, p50={tau_stats['p50']:.4f}, p90={tau_stats['p90']:.4f}"
        )
    if ordinal_state.get("difficulty_statistics") is not None:
        difficulty_stats = ordinal_state["difficulty_statistics"]
        corr_tau_difficulty = ordinal_state.get("corr_tau_difficulty")
        corr_text = "n/a" if corr_tau_difficulty is None else f"{corr_tau_difficulty:.4f}"
        lines.append(
            f"- difficulty stats: mean={difficulty_stats['mean']:.4f}, std={difficulty_stats['std']:.4f}, "
            f"corr_tau_difficulty={corr_text}"
        )

    lines.extend(["", "## Loss Comparison"])
    if comparisons:
        for loss_mode in STANDARD_ORACLE_LOSS_MODES + ["corn"]:
            payload = comparisons.get(loss_mode)
            if payload is None:
                lines.append(f"- {loss_mode}: metrics not available yet.")
                continue
            metrics_payload = payload["metrics"]
            adjacency_payload = payload.get("adjacency", {"summary": {"minor_major_bidirectional": 0}})
            severity_payload = payload.get("severity", {"mean_absolute_severity_error": 0.0})
            qwk_payload = payload.get("qwk", {"quadratic_weighted_kappa": metrics_payload.get("quadratic_weighted_kappa", 0.0)})
            emd_payload = payload.get("emd", {"emd_severity": 0.0})
            lines.append(
                f"- {loss_mode}: macro_f1={metrics_payload['macro_f1']:.4f}, "
                f"balanced_accuracy={metrics_payload['balanced_accuracy']:.4f}, "
                f"qwk={qwk_payload['quadratic_weighted_kappa']:.4f}, "
                f"emd={emd_payload['emd_severity']:.4f}, "
                f"minor_major_bidirectional={adjacency_payload['summary']['minor_major_bidirectional']}, "
                f"severity_mae={severity_payload['mean_absolute_severity_error']:.4f}"
            )
        lines.append("")
        lines.append(summarize_comparison("weighted_ce", "fixed_cda", comparisons))
        lines.append(summarize_comparison("fixed_cda", "learnable_cda", comparisons))
        lines.append(summarize_comparison("learnable_cda", "adaptive_ucl_cda", comparisons))
        lines.append(summarize_comparison("adaptive_ucl_cda", "adaptive_ucl_cda_v2", comparisons))
        lines.append(summarize_comparison("adaptive_ucl_cda_v2", "adaptive_ucl_cda_v3", comparisons))
        lines.extend(summarize_learned_gaps(comparisons))
    else:
        lines.append("- Only the current loss mode has been evaluated so far.")

    write_text(eval_dir / "oracle_upper_bound_analysis.md", "\n".join(lines) + "\n")


def print_ordinal_state(ordinal_state: dict[str, Any]) -> None:
    print("Ordinal severity axis:")
    for class_name, position in ordinal_state["positions_by_class"].items():
        print(f"  {class_name} -> {position:.4f}")
    if ordinal_state["tau"] is not None:
        print(f"Reference tau: {ordinal_state['tau']:.4f}")
    if ordinal_state.get("tau_statistics") is not None:
        tau_stats = ordinal_state["tau_statistics"]
        print(
            "Tau stats: "
            f"mean={tau_stats['mean']:.4f}, std={tau_stats['std']:.4f}, "
            f"p10={tau_stats['p10']:.4f}, p50={tau_stats['p50']:.4f}, p90={tau_stats['p90']:.4f}"
        )


def main() -> None:
    args = parse_args()
    direct_run = len(sys.argv) == 1
    config = apply_overrides(load_config(args.config), args, default_loss_mode="adaptive_ucl_cda_v3")

    checkpoint_path = resolve_checkpoint_path(config, args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if "config" in checkpoint:
        checkpoint_default_loss_mode = "adaptive_ucl_cda_v3"
        training_cfg = checkpoint["config"].get("training", {})
        if "loss_mode" not in training_cfg and "loss_state_dict" not in checkpoint:
            checkpoint_default_loss_mode = "weighted_ce"
        config = apply_overrides(copy.deepcopy(checkpoint["config"]), args, default_loss_mode=checkpoint_default_loss_mode)

    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(config["training"]["amp"]) and device.type == "cuda"

    loader, dataset = make_dataloader(config)
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    criterion = build_loss_module(config, checkpoint, device=device)

    run_dir = checkpoint_path.parent.parent
    eval_dir = Path(args.save_dir) if args.save_dir is not None else run_dir / "eval"
    ensure_dir(eval_dir)
    write_yaml(eval_dir / "eval_config.yaml", config)

    print(f"DIRECT_RUN={direct_run}")
    print(f"Device: {device}")
    print(f"Dataset instances: {len(dataset)}")
    print(f"Cache: {dataset.cache_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model type: {config['model']['model_type']}")
    print(f"Head type: {config['model'].get('head_type', 'standard')}")
    print(f"Loss mode: {config['training']['loss_mode']}")

    y_true, y_pred, prediction_records, inference_summary = run_inference(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        amp_enabled=amp_enabled,
    )
    metrics, report = compute_classification_metrics(y_true, y_pred, CLASS_NAMES)
    metrics["model_type"] = config["model"]["model_type"]
    metrics["head_type"] = config["model"].get("head_type", "standard")
    metrics["loss_mode"] = config["training"]["loss_mode"]
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["num_instances"] = len(dataset)

    ordinal_state = save_ordinal_exports(eval_dir, criterion)
    adjacency_report = write_adjacency_confusion_report(eval_dir, metrics)
    severity_report = build_severity_error_report(prediction_records, ordinal_state)
    write_json(eval_dir / "severity_error.json", severity_report)
    qwk_report = write_qwk_report(eval_dir, metrics)
    emd_report = write_emd_report(eval_dir, prediction_records, ordinal_state["positions"])
    metrics["emd_severity"] = float(emd_report["emd_severity"])
    ordinal_error_profile = write_ordinal_error_profile(eval_dir, y_true, y_pred)
    metrics["adjacent_error_rate"] = float(ordinal_error_profile["adjacent_error_rate"])
    metrics["far_error_rate"] = float(ordinal_error_profile["far_error_rate"])
    tau_report = write_tau_statistics(eval_dir, inference_summary, ordinal_state)
    tau_diag = export_tau_diagnostics(eval_dir, prediction_records)
    write_adaptive_tau_analysis(eval_dir, str(config["training"]["loss_mode"]), tau_report, tau_diag)

    write_json(eval_dir / "metrics.json", metrics)
    write_text(eval_dir / "classification_report.txt", report)
    save_confusion_matrix_plot(metrics["confusion_matrix"], CLASS_NAMES, eval_dir / "confusion_matrix.png")
    save_prediction_records(eval_dir / "predictions.jsonl", prediction_records)
    save_visual_examples(
        dataset=dataset,
        records=prediction_records,
        eval_dir=eval_dir,
        num_visuals_per_class=int(config["evaluation"]["num_visuals_per_class"]),
    )
    save_hardest_minor_major_visuals(
        dataset=dataset,
        records=prediction_records,
        eval_dir=eval_dir,
        top_k=int(config["evaluation"]["num_visuals_per_class"]),
    )

    comparisons = load_loss_mode_reports(run_dir)
    comparisons[config["training"]["loss_mode"]] = {
        "metrics": metrics,
        "adjacency": adjacency_report,
        "severity": severity_report,
        "ordinal": json.loads(json.dumps(ordinal_state)),
        "qwk": qwk_report,
        "emd": emd_report,
        "ordinal_error": ordinal_error_profile,
        "tau": tau_report,
    }
    write_analysis_report(eval_dir, metrics, adjacency_report, severity_report, ordinal_state, comparisons, emd_report)

    print("Evaluation complete.")
    print(
        f"overall_accuracy={metrics['overall_accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} "
        f"weighted_f1={metrics['weighted_f1']:.4f} "
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f} "
        f"qwk={metrics['quadratic_weighted_kappa']:.4f} "
        f"emd={metrics['emd_severity']:.4f}"
    )
    print(
        f"adjacent_error_rate={metrics['adjacent_error_rate']:.4f} "
        f"far_error_rate={metrics['far_error_rate']:.4f}"
    )
    if tau_report.get("inference_tau_statistics") is not None:
        tau_stats = tau_report["inference_tau_statistics"]
        print(
            f"tau_mean={tau_stats['mean']:.4f} tau_std={tau_stats['std']:.6f} "
            f"tau_p10={tau_stats['p10']:.4f} tau_p50={tau_stats['p50']:.4f} tau_p90={tau_stats['p90']:.4f}"
        )
    if tau_report.get("inference_difficulty_statistics") is not None:
        difficulty_stats = tau_report["inference_difficulty_statistics"]
        corr_tau_difficulty = tau_report.get("corr_tau_difficulty")
        corr_text = "n/a" if corr_tau_difficulty is None else f"{corr_tau_difficulty:.4f}"
        print(
            f"difficulty_mean={difficulty_stats['mean']:.4f} difficulty_std={difficulty_stats['std']:.4f} "
            f"corr_tau_difficulty={corr_text}"
        )
    print(f"Mean absolute severity error: {severity_report['mean_absolute_severity_error']:.4f}")
    print_ordinal_state(ordinal_state)
    print(f"Artifacts saved to: {eval_dir}")


if __name__ == "__main__":
    main()
