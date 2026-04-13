from __future__ import annotations

import argparse
import copy
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.xbd_oracle_instance_damage import (
    CLASS_NAMES,
    XBDOracleInstanceDamageDataset,
    oracle_instance_collate_fn,
)
from models import (
    MAINLINE_BACKBONE,
    MAINLINE_HEAD_TYPE,
    MAINLINE_LOSS_MODE,
    MAINLINE_MODEL_TYPE,
    build_model,
)
from utils.io import ensure_dir, load_checkpoint, read_yaml, write_json, write_text, write_yaml
from utils.losses import CORNLoss, DamageLossModule, build_loss_function, compute_expected_severity_from_probabilities
from utils.metrics import (
    compute_classification_metrics,
    compute_emd_from_probabilities,
    compute_ordinal_error_profile,
    save_confusion_matrix_plot,
)
from utils.seed import seed_worker, set_seed

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the single hybrid_vmamba CORN mainline.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--val_list", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--vmamba_pretrained_weight_path", type=str, default=None)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    return copy.deepcopy(read_yaml(path))


def ensure_mainline_config(config: dict[str, Any]) -> dict[str, Any]:
    config.setdefault("project_name", "oracle-instance-damage-classification_hybrid_vmamba_corn")
    config.setdefault("seed", 42)

    data_cfg = config.setdefault("data", {})
    data_cfg.setdefault("root_dir", "/home/lky/data/xBD")
    data_cfg.setdefault("train_list", "/home/lky/data/xBD/xBD_list/train_all.txt")
    data_cfg.setdefault("val_list", "/home/lky/data/xBD/xBD_list/val_all.txt")
    data_cfg.setdefault("instance_source", "gt_json")
    data_cfg.setdefault("allow_tier3", False)
    data_cfg.setdefault("image_size", 224)
    data_cfg.setdefault("context_ratio", 0.25)
    data_cfg.setdefault("min_polygon_area", 16.0)
    data_cfg.setdefault("min_mask_pixels", 16)
    data_cfg.setdefault("max_out_of_bound_ratio", 0.4)
    data_cfg.setdefault("cache_dir", "./cache")

    aug_cfg = config.setdefault("augmentation", {})
    aug_cfg.setdefault("hflip_prob", 0.5)
    aug_cfg.setdefault("vflip_prob", 0.5)
    aug_cfg.setdefault("rotate90_prob", 0.5)
    aug_cfg.setdefault("random_resized_crop_prob", 0.5)
    aug_cfg.setdefault("random_resized_crop_scale", [0.9, 1.0])
    aug_cfg.setdefault("random_resized_crop_ratio", [0.95, 1.05])
    aug_cfg.setdefault("min_mask_retention", 0.75)
    aug_cfg.setdefault("color_jitter_prob", 0.6)
    aug_cfg.setdefault("brightness", 0.15)
    aug_cfg.setdefault("contrast", 0.15)
    aug_cfg.setdefault("saturation", 0.1)
    aug_cfg.setdefault("hue", 0.02)
    aug_cfg.setdefault("context_dropout_prob", 0.0)
    aug_cfg.setdefault("context_blur_prob", 0.0)
    aug_cfg.setdefault("context_grayscale_prob", 0.0)
    aug_cfg.setdefault("context_noise_prob", 0.0)
    aug_cfg.setdefault("context_mix_prob", 0.0)
    aug_cfg.setdefault("context_edge_soften_pixels", 4)
    aug_cfg.setdefault("context_dilate_pixels", 3)
    aug_cfg.setdefault("context_apply_to_pre_and_post_independently", False)
    aug_cfg.setdefault("context_preserve_instance_strictly", True)
    aug_cfg.setdefault("normalize_mean", [0.485, 0.456, 0.406])
    aug_cfg.setdefault("normalize_std", [0.229, 0.224, 0.225])

    model_cfg = config.setdefault("model", {})
    model_cfg["backbone"] = MAINLINE_BACKBONE
    model_cfg["model_type"] = MAINLINE_MODEL_TYPE
    model_cfg["head_type"] = MAINLINE_HEAD_TYPE
    model_cfg.setdefault("pretrained", True)
    model_cfg.setdefault(
        "vmamba_pretrained_weight_path",
        str(PROJECT_ROOT / "checkpoints" / "vmamba_pretrained.pth"),
    )
    model_cfg.setdefault("drop_path_rate", 0.1)
    model_cfg.setdefault("dropout", 0.2)
    model_cfg.setdefault("channel_attention_reduction", 16)
    model_cfg.setdefault("ambiguity_hidden_features", 256)

    train_cfg = config.setdefault("training", {})
    train_cfg["loss_mode"] = MAINLINE_LOSS_MODE
    train_cfg.setdefault("batch_size", 32)
    train_cfg.setdefault("num_workers", 8)
    train_cfg.setdefault("amp", True)
    train_cfg.setdefault("label_smoothing", 0.05)
    train_cfg.setdefault("lambda_gap_reg", 1e-3)
    train_cfg.setdefault("lambda_tau_mean", 0.05)
    train_cfg.setdefault("lambda_tau_diff", 0.20)
    train_cfg.setdefault("lambda_tau_rank", 0.05)
    train_cfg.setdefault("lambda_raw_tau_diff", 0.10)
    train_cfg.setdefault("lambda_raw_tau_center", 0.02)
    train_cfg.setdefault("lambda_raw_tau_bound", 0.02)
    train_cfg.setdefault("lambda_corn_soft", 0.03)
    train_cfg.setdefault("tau_init", 0.22)
    train_cfg.setdefault("tau_min", 0.12)
    train_cfg.setdefault("tau_max", 0.45)
    train_cfg.setdefault("tau_base", 0.22)
    train_cfg.setdefault("delta_scale", 0.10)
    train_cfg.setdefault("tau_parameterization", "bounded_sigmoid")
    train_cfg.setdefault("tau_logit_scale", 2.0)
    train_cfg.setdefault("tau_target", 0.22)
    train_cfg.setdefault("tau_easy", 0.16)
    train_cfg.setdefault("tau_hard", 0.32)
    train_cfg.setdefault("tau_variance_weight", 0.01)
    train_cfg.setdefault("tau_std_floor", 0.03)
    train_cfg.setdefault("tau_rank_margin_difficulty", 0.10)
    train_cfg.setdefault("tau_rank_margin_value", 0.01)
    train_cfg.setdefault("raw_tau_soft_margin", 1.5)
    train_cfg.setdefault("concentration_margin", 0.05)

    eval_cfg = config.setdefault("evaluation", {})
    eval_cfg.setdefault("split_name", "val")

    output_cfg = config.setdefault("output", {})
    output_cfg.setdefault("output_root", "./outputs")
    output_cfg.setdefault("exp_name", "hybrid_vmamba_mainline")
    return config


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = ensure_mainline_config(config)
    config["_config_path"] = str(Path(args.config).resolve())

    if args.seed is not None:
        config["seed"] = args.seed

    data_cfg = config["data"]
    if args.root_dir is not None:
        data_cfg["root_dir"] = args.root_dir
    if args.val_list is not None:
        data_cfg["val_list"] = args.val_list
    if args.cache_dir is not None:
        data_cfg["cache_dir"] = args.cache_dir
    if args.image_size is not None:
        data_cfg["image_size"] = args.image_size

    model_cfg = config["model"]
    if args.vmamba_pretrained_weight_path is not None:
        model_cfg["vmamba_pretrained_weight_path"] = args.vmamba_pretrained_weight_path
    if args.no_pretrained:
        model_cfg["pretrained"] = False

    train_cfg = config["training"]
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
    return config


def make_run_dir(config: dict[str, Any]) -> Path:
    return (
        Path(config["output"]["output_root"])
        / config["output"]["exp_name"]
        / config["model"]["model_type"]
        / config["training"]["loss_mode"]
    )


def resolve_checkpoint_path(
    config: dict[str, Any],
    *,
    explicit_checkpoint: str | None,
    resume_checkpoint: str | None,
) -> Path:
    if explicit_checkpoint is not None:
        return Path(explicit_checkpoint)
    if resume_checkpoint is not None:
        return Path(resume_checkpoint)
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


def build_loss_module(
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    *,
    device: torch.device,
) -> DamageLossModule:
    checkpoint_weights = checkpoint.get("class_weights")
    class_weights = None
    if checkpoint_weights is not None:
        class_weights = (
            checkpoint_weights.float()
            if isinstance(checkpoint_weights, torch.Tensor)
            else torch.tensor(checkpoint_weights, dtype=torch.float32)
        )
    criterion = build_loss_function(
        class_weights=class_weights,
        loss_mode=MAINLINE_LOSS_MODE,
        label_smoothing=float(config["training"]["label_smoothing"]),
        lambda_gap_reg=float(config["training"]["lambda_gap_reg"]),
        lambda_tau_mean=float(config["training"]["lambda_tau_mean"]),
        lambda_tau_diff=float(config["training"]["lambda_tau_diff"]),
        lambda_tau_rank=float(config["training"]["lambda_tau_rank"]),
        lambda_raw_tau_diff=float(config["training"]["lambda_raw_tau_diff"]),
        lambda_raw_tau_center=float(config["training"]["lambda_raw_tau_center"]),
        lambda_raw_tau_bound=float(config["training"]["lambda_raw_tau_bound"]),
        lambda_corn_soft=float(config["training"]["lambda_corn_soft"]),
        tau_init=float(config["training"]["tau_init"]),
        tau_min=float(config["training"]["tau_min"]),
        tau_max=float(config["training"]["tau_max"]),
        tau_base=float(config["training"]["tau_base"]),
        delta_scale=float(config["training"]["delta_scale"]),
        tau_parameterization=str(config["training"]["tau_parameterization"]),
        tau_logit_scale=float(config["training"]["tau_logit_scale"]),
        tau_target=float(config["training"]["tau_target"]),
        tau_easy=float(config["training"]["tau_easy"]),
        tau_hard=float(config["training"]["tau_hard"]),
        tau_variance_weight=float(config["training"]["tau_variance_weight"]),
        tau_std_floor=float(config["training"]["tau_std_floor"]),
        tau_rank_margin_difficulty=float(config["training"]["tau_rank_margin_difficulty"]),
        tau_rank_margin_value=float(config["training"]["tau_rank_margin_value"]),
        raw_tau_soft_margin=float(config["training"]["raw_tau_soft_margin"]),
        concentration_margin=float(config["training"]["concentration_margin"]),
        num_classes=len(CLASS_NAMES),
    )
    if "loss_state_dict" in checkpoint:
        criterion.load_state_dict(checkpoint["loss_state_dict"], strict=False)
    criterion = criterion.to(device)
    criterion.eval()
    return criterion


def summarize_distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "p10": float(np.percentile(array, 10)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "support": int(array.size),
    }


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


def build_prediction_record(
    *,
    sample_index: int,
    label: int,
    prediction: int,
    probabilities: torch.Tensor,
    positions: torch.Tensor,
    tau: float | None,
    raw_tau: float | None,
    meta: dict[str, Any],
) -> dict[str, Any]:
    probability_row = probabilities.float().clamp_min(1e-8)
    confidence = float(probability_row.max().item())
    true_probability = float(probability_row[int(label)].item())
    difficulty = float(1.0 - true_probability)
    expected_severity = float(compute_expected_severity_from_probabilities(probability_row.unsqueeze(0), positions).item())
    gt_severity = float(positions[int(label)].item())
    return {
        "sample_index": int(sample_index),
        "gt": int(label),
        "pred": int(prediction),
        "confidence": confidence,
        "true_probability": true_probability,
        "difficulty": difficulty,
        "tau": tau,
        "raw_tau": raw_tau,
        "expected_severity": expected_severity,
        "gt_severity": gt_severity,
        "severity_abs_error": abs(expected_severity - gt_severity),
        "probabilities": [float(value) for value in probability_row.cpu().tolist()],
        "meta": meta,
    }


def run_inference(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: DamageLossModule,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[list[int], list[int], list[dict[str, Any]], dict[str, Any], torch.Tensor]:
    model.eval()
    criterion.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    tau_values: list[float] = []
    raw_tau_values: list[float] = []
    difficulty_values: list[float] = []
    probability_rows: list[torch.Tensor] = []
    prediction_records: list[dict[str, Any]] = []
    positions = criterion.get_current_positions().to(device=device, dtype=torch.float32)

    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc="eval"):
            pre_image = batch["pre_image"].to(device, non_blocking=True)
            post_image = batch["post_image"].to(device, non_blocking=True)
            instance_mask = batch["instance_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            context = torch.autocast(device_type=device.type, dtype=torch.float16) if amp_enabled else nullcontext()
            with context:
                outputs = model(pre_image, post_image, instance_mask)
                logits = outputs["logits"]
                probabilities = CORNLoss.logits_to_class_probabilities(logits)

            predictions = probabilities.argmax(dim=1)
            true_probabilities = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
            difficulties = 1.0 - true_probabilities

            sample_indices = batch["sample_index"].cpu().tolist()
            metas = batch["meta"]
            tau_cpu = outputs["tau"].detach().float().cpu().tolist()
            raw_tau_cpu = outputs["raw_tau"].detach().float().cpu().tolist()

            for row_index, sample_index in enumerate(sample_indices):
                label = int(labels[row_index].item())
                prediction = int(predictions[row_index].item())
                tau_value = float(tau_cpu[row_index])
                raw_tau_value = float(raw_tau_cpu[row_index])
                tau_values.append(tau_value)
                raw_tau_values.append(raw_tau_value)
                difficulty_values.append(float(difficulties[row_index].item()))
                probability_rows.append(probabilities[row_index].detach().cpu())
                y_true.append(label)
                y_pred.append(prediction)
                prediction_records.append(
                    build_prediction_record(
                        sample_index=int(sample_index),
                        label=label,
                        prediction=prediction,
                        probabilities=probabilities[row_index].detach().cpu(),
                        positions=positions.detach().cpu(),
                        tau=tau_value,
                        raw_tau=raw_tau_value,
                        meta=metas[row_index],
                    )
                )

    inference_summary = {
        "tau_stats": summarize_distribution(tau_values),
        "raw_tau_stats": summarize_distribution(raw_tau_values),
        "difficulty_stats": summarize_distribution(difficulty_values),
        "corr_tau_difficulty": compute_scalar_correlation(tau_values, difficulty_values),
    }
    probability_tensor = (
        torch.stack(probability_rows, dim=0)
        if probability_rows
        else torch.zeros((0, len(CLASS_NAMES)), dtype=torch.float32)
    )
    return y_true, y_pred, prediction_records, inference_summary, probability_tensor


def save_prediction_records(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_ordinal_exports(eval_dir: Path, criterion: DamageLossModule) -> dict[str, Any]:
    ordinal_state = criterion.export_state(CLASS_NAMES)
    write_json(
        eval_dir / "ordinal_positions.json",
        {
            "loss_mode": ordinal_state["loss_mode"],
            "positions": ordinal_state["positions_by_class"],
            "positions_list": ordinal_state["positions"],
            "gaps": ordinal_state["gaps"],
            "tau_statistics": ordinal_state.get("tau_statistics"),
            "difficulty_statistics": ordinal_state.get("difficulty_statistics"),
            "corr_tau_difficulty": ordinal_state.get("corr_tau_difficulty"),
        },
    )
    return ordinal_state


def write_qwk_report(eval_dir: Path, metrics: dict[str, Any]) -> None:
    write_json(
        eval_dir / "qwk.json",
        {
            "quadratic_weighted_kappa": float(metrics["quadratic_weighted_kappa"]),
            "num_instances": int(metrics["num_instances"]),
            "model_type": MAINLINE_MODEL_TYPE,
            "loss_mode": MAINLINE_LOSS_MODE,
        },
    )


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
    payload = {
        "emd_severity": float(mean_emd),
        "positions": {class_name: float(position) for class_name, position in zip(CLASS_NAMES, positions)},
    }
    write_json(eval_dir / "emd.json", payload)
    return payload


def write_ordinal_error_profile(eval_dir: Path, y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    profile = compute_ordinal_error_profile(y_true, y_pred, CLASS_NAMES)
    write_json(eval_dir / "ordinal_error_profile.json", profile)
    write_text(
        eval_dir / "ordinal_error_profile.txt",
        "\n".join(
            [
                "Ordinal error profile",
                f"exact_match_rate={profile['exact_match_rate']:.4f}",
                f"adjacent_error_rate={profile['adjacent_error_rate']:.4f}",
                f"far_error_rate={profile['far_error_rate']:.4f}",
                f"mean_absolute_class_distance={profile['mean_absolute_class_distance']:.4f}",
            ]
        )
        + "\n",
    )
    return profile


def write_tau_statistics(
    eval_dir: Path,
    inference_summary: dict[str, Any],
    ordinal_state: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "inference_tau_statistics": inference_summary.get("tau_stats"),
        "inference_raw_tau_statistics": inference_summary.get("raw_tau_stats"),
        "inference_difficulty_statistics": inference_summary.get("difficulty_stats"),
        "corr_tau_difficulty": inference_summary.get("corr_tau_difficulty"),
        "criterion_tau_statistics": ordinal_state.get("tau_statistics"),
        "criterion_difficulty_statistics": ordinal_state.get("difficulty_statistics"),
        "criterion_corr_tau_difficulty": ordinal_state.get("corr_tau_difficulty"),
    }
    write_json(eval_dir / "tau_stats.json", payload)
    return payload


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    checkpoint_path = resolve_checkpoint_path(
        config,
        explicit_checkpoint=args.checkpoint,
        resume_checkpoint=args.resume,
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if "config" in checkpoint:
        config = apply_overrides(copy.deepcopy(checkpoint["config"]), args)

    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(config["training"]["amp"]) and device.type == "cuda"

    loader, dataset = make_dataloader(config)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    criterion = build_loss_module(config, checkpoint, device=device)

    eval_dir = (
        Path(args.save_dir)
        if args.save_dir is not None
        else checkpoint_path.parent.parent / "eval"
    )
    ensure_dir(eval_dir)
    write_yaml(eval_dir / "eval_config.yaml", config)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Config path: {config.get('_config_path')}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Dataset size: {len(dataset)}")
    print(f"backbone={MAINLINE_BACKBONE}")
    print(f"model_type={MAINLINE_MODEL_TYPE}")
    print(f"loss_mode={MAINLINE_LOSS_MODE}")
    print(f"vmamba_pretrained_weight_path={config['model'].get('vmamba_pretrained_weight_path', '')}")
    print(f"encoder feature channels={dict(getattr(model.encoder, 'feature_channels', {}))}")

    y_true, y_pred, prediction_records, inference_summary, probabilities = run_inference(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        amp_enabled=amp_enabled,
    )

    metrics, report = compute_classification_metrics(y_true, y_pred, CLASS_NAMES)
    metrics["model_type"] = MAINLINE_MODEL_TYPE
    metrics["head_type"] = MAINLINE_HEAD_TYPE
    metrics["loss_mode"] = MAINLINE_LOSS_MODE
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["num_instances"] = len(dataset)

    ordinal_state = save_ordinal_exports(eval_dir, criterion)
    emd_report = write_emd_report(eval_dir, prediction_records, ordinal_state["positions"])
    ordinal_error_profile = write_ordinal_error_profile(eval_dir, y_true, y_pred)
    tau_report = write_tau_statistics(eval_dir, inference_summary, ordinal_state)
    write_qwk_report(eval_dir, metrics)

    metrics["emd_severity"] = float(emd_report["emd_severity"])
    metrics["tau_stats"] = tau_report
    metrics["ordinal_error_profile"] = ordinal_error_profile
    metrics["adjacent_error_rate"] = float(ordinal_error_profile["adjacent_error_rate"])
    metrics["far_error_rate"] = float(ordinal_error_profile["far_error_rate"])

    write_json(eval_dir / "metrics.json", metrics)
    write_text(eval_dir / "classification_report.txt", report)
    save_prediction_records(eval_dir / "predictions.jsonl", prediction_records)
    save_confusion_matrix_plot(metrics["confusion_matrix"], CLASS_NAMES, eval_dir / "confusion_matrix.png")

    print("Evaluation complete.")
    print(
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


if __name__ == "__main__":
    main()
