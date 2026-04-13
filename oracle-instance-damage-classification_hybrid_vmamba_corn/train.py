from __future__ import annotations

import argparse
import copy
import math
import sys
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any

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
from utils.io import (
    append_jsonl,
    ensure_dir,
    load_checkpoint,
    read_yaml,
    save_checkpoint,
    write_json,
    write_text,
    write_yaml,
)
from utils.losses import CORNLoss, DamageLossModule, build_loss_function, compute_class_weights
from utils.metrics import compute_classification_metrics
from utils.seed import seed_worker, set_seed

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


class TeeStream:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8") if streams else "utf-8"

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the single hybrid_vmamba CORN mainline.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--train_list", type=str, default=None)
    parser.add_argument("--val_list", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--vmamba_pretrained_weight_path", type=str, default=None)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
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
    train_cfg.setdefault("epochs", 50)
    train_cfg.setdefault("lr", 3e-4)
    train_cfg.setdefault("weight_decay", 1e-4)
    train_cfg.setdefault("optimizer", "adamw")
    train_cfg.setdefault("warmup_epochs", 3)
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
    train_cfg.setdefault("tau_freeze_epochs", 1)
    train_cfg.setdefault("tau_warmup_value", 0.22)
    train_cfg.setdefault("corn_soft_start_epoch", 3)
    train_cfg.setdefault("ambiguity_lr_scale", 0.30)
    train_cfg.setdefault("concentration_margin", 0.05)
    train_cfg.setdefault("early_stop_patience", 6)

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
    if args.train_list is not None:
        data_cfg["train_list"] = args.train_list
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
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.num_workers is not None:
        train_cfg["num_workers"] = args.num_workers
    if args.lr is not None:
        train_cfg["lr"] = args.lr
    if args.weight_decay is not None:
        train_cfg["weight_decay"] = args.weight_decay
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


def make_dataloaders(
    config: dict[str, Any],
) -> tuple[DataLoader, DataLoader, XBDOracleInstanceDamageDataset, XBDOracleInstanceDamageDataset]:
    train_dataset = XBDOracleInstanceDamageDataset(
        config=config,
        split_name="train",
        list_path=config["data"]["train_list"],
        is_train=True,
    )
    val_dataset = XBDOracleInstanceDamageDataset(
        config=config,
        split_name=config["evaluation"]["split_name"],
        list_path=config["data"]["val_list"],
        is_train=False,
    )
    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]))
    common_loader_kwargs = {
        "num_workers": int(config["training"]["num_workers"]),
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": oracle_instance_collate_fn,
        "worker_init_fn": seed_worker,
        "persistent_workers": int(config["training"]["num_workers"]) > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        generator=generator,
        **common_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        **common_loader_kwargs,
    )
    return train_loader, val_loader, train_dataset, val_dataset


def build_loss_module_from_config(
    config: dict[str, Any],
    class_weights: torch.Tensor,
    *,
    device: torch.device,
) -> DamageLossModule:
    return build_loss_function(
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
    ).to(device)


def build_optimizer(
    model: torch.nn.Module,
    criterion: DamageLossModule,
    config: dict[str, Any],
) -> torch.optim.Optimizer:
    if str(config["training"]["optimizer"]).lower() != "adamw":
        raise ValueError("Only optimizer=adamw is supported.")

    weight_decay = float(config["training"]["weight_decay"])
    base_lr = float(config["training"]["lr"])
    ambiguity_lr_scale = float(config["training"]["ambiguity_lr_scale"])

    param_groups: list[dict[str, Any]] = []

    trunk_params = [param for param in model.get_trunk_parameters() if param.requires_grad]
    if trunk_params:
        param_groups.append(
            {
                "params": trunk_params,
                "weight_decay": weight_decay,
                "lr_scale": 1.0,
                "group_name": "trunk",
            }
        )

    classifier_params = [param for param in model.get_primary_classifier_head_parameters() if param.requires_grad]
    if classifier_params:
        param_groups.append(
            {
                "params": classifier_params,
                "weight_decay": weight_decay,
                "lr_scale": 1.0,
                "group_name": "corn_head",
            }
        )

    ordinal_params = [param for param in criterion.get_gap_parameters() if param.requires_grad]
    if ordinal_params:
        param_groups.append(
            {
                "params": ordinal_params,
                "weight_decay": 0.0,
                "lr_scale": 0.5,
                "group_name": "ordinal",
            }
        )

    ambiguity_params = [param for param in model.get_ambiguity_head_parameters() if param.requires_grad]
    if ambiguity_params:
        param_groups.append(
            {
                "params": ambiguity_params,
                "weight_decay": 0.0,
                "lr_scale": ambiguity_lr_scale,
                "group_name": "ambiguity",
            }
        )

    return torch.optim.AdamW(param_groups, lr=base_lr)


def set_epoch_lr(
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
    epoch_index: int,
    total_epochs: int,
    warmup_epochs: int,
) -> float:
    if warmup_epochs > 0 and epoch_index < warmup_epochs:
        multiplier = float(epoch_index + 1) / float(warmup_epochs)
    else:
        denom = max(total_epochs - warmup_epochs, 1)
        progress = float(epoch_index - warmup_epochs + 1) / float(denom)
        progress = min(max(progress, 0.0), 1.0)
        multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))

    current_lr = base_lr * multiplier
    for group in optimizer.param_groups:
        group["lr"] = current_lr * float(group.get("lr_scale", 1.0))
    return current_lr


def set_ambiguity_head_trainable(model: torch.nn.Module, trainable: bool) -> None:
    for param in model.get_ambiguity_head_parameters():
        param.requires_grad = trainable


def resolve_tau_phase(config: dict[str, Any], epoch_index: int) -> dict[str, Any]:
    freeze_epochs = int(config["training"]["tau_freeze_epochs"])
    warmup_value = float(config["training"]["tau_warmup_value"])
    if epoch_index < freeze_epochs:
        return {
            "phase_name": "warmup_fixed_tau",
            "tau_override_value": warmup_value,
            "ambiguity_trainable": False,
        }
    return {
        "phase_name": "adaptive_tau",
        "tau_override_value": None,
        "ambiguity_trainable": True,
    }


def summarize_distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    tensor = torch.tensor(values, dtype=torch.float32)
    quantiles = torch.quantile(tensor, torch.tensor([0.10, 0.50, 0.90], dtype=torch.float32))
    std = tensor.std(unbiased=False) if tensor.numel() > 1 else tensor.new_tensor(0.0)
    return {
        "mean": float(tensor.mean().item()),
        "std": float(std.item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "p10": float(quantiles[0].item()),
        "p50": float(quantiles[1].item()),
        "p90": float(quantiles[2].item()),
    }


def compute_scalar_correlation(first: list[float], second: list[float]) -> float | None:
    if not first or not second or len(first) != len(second):
        return None
    first_tensor = torch.tensor(first, dtype=torch.float32)
    second_tensor = torch.tensor(second, dtype=torch.float32)
    if first_tensor.numel() < 2:
        return 0.0
    first_centered = first_tensor - first_tensor.mean()
    second_centered = second_tensor - second_tensor.mean()
    denominator = torch.sqrt(first_centered.pow(2).sum() * second_centered.pow(2).sum()).clamp_min(1e-12)
    return float(((first_centered * second_centered).sum() / denominator).item())


def run_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: DamageLossModule,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | torch.amp.GradScaler | None,
    device: torch.device,
    amp_enabled: bool,
    tau_override_value: float | None,
    corn_soft_enabled: bool,
    classifier_head_parameters: list[torch.nn.Parameter],
) -> dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    criterion.train(is_train)

    total_loss = 0.0
    total_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    tau_values: list[float] = []
    raw_tau_values: list[float] = []
    difficulty_values: list[float] = []
    loss_terms = {
        "loss_ord": 0.0,
        "loss_corn_main": 0.0,
        "loss_corn_soft": 0.0,
        "loss_corn_soft_emd": 0.0,
        "loss_corn_unimodal": 0.0,
        "loss_gap_reg": 0.0,
        "loss_tau_reg": 0.0,
        "loss_tau_mean": 0.0,
        "loss_tau_var": 0.0,
        "loss_tau_diff": 0.0,
        "loss_tau_rank": 0.0,
        "loss_raw_tau_diff": 0.0,
        "loss_raw_tau_center": 0.0,
        "loss_raw_tau_bound": 0.0,
    }

    progress = tqdm(loader, leave=False, desc="train" if is_train else "val")
    for batch in progress:
        pre_image = batch["pre_image"].to(device, non_blocking=True)
        post_image = batch["post_image"].to(device, non_blocking=True)
        instance_mask = batch["instance_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        if is_train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

        context = torch.autocast(device_type=device.type, dtype=torch.float16) if amp_enabled else nullcontext()
        with context:
            outputs = model(pre_image, post_image, instance_mask)
            logits = outputs["logits"]
            tau = outputs["tau"]
            if tau_override_value is not None:
                tau = torch.full_like(labels, float(tau_override_value), dtype=logits.dtype)
            loss_outputs = criterion(
                logits,
                labels,
                sample_tau=tau,
                raw_tau=outputs["raw_tau"],
                corn_soft_enabled=corn_soft_enabled,
            )
            loss = loss_outputs["loss"]
            loss_backward = loss_outputs.get("loss_backward", loss)
            probabilities = loss_outputs.get("class_probabilities")
            if probabilities is None:
                probabilities = CORNLoss.logits_to_class_probabilities(logits)

        if not torch.isfinite(loss.detach().float()).all():
            raise FloatingPointError("Non-finite loss detected during training.")

        if is_train:
            assert optimizer is not None
            assert scaler is not None
            head_only_soft_grads: tuple[torch.Tensor | None, ...] | None = None
            if corn_soft_enabled and classifier_head_parameters:
                loss_safe_head_only = loss_outputs.get("loss_safe_head_only")
                if loss_safe_head_only is not None and float(loss_safe_head_only.detach().item()) > 0.0:
                    head_only_soft_grads = torch.autograd.grad(
                        scaler.scale(loss_safe_head_only),
                        classifier_head_parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
            scaler.scale(loss_backward).backward()
            if head_only_soft_grads is not None:
                for param, grad in zip(classifier_head_parameters, head_only_soft_grads):
                    if grad is None:
                        continue
                    if param.grad is None:
                        param.grad = grad.detach()
                    else:
                        param.grad.add_(grad.detach())
            scaler.step(optimizer)
            scaler.update()

        batch_size = labels.size(0)
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size

        predictions = probabilities.detach().argmax(dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())

        tau_values.extend(float(value) for value in tau.detach().float().cpu().tolist())
        raw_tau_values.extend(float(value) for value in outputs["raw_tau"].detach().float().cpu().tolist())

        batch_difficulty = loss_outputs.get("difficulty")
        if batch_difficulty is not None:
            difficulty_values.extend(float(value) for value in batch_difficulty.detach().float().cpu().tolist())

        for key in loss_terms:
            value = loss_outputs.get(key)
            if value is not None:
                loss_terms[key] += float(value.detach().item()) * batch_size

        progress.set_postfix(
            loss=f"{loss.detach().item():.4f}",
            corn=f"{loss_outputs['loss_corn_main'].detach().item():.4f}",
            tau=f"{tau.detach().float().mean().item():.3f}",
        )

    metrics, _ = compute_classification_metrics(y_true, y_pred, CLASS_NAMES)
    metrics["loss"] = total_loss / max(total_samples, 1)
    metrics["loss_terms"] = {key: value / max(total_samples, 1) for key, value in loss_terms.items()}
    metrics["tau_stats"] = summarize_distribution(tau_values)
    metrics["raw_tau_stats"] = summarize_distribution(raw_tau_values)
    metrics["difficulty_stats"] = summarize_distribution(difficulty_values)
    metrics["corr_tau_difficulty"] = compute_scalar_correlation(tau_values, difficulty_values)
    return metrics


def save_class_statistics(run_dir: Path, class_counts: list[int], class_weights: torch.Tensor) -> None:
    payload = {
        "class_names": CLASS_NAMES,
        "class_counts": {name: int(count) for name, count in zip(CLASS_NAMES, class_counts)},
        "class_weights": {name: float(weight) for name, weight in zip(CLASS_NAMES, class_weights.tolist())},
    }
    write_json(run_dir / "class_stats_train.json", payload)


def make_checkpoint_state(
    *,
    epoch: int,
    model: torch.nn.Module,
    criterion: DamageLossModule,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | torch.amp.GradScaler,
    config: dict[str, Any],
    class_weights: torch.Tensor,
    class_counts: list[int],
    best_macro_f1: float,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    tau_phase: dict[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "loss_state_dict": criterion.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": config,
        "class_names": CLASS_NAMES,
        "class_weights": class_weights.detach().cpu(),
        "class_counts": [int(count) for count in class_counts],
        "best_macro_f1": float(best_macro_f1),
        "ordinal_state": criterion.export_state(CLASS_NAMES),
        "tau_phase": tau_phase,
        "tau_statistics": {
            "train": train_metrics.get("tau_stats"),
            "val": val_metrics.get("tau_stats"),
        },
    }


def print_startup_summary(
    *,
    config: dict[str, Any],
    run_dir: Path,
    device: torch.device,
    model: torch.nn.Module,
    train_dataset: XBDOracleInstanceDamageDataset,
    val_dataset: XBDOracleInstanceDamageDataset,
    train_loader: DataLoader,
) -> None:
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Config path: {config.get('_config_path')}")
    print(f"Run dir: {run_dir}")
    print(f"Device: {device}")
    print(f"backbone={MAINLINE_BACKBONE}")
    print(f"model_type={MAINLINE_MODEL_TYPE}")
    print(f"loss_mode={MAINLINE_LOSS_MODE}")
    print(f"vmamba_pretrained_weight_path={config['model'].get('vmamba_pretrained_weight_path', '')}")
    print(f"encoder feature channels={dict(getattr(model.encoder, 'feature_channels', {}))}")
    print(f"train dataset size={len(train_dataset)}")
    print(f"val dataset size={len(val_dataset)}")
    print(f"train loader steps={len(train_loader)}")


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    run_dir = make_run_dir(config)
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    logs_dir = ensure_dir(run_dir / "logs")
    history_path = logs_dir / "history.jsonl"
    train_log_path = logs_dir / "train.log"

    if args.resume is None:
        write_text(history_path, "")

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with train_log_path.open("a", encoding="utf-8") as log_file:
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        try:
            run_training(
                config=config,
                run_dir=run_dir,
                checkpoints_dir=checkpoints_dir,
                history_path=history_path,
                resume_path=args.resume,
            )
        except Exception:
            traceback.print_exc()
            raise
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def run_training(
    *,
    config: dict[str, Any],
    run_dir: Path,
    checkpoints_dir: Path,
    history_path: Path,
    resume_path: str | None,
) -> None:
    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(config["training"]["amp"]) and device.type == "cuda"

    write_yaml(run_dir / "run_config.yaml", config)

    train_loader, val_loader, train_dataset, val_dataset = make_dataloaders(config)
    class_weights = compute_class_weights(train_dataset.class_counts)
    save_class_statistics(run_dir, train_dataset.class_counts, class_weights)

    model = build_model(config).to(device)
    criterion = build_loss_module_from_config(config, class_weights, device=device)
    optimizer = build_optimizer(model, criterion, config)
    try:
        scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_epoch = 0
    best_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        criterion.load_state_dict(checkpoint["loss_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        start_epoch = int(checkpoint.get("epoch", 0))
        best_macro_f1 = float(checkpoint.get("best_macro_f1", -1.0))
        best_epoch = int(checkpoint.get("epoch", 0))

    classifier_head_parameters = [
        param for param in model.get_primary_classifier_head_parameters() if param.requires_grad
    ]

    print_startup_summary(
        config=config,
        run_dir=run_dir,
        device=device,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_loader=train_loader,
    )

    total_epochs = int(config["training"]["epochs"])
    early_stop_patience = int(config["training"]["early_stop_patience"])

    for epoch_index in range(start_epoch, total_epochs):
        tau_phase = resolve_tau_phase(config, epoch_index)
        set_ambiguity_head_trainable(model, bool(tau_phase["ambiguity_trainable"]))
        corn_soft_enabled = bool((epoch_index + 1) >= int(config["training"]["corn_soft_start_epoch"]))

        current_lr = set_epoch_lr(
            optimizer,
            base_lr=float(config["training"]["lr"]),
            epoch_index=epoch_index,
            total_epochs=total_epochs,
            warmup_epochs=int(config["training"]["warmup_epochs"]),
        )
        print(
            f"\nEpoch [{epoch_index + 1}/{total_epochs}] "
            f"lr={current_lr:.8f} "
            f"tau_phase={tau_phase['phase_name']} "
            f"corn_soft_enabled={corn_soft_enabled}"
        )

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            tau_override_value=tau_phase["tau_override_value"],
            corn_soft_enabled=corn_soft_enabled,
            classifier_head_parameters=classifier_head_parameters,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            scaler=None,
            device=device,
            amp_enabled=amp_enabled,
            tau_override_value=tau_phase["tau_override_value"],
            corn_soft_enabled=corn_soft_enabled,
            classifier_head_parameters=classifier_head_parameters,
        )

        print(
            "Train "
            f"loss={train_metrics['loss']:.4f} "
            f"macro_f1={train_metrics['macro_f1']:.4f} "
            f"balanced_accuracy={train_metrics['balanced_accuracy']:.4f}"
        )
        print(
            "Val   "
            f"loss={val_metrics['loss']:.4f} "
            f"macro_f1={val_metrics['macro_f1']:.4f} "
            f"balanced_accuracy={val_metrics['balanced_accuracy']:.4f}"
        )
        if train_metrics.get("tau_stats") is not None:
            print(f"Train tau_stats={train_metrics['tau_stats']}")
        if val_metrics.get("tau_stats") is not None:
            print(f"Val tau_stats={val_metrics['tau_stats']}")

        improved = float(val_metrics["macro_f1"]) > best_macro_f1
        if improved:
            best_macro_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch_index + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        checkpoint_state = make_checkpoint_state(
            epoch=epoch_index + 1,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            class_weights=class_weights,
            class_counts=train_dataset.class_counts,
            best_macro_f1=best_macro_f1,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            tau_phase=tau_phase,
        )
        save_checkpoint(checkpoints_dir / "last.pth", checkpoint_state)
        if improved:
            save_checkpoint(checkpoints_dir / "best_macro_f1.pth", checkpoint_state)

        append_jsonl(
            history_path,
            {
                "epoch": epoch_index + 1,
                "lr": current_lr,
                "tau_phase": tau_phase,
                "corn_soft_enabled": corn_soft_enabled,
                "train": train_metrics,
                "val": val_metrics,
                "ordinal": criterion.export_state(CLASS_NAMES),
            },
        )

        if epochs_without_improvement >= early_stop_patience:
            print(
                f"Early stopping after {epochs_without_improvement} epochs without macro_f1 improvement."
            )
            break

    summary = {
        "run_dir": str(run_dir),
        "best_epoch": int(best_epoch),
        "best_macro_f1": float(best_macro_f1),
        "best_checkpoint": str(checkpoints_dir / "best_macro_f1.pth"),
        "last_checkpoint": str(checkpoints_dir / "last.pth"),
        "train_instances": len(train_dataset),
        "val_instances": len(val_dataset),
        "model_type": MAINLINE_MODEL_TYPE,
        "backbone": MAINLINE_BACKBONE,
        "loss_mode": MAINLINE_LOSS_MODE,
    }
    write_json(run_dir / "train_summary.json", summary)
    print("\nTraining complete.")
    print(f"Best macro F1: {best_macro_f1:.4f} at epoch {best_epoch}")
    print(f"Checkpoints saved to: {checkpoints_dir}")


if __name__ == "__main__":
    main()
