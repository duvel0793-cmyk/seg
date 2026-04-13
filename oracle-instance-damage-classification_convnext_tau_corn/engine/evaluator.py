"""Evaluation loop for validation and test splits."""

from __future__ import annotations

from typing import Any, Dict

import torch

from datasets.label_mapping import CLASS_NAMES
from losses.corn_loss import corn_loss
from metrics.cls_metrics import compute_classification_metrics
from utils.misc import AverageMeter, TensorStatsMeter, move_batch_to_device


def _format_per_class_f1(per_class):
    return {item["class_name"]: round(float(item["f1"]), 4) for item in per_class}


def _tensor_to_list(value):
    if isinstance(value, torch.Tensor):
        return [float(item) for item in value.detach().cpu().view(-1).tolist()]
    return [float(item) for item in value]


class Evaluator:
    """Structured evaluator for oracle damage classification."""

    def __init__(self, model: torch.nn.Module, config: Dict[str, Any], device: torch.device, logger) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.logger = logger

    @torch.no_grad()
    def evaluate(self, dataloader, split: str = "val") -> Dict[str, Any]:
        self.model.eval()
        total_loss_meter = AverageMeter()
        corn_loss_meter = AverageMeter()
        center_reg_meter = AverageMeter()
        diff_reg_meter = AverageMeter()
        base_logits_meter = TensorStatsMeter()
        final_logits_meter = TensorStatsMeter()

        all_targets = []
        all_preds = []

        model_cfg = self.config["model"]
        use_amp = bool(self.config["runtime"].get("amp", True)) and self.device.type == "cuda"
        autocast_device = "cuda" if self.device.type == "cuda" else "cpu"
        last_tau = []
        last_tau_info = {}

        for batch in dataloader:
            batch = move_batch_to_device(batch, self.device)
            with torch.autocast(device_type=autocast_device, enabled=use_amp):
                outputs = self.model(
                    pre_image=batch["pre_image"],
                    post_image=batch["post_image"],
                    mask=batch["mask"],
                )
                loss_corn = corn_loss(outputs["logits"], batch["label"], num_classes=model_cfg["num_classes"])
                center_reg = outputs["tau_center_reg"]
                diff_reg = outputs["tau_diff_reg"]
                total_loss = (
                    loss_corn
                    + model_cfg["tau_reg_weight"] * center_reg
                    + model_cfg["tau_diff_weight"] * diff_reg
                )

            batch_size = batch["label"].shape[0]
            total_loss_meter.update(total_loss.item(), batch_size)
            corn_loss_meter.update(loss_corn.item(), batch_size)
            center_reg_meter.update(center_reg.item(), batch_size)
            diff_reg_meter.update(diff_reg.item(), batch_size)
            base_logits_meter.update(outputs["base_logits"])
            final_logits_meter.update(outputs["logits"])

            all_targets.extend(batch["label"].detach().cpu().tolist())
            all_preds.extend(outputs["pred_labels"].detach().cpu().tolist())
            last_tau = outputs["tau"].detach().cpu().tolist()
            last_tau_info = outputs["tau_info"]

        metrics = compute_classification_metrics(
            targets=all_targets,
            preds=all_preds,
            num_classes=model_cfg["num_classes"],
            class_names=CLASS_NAMES,
        )
        metrics.update(
            {
                "split": split,
                "loss": total_loss_meter.avg,
                "corn_loss": corn_loss_meter.avg,
                "tau_center_reg": center_reg_meter.avg,
                "tau_diff_reg": diff_reg_meter.avg,
                "tau": last_tau,
                "tau_raw": _tensor_to_list(last_tau_info.get("tau_raw", [])),
                "tau_positive_before_clamp": _tensor_to_list(last_tau_info.get("tau_positive_before_clamp", [])),
                "tau_clamped": _tensor_to_list(last_tau_info.get("tau_clamped", [])),
                "base_logits_mean": base_logits_meter.mean,
                "base_logits_std": base_logits_meter.std,
                "final_logits_mean": final_logits_meter.mean,
                "final_logits_std": final_logits_meter.std,
                "decode_mode": model_cfg.get("decode_mode", "threshold_count"),
                "per_class_f1": _format_per_class_f1(metrics["per_class"]),
            }
        )
        self.logger.info(
            "%s loss=%.4f corn_loss=%.4f tau_center_reg=%.6f tau_diff_reg=%.6f "
            "acc=%.4f macro_f1=%.4f weighted_f1=%.4f per_class_f1=%s "
            "tau_clamped=%s tau_positive_before_clamp=%s base_logits_mean=%.4f base_logits_std=%.4f "
            "final_logits_mean=%.4f final_logits_std=%.4f decode_mode=%s",
            split,
            metrics["loss"],
            metrics["corn_loss"],
            metrics["tau_center_reg"],
            metrics["tau_diff_reg"],
            metrics["accuracy"],
            metrics["macro_f1"],
            metrics["weighted_f1"],
            metrics["per_class_f1"],
            metrics["tau_clamped"],
            metrics["tau_positive_before_clamp"],
            metrics["base_logits_mean"],
            metrics["base_logits_std"],
            metrics["final_logits_mean"],
            metrics["final_logits_std"],
            metrics["decode_mode"],
        )
        return metrics
