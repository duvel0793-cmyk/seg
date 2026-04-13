"""Training loop for oracle instance classification."""

from __future__ import annotations

import sys
import time
from typing import Any, Dict, List, Optional

import torch

from datasets.label_mapping import CLASS_NAMES
from engine.hooks import HookBase
from losses.corn_loss import corn_loss
from metrics.cls_metrics import compute_classification_metrics
from utils.misc import AverageMeter, TensorStatsMeter, move_batch_to_device


def _format_per_class_f1(per_class: List[Dict[str, Any]]) -> Dict[str, float]:
    return {item["class_name"]: round(float(item["f1"]), 4) for item in per_class}


def _tensor_to_list(value: torch.Tensor | List[float]) -> List[float]:
    if isinstance(value, torch.Tensor):
        return [float(item) for item in value.detach().cpu().view(-1).tolist()]
    return [float(item) for item in value]


class _ProgressTracker:
    """Minimal terminal progress bar with elapsed time and ETA."""

    def __init__(self, total_steps: int, desc: str, enabled: bool = True, width: int = 28) -> None:
        self.total_steps = max(int(total_steps), 1)
        self.desc = desc
        self.enabled = enabled
        self.width = width
        self.start_time = time.perf_counter()
        self.last_render_len = 0

    def update(self, step: int, extra_text: str = "") -> None:
        if not self.enabled:
            return
        step = min(max(int(step), 0), self.total_steps)
        elapsed = time.perf_counter() - self.start_time
        avg_step_time = elapsed / step if step > 0 else 0.0
        eta = avg_step_time * max(self.total_steps - step, 0)
        ratio = step / self.total_steps
        filled = int(round(self.width * ratio))
        bar = "#" * filled + "-" * (self.width - filled)
        line = (
            f"\r{self.desc} [{bar}] {step}/{self.total_steps} "
            f"{ratio * 100:5.1f}% elapsed {elapsed:6.1f}s eta {eta:6.1f}s step {avg_step_time:5.2f}s"
        )
        if extra_text:
            line += f" {extra_text}"
        padding = max(self.last_render_len - len(line), 0)
        sys.stderr.write(line + (" " * padding))
        sys.stderr.flush()
        self.last_render_len = len(line)

    def close(self, final_text: str = "") -> Dict[str, float]:
        elapsed = time.perf_counter() - self.start_time
        avg_step_time = elapsed / self.total_steps if self.total_steps > 0 else 0.0
        if self.enabled:
            self.update(self.total_steps, extra_text=final_text)
            sys.stderr.write("\n")
            sys.stderr.flush()
        return {
            "elapsed_sec": elapsed,
            "avg_step_sec": avg_step_time,
        }


class Trainer:
    """Encapsulates one training run."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        config: Dict[str, Any],
        device: torch.device,
        logger,
        hooks: Optional[List[HookBase]] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.logger = logger
        self.hooks = hooks or []
        amp_enabled = bool(config["runtime"].get("amp", True)) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    def train_one_epoch(self, dataloader, epoch: int) -> Dict[str, Any]:
        self.model.train()
        runtime_cfg = self.config["runtime"]
        model_cfg = self.config["model"]
        use_amp = bool(runtime_cfg.get("amp", True)) and self.device.type == "cuda"
        autocast_device = "cuda" if self.device.type == "cuda" else "cpu"
        last_tau = []
        weighted_sampler_enabled = bool(getattr(dataloader, "weighted_sampler_enabled", False))

        loss_meter = AverageMeter()
        corn_loss_meter = AverageMeter()
        center_reg_meter = AverageMeter()
        diff_reg_meter = AverageMeter()
        base_logits_meter = TensorStatsMeter()
        final_logits_meter = TensorStatsMeter()

        all_targets = []
        all_preds = []
        last_tau_info: Dict[str, Any] = {}
        progress = _ProgressTracker(
            total_steps=len(dataloader),
            desc=f"train epoch {epoch}",
            enabled=bool(runtime_cfg.get("enable_progress_bar", True)),
        )

        for hook in self.hooks:
            hook.before_epoch(self, epoch)

        for step, batch in enumerate(dataloader, start=1):
            for hook in self.hooks:
                hook.before_step(self, step)

            batch = move_batch_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=autocast_device, enabled=use_amp):
                outputs = self.model(
                    pre_image=batch["pre_image"],
                    post_image=batch["post_image"],
                    mask=batch["mask"],
                )
                loss_corn = corn_loss(outputs["logits"], batch["label"], num_classes=model_cfg["num_classes"])
                center_reg = outputs["tau_center_reg"]
                diff_reg = outputs["tau_diff_reg"]
                loss = (
                    loss_corn
                    + model_cfg["tau_reg_weight"] * center_reg
                    + model_cfg["tau_diff_weight"] * diff_reg
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if runtime_cfg.get("grad_clip", 0.0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), runtime_cfg["grad_clip"])
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = batch["label"].shape[0]
            loss_meter.update(loss.item(), batch_size)
            corn_loss_meter.update(loss_corn.item(), batch_size)
            center_reg_meter.update(center_reg.item(), batch_size)
            diff_reg_meter.update(diff_reg.item(), batch_size)
            base_logits_meter.update(outputs["base_logits"])
            final_logits_meter.update(outputs["logits"])

            all_targets.extend(batch["label"].detach().cpu().tolist())
            all_preds.extend(outputs["pred_labels"].detach().cpu().tolist())
            last_tau = outputs["tau"].detach().cpu().tolist()
            last_tau_info = outputs["tau_info"]
            progress.update(step, extra_text=f"loss {loss_meter.avg:.4f}")

            if step % runtime_cfg["print_freq"] == 0 or step == len(dataloader):
                step_metrics = compute_classification_metrics(
                    targets=all_targets,
                    preds=all_preds,
                    num_classes=model_cfg["num_classes"],
                    class_names=CLASS_NAMES,
                )
                tau_values = outputs["tau"].detach().cpu().tolist()
                self.logger.info(
                    "Epoch [%d] Step [%d/%d] loss=%.4f corn_loss=%.4f tau_center_reg=%.6f tau_diff_reg=%.6f acc=%.4f macro_f1=%.4f tau=%s",
                    epoch,
                    step,
                    len(dataloader),
                    loss_meter.avg,
                    corn_loss_meter.avg,
                    center_reg_meter.avg,
                    diff_reg_meter.avg,
                    step_metrics["accuracy"],
                    step_metrics["macro_f1"],
                    tau_values,
                )

            for hook in self.hooks:
                hook.after_step(self, step, outputs)

        timing = progress.close(final_text=f"loss {loss_meter.avg:.4f}")

        epoch_metrics = compute_classification_metrics(
            targets=all_targets,
            preds=all_preds,
            num_classes=model_cfg["num_classes"],
            class_names=CLASS_NAMES,
        )
        epoch_metrics.update(
            {
                "loss": loss_meter.avg,
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
                "tau_mode": last_tau_info.get("tau_mode", model_cfg.get("tau_mode", "per_threshold")),
                "per_class_f1": _format_per_class_f1(epoch_metrics["per_class"]),
                "weighted_sampler_enabled": weighted_sampler_enabled,
                "epoch_time_sec": timing["elapsed_sec"],
                "avg_step_time_sec": timing["avg_step_sec"],
            }
        )

        self.logger.info(
            "Train epoch [%d] loss=%.4f corn_loss=%.4f tau_center_reg=%.6f tau_diff_reg=%.6f "
            "acc=%.4f macro_f1=%.4f weighted_f1=%.4f per_class_f1=%s "
            "tau_raw=%s tau_positive_before_clamp=%s tau_clamped=%s "
            "base_logits_mean=%.4f base_logits_std=%.4f "
            "final_logits_mean=%.4f final_logits_std=%.4f decode_mode=%s tau_mode=%s weighted_sampler=%s "
            "epoch_time_sec=%.2f avg_step_time_sec=%.2f",
            epoch,
            epoch_metrics["loss"],
            epoch_metrics["corn_loss"],
            epoch_metrics["tau_center_reg"],
            epoch_metrics["tau_diff_reg"],
            epoch_metrics["accuracy"],
            epoch_metrics["macro_f1"],
            epoch_metrics["weighted_f1"],
            epoch_metrics["per_class_f1"],
            epoch_metrics["tau_raw"],
            epoch_metrics["tau_positive_before_clamp"],
            epoch_metrics["tau_clamped"],
            epoch_metrics["base_logits_mean"],
            epoch_metrics["base_logits_std"],
            epoch_metrics["final_logits_mean"],
            epoch_metrics["final_logits_std"],
            epoch_metrics["decode_mode"],
            epoch_metrics["tau_mode"],
            epoch_metrics["weighted_sampler_enabled"],
            epoch_metrics["epoch_time_sec"],
            epoch_metrics["avg_step_time_sec"],
        )

        if self.scheduler is not None:
            self.scheduler.step()

        for hook in self.hooks:
            hook.after_epoch(self, epoch, epoch_metrics)
        return epoch_metrics
