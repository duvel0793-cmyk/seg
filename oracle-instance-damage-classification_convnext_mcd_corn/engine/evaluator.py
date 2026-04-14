from __future__ import annotations

import time
from typing import Any

import torch
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from datasets.label_mapping import CLASS_NAMES
from losses.corn_loss import decode_threshold_count
from metrics.classification import compute_classification_metrics
from metrics.corn_decode import decode_corn_logits_with_thresholds, search_best_thresholds


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "pre_image": batch["pre_image"].to(device, non_blocking=True),
        "post_image": batch["post_image"].to(device, non_blocking=True),
        "instance_mask": batch["instance_mask"].to(device, non_blocking=True),
        "label": batch["label"].to(device, non_blocking=True),
        "meta": batch["meta"],
    }


def _merge_loss_summaries(loss_sums: dict[str, float], loss_dict: dict[str, Any], batch_size: int) -> None:
    for key, value in loss_dict.items():
        if key == "loss":
            scalar = float(value.detach().cpu().item())
        elif torch.is_tensor(value) and value.ndim == 0:
            scalar = float(value.detach().cpu().item())
        else:
            continue
        loss_sums[key] = loss_sums.get(key, 0.0) + (scalar * batch_size)


def _build_progress(iterable, *, enabled: bool, desc: str, total: int | None = None):
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, leave=False, dynamic_ncols=True)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    criterion=None,
    amp: bool = True,
    threshold_candidates: list[float] | None = None,
    fit_thresholds: bool = False,
    preset_thresholds: list[float] | None = None,
    show_progress: bool = False,
    progress_desc: str = "Val",
) -> dict[str, Any]:
    model.eval()
    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    tau_list: list[torch.Tensor] = []
    ce_logits_list: list[torch.Tensor] = []
    metas: list[dict[str, Any]] = []
    loss_sums: dict[str, float] = {}
    total_samples = 0

    use_amp = bool(amp and device.type == "cuda")
    start_time = time.perf_counter()
    num_batches = 0
    progress = _build_progress(
        loader,
        enabled=show_progress,
        desc=progress_desc,
        total=len(loader) if hasattr(loader, "__len__") else None,
    )
    for num_batches, batch in enumerate(progress, start=1):
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(batch["pre_image"], batch["post_image"], batch["instance_mask"])
            loss_dict = criterion(outputs, batch["label"]) if criterion is not None else None

        batch_size = int(batch["label"].shape[0])
        total_samples += batch_size
        logits_list.append(outputs["tau_adjusted_logits"].detach().cpu())
        labels_list.append(batch["label"].detach().cpu())
        tau_list.append(outputs["tau"].detach().cpu())
        ce_logits_list.append(outputs["ce_logits"].detach().cpu())
        metas.extend(batch["meta"])

        if loss_dict is not None:
            _merge_loss_summaries(loss_sums, loss_dict, batch_size)
            if show_progress and tqdm is not None:
                progress.set_postfix(
                    loss=f"{float(loss_dict['loss'].detach().cpu().item()):.4f}",
                    tau=f"{float(outputs['tau'].detach().mean().cpu().item()):.3f}",
                )

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    tau = torch.cat(tau_list, dim=0)
    ce_logits = torch.cat(ce_logits_list, dim=0)
    raw_preds = decode_threshold_count(logits).cpu().tolist()
    raw_metrics = compute_classification_metrics(labels.tolist(), raw_preds, CLASS_NAMES)

    calibrated_thresholds = list(preset_thresholds) if preset_thresholds is not None else None
    threshold_search = None
    if fit_thresholds:
        threshold_search = search_best_thresholds(
            logits=logits,
            labels=labels,
            candidates=threshold_candidates or [0.5],
            class_names=CLASS_NAMES,
        )
        calibrated_thresholds = list(threshold_search["best_thresholds"])

    calibrated_metrics = None
    calibrated_preds = None
    if calibrated_thresholds is not None:
        calibrated_preds = decode_corn_logits_with_thresholds(logits, calibrated_thresholds).cpu().tolist()
        calibrated_metrics = compute_classification_metrics(labels.tolist(), calibrated_preds, CLASS_NAMES)

    avg_losses = {key: value / max(total_samples, 1) for key, value in loss_sums.items()}
    elapsed_seconds = time.perf_counter() - start_time
    return {
        "logits": logits,
        "ce_logits": ce_logits,
        "labels": labels,
        "tau": tau,
        "raw_predictions": raw_preds,
        "raw_metrics": raw_metrics,
        "calibrated_predictions": calibrated_preds,
        "calibrated_metrics": calibrated_metrics,
        "threshold_search": threshold_search,
        "best_thresholds": calibrated_thresholds,
        "avg_losses": avg_losses,
        "meta": metas,
        "timing": {
            "elapsed_seconds": elapsed_seconds,
            "samples_per_second": float(total_samples / max(elapsed_seconds, 1e-6)),
            "num_batches": int(num_batches),
            "num_samples": int(total_samples),
        },
    }
