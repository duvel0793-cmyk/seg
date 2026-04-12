from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.engine.trainer import compute_loss_dict, get_autocast_context
from src.utils.metrics import RunningSegmentationMetrics
from src.utils.visualize import save_overlay, save_prediction_mask


def evaluate(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    damage_criterion=None,
    localization_criterion=None,
    lambda_loc: float = 0.3,
    prior_mode: str = "none",
    prior_alpha: float = 1.0,
    ignore_background_in_macro: bool = True,
    amp: bool = False,
    save_pred_masks: bool = False,
    save_overlay_images: bool = False,
    output_dir: Optional[str | Path] = None,
    desc: str = "Eval",
) -> dict[str, Any]:
    model.eval()
    use_amp = bool(amp and device.type == "cuda")
    metrics = RunningSegmentationMetrics(num_classes=5)

    running_total = 0.0
    running_damage = 0.0
    running_loc = 0.0
    sample_batches = 0

    output_path = Path(output_dir).resolve() if output_dir is not None else None
    mask_dir = output_path / "pred_masks" if output_path is not None else None
    overlay_dir = output_path / "overlays" if output_path is not None else None

    if save_pred_masks and mask_dir is not None:
        mask_dir.mkdir(parents=True, exist_ok=True)
    if save_overlay_images and overlay_dir is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    progress_bar = tqdm(dataloader, desc=desc, dynamic_ncols=True, leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            image = batch["image"].to(device, non_blocking=device.type == "cuda")
            target = batch["target"].to(device, non_blocking=device.type == "cuda")
            loc_target = batch["loc_target"].to(device, non_blocking=device.type == "cuda")
            prior_mask = batch["prior_mask"].to(device, non_blocking=device.type == "cuda")

            with get_autocast_context(device, enabled=use_amp):
                outputs = model(image)

                if damage_criterion is not None and localization_criterion is not None:
                    loss_dict = compute_loss_dict(
                        outputs=outputs,
                        batch={
                            "target": target,
                            "loc_target": loc_target,
                            "prior_mask": prior_mask,
                        },
                        damage_criterion=damage_criterion,
                        localization_criterion=localization_criterion,
                        lambda_loc=lambda_loc,
                        prior_mode=prior_mode,
                        prior_alpha=prior_alpha,
                    )
                    running_total += float(loss_dict["total_loss"].detach().item())
                    running_damage += float(loss_dict["damage_loss"].detach().item())
                    running_loc += float(loss_dict["loc_loss"].detach().item())
                    sample_batches += 1

            prediction = outputs["damage_logits"].argmax(dim=1)
            metrics.update(prediction=prediction, target=target)

            if save_pred_masks or save_overlay_images:
                prediction_np = prediction.detach().cpu().numpy().astype(np.uint8, copy=False)
                for index, sample_id in enumerate(batch["sample_id"]):
                    if save_pred_masks and mask_dir is not None:
                        save_prediction_mask(
                            prediction_np[index],
                            mask_dir / f"{sample_id}_pred.png",
                        )
                    if save_overlay_images and overlay_dir is not None:
                        post_image = Image.open(batch["post_path"][index]).convert("RGB")
                        save_overlay(
                            post_image,
                            prediction_np[index],
                            overlay_dir / f"{sample_id}_overlay.png",
                        )

            if sample_batches > 0:
                progress_bar.set_postfix(loss=f"{running_total / sample_batches:.4f}")

    summary = metrics.summarize(ignore_background_in_macro=ignore_background_in_macro)
    divisor = max(sample_batches, 1)
    summary["loss"] = running_total / divisor if sample_batches > 0 else None
    summary["damage_loss"] = running_damage / divisor if sample_batches > 0 else None
    summary["loc_loss"] = running_loc / divisor if sample_batches > 0 else None
    return summary
