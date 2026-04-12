from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from src.losses.damage_losses import build_prior_weight_map


def get_autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def compute_loss_dict(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    damage_criterion,
    localization_criterion,
    lambda_loc: float,
    prior_mode: str,
    prior_alpha: float,
) -> dict[str, torch.Tensor]:
    damage_logits = outputs["damage_logits"]
    loc_logits = outputs["loc_logits"]
    damage_target = batch["target"]
    loc_target = batch["loc_target"]

    weight_map = None
    if prior_mode == "loss_weight":
        weight_map = build_prior_weight_map(batch["prior_mask"], alpha=prior_alpha)

    damage_loss = damage_criterion(damage_logits, damage_target, weight_map=weight_map)
    loc_loss = localization_criterion(loc_logits, loc_target)
    total_loss = damage_loss + (lambda_loc * loc_loss)

    return {
        "total_loss": total_loss,
        "damage_loss": damage_loss,
        "loc_loss": loc_loss,
    }


def train_one_epoch(
    model: torch.nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    damage_criterion,
    localization_criterion,
    epoch_index: int,
    total_epochs: int,
    train_config: dict[str, Any],
    loss_config: dict[str, Any],
    scaler,
):
    model.train()
    use_amp = bool(train_config.get("amp", False) and device.type == "cuda")
    grad_clip = float(train_config.get("grad_clip", 0.0) or 0.0)
    print_freq = max(1, int(train_config.get("print_freq", 10)))

    running_total = 0.0
    running_damage = 0.0
    running_loc = 0.0
    sample_batches = 0

    progress_bar = tqdm(
        dataloader,
        desc=f"Train {epoch_index}/{total_epochs}",
        dynamic_ncols=True,
        leave=False,
    )

    for step, batch in enumerate(progress_bar, start=1):
        image = batch["image"].to(device, non_blocking=device.type == "cuda")
        target = batch["target"].to(device, non_blocking=device.type == "cuda")
        loc_target = batch["loc_target"].to(device, non_blocking=device.type == "cuda")
        prior_mask = batch["prior_mask"].to(device, non_blocking=device.type == "cuda")

        optimizer.zero_grad(set_to_none=True)

        with get_autocast_context(device, enabled=use_amp):
            outputs = model(image)
            loss_dict = compute_loss_dict(
                outputs=outputs,
                batch={
                    "target": target,
                    "loc_target": loc_target,
                    "prior_mask": prior_mask,
                },
                damage_criterion=damage_criterion,
                localization_criterion=localization_criterion,
                lambda_loc=float(loss_config["lambda_loc"]),
                prior_mode=str(loss_config["prior_mode"]),
                prior_alpha=float(loss_config["prior_alpha"]),
            )

        total_loss = loss_dict["total_loss"]
        scaler.scale(total_loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        running_total += float(loss_dict["total_loss"].detach().item())
        running_damage += float(loss_dict["damage_loss"].detach().item())
        running_loc += float(loss_dict["loc_loss"].detach().item())
        sample_batches += 1

        if step % print_freq == 0 or step == len(dataloader):
            progress_bar.set_postfix(
                loss=f"{running_total / sample_batches:.4f}",
                damage=f"{running_damage / sample_batches:.4f}",
                loc=f"{running_loc / sample_batches:.4f}",
            )

    divisor = max(sample_batches, 1)
    return {
        "loss": running_total / divisor,
        "damage_loss": running_damage / divisor,
        "loc_loss": running_loc / divisor,
    }
