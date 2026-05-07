from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.misc import SEVERITY_TARGETS, get_enabled_scale_names


def resolve_change_block_type(model_cfg: dict[str, Any]) -> str:
    block_type = str(model_cfg.get("change_block_type", "auto")).lower()
    if block_type not in {"auto", "none", "legacy", "bdfm_lite"}:
        raise ValueError(f"Unsupported change_block_type='{model_cfg.get('change_block_type')}'")
    if block_type == "auto":
        if bool(model_cfg.get("enable_damage_bdfm_lite", False)):
            return "bdfm_lite"
        if bool(model_cfg.get("enable_damage_aware_block", True)):
            return "legacy"
        return "none"
    return block_type


def corn_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.numel() == 0 or labels.numel() == 0:
        return logits.new_tensor(0.0)
    num_classes = logits.size(1) + 1
    levels = []
    for threshold in range(num_classes - 1):
        levels.append((labels > threshold).float())
    level_targets = torch.stack(levels, dim=1)
    return F.binary_cross_entropy_with_logits(logits.float(), level_targets, reduction="mean")


def _dataset_class_weights_from_counts(counts: list[int] | tuple[int, ...], mode: str, device: torch.device) -> torch.Tensor | None:
    if not counts:
        return None
    counts_tensor = torch.as_tensor(counts, dtype=torch.float32, device=device)
    valid = counts_tensor > 0
    if not valid.any():
        return None
    weights = torch.ones_like(counts_tensor)
    if mode == "dataset_inverse":
        weights[valid] = 1.0 / counts_tensor[valid]
    elif mode == "dataset_sqrt_inverse":
        weights[valid] = 1.0 / torch.sqrt(counts_tensor[valid])
    else:
        return None
    weights = weights / weights.mean().clamp_min(1e-8)
    return weights


def resolve_final_class_weights(loss_cfg: dict[str, Any], labels: torch.Tensor, num_classes: int) -> torch.Tensor | None:
    mode = str(loss_cfg.get("final_class_weight_mode", "auto")).lower()
    if mode == "auto":
        mode = "batch_inverse" if bool(loss_cfg.get("balanced_final_ce", False)) else "none"
    if mode == "none":
        return None
    if mode == "batch_inverse":
        counts = torch.bincount(labels, minlength=num_classes).float()
        valid = counts > 0
        weights = torch.ones_like(counts)
        if valid.any():
            weights[valid] = counts[valid].sum() / (counts[valid] * valid.sum())
            weights = weights / weights.mean().clamp_min(1e-8)
        return weights
    if mode in {"dataset_inverse", "dataset_sqrt_inverse"}:
        counts = loss_cfg.get("dataset_class_counts", [])
        return _dataset_class_weights_from_counts(counts, mode, device=labels.device)
    raise ValueError(f"Unsupported final_class_weight_mode='{mode}'")


def probability_ce_loss(probabilities: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor | None = None) -> torch.Tensor:
    log_probs = torch.log(probabilities.clamp_min(1e-8))
    return F.nll_loss(log_probs, labels, weight=class_weights, reduction="mean")


def _resolve_train_target(config: dict[str, Any]) -> str:
    experiment_cfg = config.get("experiment", {}) if isinstance(config.get("experiment"), dict) else {}
    return str(experiment_cfg.get("train_target", "instance")).lower()


def _build_corn_targets(labels: torch.Tensor, num_thresholds: int) -> torch.Tensor:
    return torch.stack([(labels > threshold).float() for threshold in range(num_thresholds)], dim=1)


def _resize_valid_mask(mask: torch.Tensor | None, spatial_size: tuple[int, int], device: torch.device) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    return F.interpolate(mask.float(), size=spatial_size, mode="nearest").to(device=device)


def _masked_average(loss_map: torch.Tensor, valid_mask: torch.Tensor, normalize_by_mask_area: bool) -> torch.Tensor:
    if loss_map.numel() == 0 or valid_mask.numel() == 0:
        return loss_map.new_tensor(0.0)
    valid_mask = valid_mask.float()
    per_sample_sum = (loss_map * valid_mask).flatten(1).sum(dim=1)
    if bool(normalize_by_mask_area):
        per_sample_denominator = valid_mask.flatten(1).sum(dim=1).clamp_min(1.0)
        per_sample_value = per_sample_sum / per_sample_denominator
        active = per_sample_denominator > 0
        if active.any():
            return per_sample_value[active].mean()
        return loss_map.new_tensor(0.0)
    denominator = valid_mask.sum().clamp_min(1.0)
    return per_sample_sum.sum() / denominator


def pixel_two_stage_dense_loss(
    *,
    damage_binary_logit: torch.Tensor | None,
    severity_corn_logits: torch.Tensor | None,
    valid_mask: torch.Tensor | None,
    labels: torch.Tensor,
    binary_weight: float,
    severity_weight: float,
    normalize_by_mask_area: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    zero = labels.new_tensor(0.0, dtype=torch.float32)
    if damage_binary_logit is None or severity_corn_logits is None or valid_mask is None:
        return zero, zero
    valid_mask = valid_mask.to(device=damage_binary_logit.device, dtype=torch.float32)
    binary_target = (labels > 0).float().view(-1, 1, 1, 1).expand_as(damage_binary_logit)
    binary_loss_map = F.binary_cross_entropy_with_logits(
        damage_binary_logit.float(),
        binary_target,
        reduction="none",
    )
    loss_binary = float(binary_weight) * _masked_average(binary_loss_map, valid_mask, normalize_by_mask_area)

    damaged_samples = labels > 0
    if not damaged_samples.any():
        return loss_binary, zero
    severity_labels = (labels[damaged_samples] - 1).long()
    severity_logits = severity_corn_logits[damaged_samples]
    severity_valid_mask = valid_mask[damaged_samples].expand(-1, severity_logits.size(1), -1, -1)
    severity_targets = _build_corn_targets(severity_labels, num_thresholds=severity_logits.size(1)).to(
        device=severity_logits.device,
        dtype=severity_logits.dtype,
    )
    severity_targets = severity_targets[:, :, None, None].expand_as(severity_logits)
    severity_loss_map = F.binary_cross_entropy_with_logits(
        severity_logits.float(),
        severity_targets.float(),
        reduction="none",
    )
    loss_severity = float(severity_weight) * _masked_average(
        severity_loss_map.mean(dim=1, keepdim=True),
        severity_valid_mask[:, :1],
        normalize_by_mask_area,
    )
    return loss_binary, loss_severity


def pixel_flat_corn_dense_loss(
    *,
    corn_logits: torch.Tensor | None,
    valid_mask: torch.Tensor | None,
    labels: torch.Tensor,
    normalize_by_mask_area: bool,
) -> torch.Tensor:
    zero = labels.new_tensor(0.0, dtype=torch.float32)
    if corn_logits is None or valid_mask is None:
        return zero
    valid_mask = valid_mask.to(device=corn_logits.device, dtype=torch.float32)
    num_thresholds = corn_logits.size(1)
    targets = _build_corn_targets(labels.long(), num_thresholds=num_thresholds).to(device=corn_logits.device, dtype=corn_logits.dtype)
    targets = targets[:, :, None, None].expand_as(corn_logits)
    loss_map = F.binary_cross_entropy_with_logits(corn_logits.float(), targets.float(), reduction="none")
    return _masked_average(loss_map.mean(dim=1, keepdim=True), valid_mask, normalize_by_mask_area)


def pixel_entropy_regularization(
    pixel_probabilities: torch.Tensor | None,
    valid_mask: torch.Tensor | None,
    normalize_by_mask_area: bool,
) -> torch.Tensor:
    if pixel_probabilities is None or valid_mask is None:
        device = pixel_probabilities.device if pixel_probabilities is not None else valid_mask.device if valid_mask is not None else "cpu"
        return torch.tensor(0.0, device=device)
    entropy_map = -(pixel_probabilities.clamp_min(1e-8) * torch.log(pixel_probabilities.clamp_min(1e-8))).sum(dim=1, keepdim=True)
    return _masked_average(entropy_map, valid_mask.float(), normalize_by_mask_area)


def masked_evidence_pixel_loss(
    logits: torch.Tensor | None,
    target: torch.Tensor | None,
    labels: torch.Tensor,
    ignore_index: int,
    class_mode: str,
) -> torch.Tensor:
    if logits is None or target is None:
        device = logits.device if logits is not None else target.device if target is not None else "cpu"
        return torch.tensor(0.0, device=device)
    if target.ndim == 3:
        target = target.unsqueeze(1).float()
    if target.ndim == 4:
        target = F.interpolate(target, size=logits.shape[-2:], mode="nearest").squeeze(1)
    pixel_loss = F.cross_entropy(logits.float(), target.long(), ignore_index=int(ignore_index), reduction="none")
    valid = target.long() != int(ignore_index)
    sample_losses = []
    sample_weights = []
    class_mode = str(class_mode).lower()
    for idx in range(logits.size(0)):
        valid_pixels = valid[idx]
        if int(valid_pixels.sum().item()) <= 0:
            continue
        sample_loss = pixel_loss[idx][valid_pixels].mean()
        sample_weight = 1.0
        if class_mode == "weak_minor_major" and int(labels[idx].item()) in {1, 2}:
            sample_weight = 0.25
        sample_losses.append(sample_loss * sample_weight)
        sample_weights.append(sample_weight)
    if not sample_losses:
        return labels.new_tensor(0.0, dtype=torch.float32)
    return torch.stack(sample_losses).sum() / max(float(sum(sample_weights)), 1e-8)


def evidence_mil_loss(logits: torch.Tensor | None, mask: torch.Tensor | None, labels: torch.Tensor, topk_ratio: float = 0.1) -> torch.Tensor:
    if logits is None or mask is None:
        return labels.new_tensor(0.0, dtype=torch.float32)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    mask_resized = F.interpolate(mask.float(), size=logits.shape[-2:], mode="nearest")
    prob = logits.softmax(dim=1)
    losses = []
    for idx in range(logits.size(0)):
        valid = mask_resized[idx, 0] > 0.5
        if int(valid.sum().item()) <= 0:
            continue
        flat = prob[idx].permute(1, 2, 0)[valid]
        k = max(1, int(round(flat.size(0) * topk_ratio)))
        pooled = torch.topk(flat, k=k, dim=0).values.mean(dim=0)
        losses.append(F.nll_loss(torch.log(pooled.clamp_min(1e-8)).unsqueeze(0), labels[idx : idx + 1], reduction="mean"))
    if not losses:
        return labels.new_tensor(0.0, dtype=torch.float32)
    return torch.stack(losses).mean()


def severity_map_regression_loss(severity_map: torch.Tensor | None, mask: torch.Tensor | None, labels: torch.Tensor) -> torch.Tensor:
    if severity_map is None or mask is None:
        return labels.new_tensor(0.0, dtype=torch.float32)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    mask_resized = F.interpolate(mask.float(), size=severity_map.shape[-2:], mode="nearest")
    target_values = SEVERITY_TARGETS.to(device=labels.device)[labels]
    pred_values = torch.sigmoid(severity_map.float())
    losses = []
    for idx in range(severity_map.size(0)):
        valid = mask_resized[idx, 0] > 0.5
        if int(valid.sum().item()) <= 0:
            continue
        values = pred_values[idx, 0][valid]
        k = max(1, int(round(values.numel() * 0.1)))
        pooled = 0.5 * (values.mean() + torch.topk(values, k=k).values.mean())
        losses.append(F.smooth_l1_loss(pooled, target_values[idx], reduction="mean"))
    if not losses:
        return labels.new_tensor(0.0, dtype=torch.float32)
    return torch.stack(losses).mean()


def compute_effective_evidence_weight(
    *,
    base_weight: float,
    loss_cfg: dict[str, Any],
    epoch_index: int,
) -> float:
    schedule_cfg = loss_cfg.get("evidence_schedule", {}) if isinstance(loss_cfg.get("evidence_schedule"), dict) else {}
    if not bool(schedule_cfg.get("enabled", False)):
        return float(base_weight)
    epoch_number = int(epoch_index) + 1
    warmup_epochs = int(schedule_cfg.get("warmup_epochs", 0))
    ramp_epochs = int(schedule_cfg.get("ramp_epochs", 0))
    if epoch_number <= warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return float(base_weight)
    ramp_position = epoch_number - warmup_epochs
    if ramp_position >= ramp_epochs:
        return float(base_weight)
    scale = float(ramp_position) / float(ramp_epochs)
    return float(base_weight) * max(0.0, min(scale, 1.0))


def compute_gate_losses(
    change_gate: torch.Tensor | None,
    mask: torch.Tensor | None,
    labels: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if change_gate is None or mask is None:
        zero = labels.new_tensor(0.0, dtype=torch.float32)
        return zero, zero, zero
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    mask_resized = F.interpolate(mask.float(), size=change_gate.shape[-2:], mode="nearest")
    gate = change_gate.float()
    bg_losses = []
    contrast_losses = []
    unchanged_losses = []
    for idx in range(gate.size(0)):
        fg_mask = mask_resized[idx, 0] > 0.5
        bg_mask = ~fg_mask
        gate_map = gate[idx, 0]
        fg_values = gate_map[fg_mask]
        bg_values = gate_map[bg_mask]
        if bg_values.numel() > 0:
            bg_losses.append(bg_values.mean())
        if int(labels[idx].item()) > 0 and fg_values.numel() > 0 and bg_values.numel() > 0:
            contrast_losses.append(F.relu(float(margin) - (fg_values.mean() - bg_values.mean())))
        if int(labels[idx].item()) == 0 and fg_values.numel() > 0:
            unchanged_losses.append(fg_values.mean())
    zero = labels.new_tensor(0.0, dtype=torch.float32)
    loss_bg = torch.stack(bg_losses).mean() if bg_losses else zero
    loss_contrast = torch.stack(contrast_losses).mean() if contrast_losses else zero
    loss_unchanged = torch.stack(unchanged_losses).mean() if unchanged_losses else zero
    return loss_bg, loss_contrast, loss_unchanged


class HierarchicalLoss(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.loss_cfg = config["loss"]
        self.model_cfg = config["model"]

    def forward(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
        *,
        epoch_index: int = 0,
        batch_idx: int = 0,
        is_train: bool = False,
    ) -> dict[str, torch.Tensor]:
        labels = batch["label"]
        damaged_target = (labels > 0).float()
        train_target = _resolve_train_target(self.config)
        pixel_loss_cfg = self.loss_cfg.get("pixel_line", {}) if isinstance(self.loss_cfg.get("pixel_line"), dict) else {}
        pixel_line_loss_enabled = bool(pixel_loss_cfg.get("enabled", False))
        loss_terms: dict[str, torch.Tensor] = {}
        use_hierarchical_head = bool(self.model_cfg.get("use_hierarchical_head", True))
        use_evidence_head = bool(self.model_cfg.get("use_evidence_head", True))
        use_global_corn_aux = bool(self.model_cfg.get("use_global_corn_aux", True))
        enable_minor_boundary_aux = bool(self.model_cfg.get("enable_minor_boundary_aux", False))
        use_structural_two_stage_head = bool(self.model_cfg.get("use_structural_two_stage_head", False))
        use_conditional_review_head = bool(self.model_cfg.get("use_conditional_review_head", False))
        use_severity_aware_scale_router = bool(self.model_cfg.get("use_severity_aware_scale_router", False))
        use_scale_aux_fusion_head = bool(self.model_cfg.get("use_scale_aux_fusion_head", False))
        effective_change_block_type = resolve_change_block_type(self.model_cfg)
        enable_damage_aware_block = effective_change_block_type != "none"
        enable_change_gate = bool(self.model_cfg.get("enable_change_gate", True)) and enable_damage_aware_block
        active_scales = get_enabled_scale_names(self.config)

        loss_terms["loss_corn"] = corn_loss(outputs["corn_logits"], labels) if outputs.get("corn_logits") is not None else labels.new_tensor(0.0, dtype=torch.float32)

        if use_hierarchical_head and outputs["damage_binary_logit"] is not None:
            if bool(self.loss_cfg.get("balanced_binary_loss", False)):
                pos_count = damaged_target.sum()
                neg_count = damaged_target.numel() - pos_count
                pos_weight = (neg_count / pos_count.clamp_min(1.0)).to(dtype=torch.float32)
                loss_terms["loss_binary_damage"] = F.binary_cross_entropy_with_logits(
                    outputs["damage_binary_logit"].float(),
                    damaged_target,
                    pos_weight=pos_weight,
                    reduction="mean",
                )
            else:
                loss_terms["loss_binary_damage"] = F.binary_cross_entropy_with_logits(
                    outputs["damage_binary_logit"].float(),
                    damaged_target,
                    reduction="mean",
                )
        else:
            loss_terms["loss_binary_damage"] = labels.new_tensor(0.0, dtype=torch.float32)

        damaged_mask = labels > 0
        if use_hierarchical_head and outputs["severity_corn_logits"] is not None and damaged_mask.any():
            severity_labels = labels[damaged_mask] - 1
            loss_terms["loss_severity_corn"] = corn_loss(outputs["severity_corn_logits"][damaged_mask], severity_labels)
        else:
            loss_terms["loss_severity_corn"] = labels.new_tensor(0.0, dtype=torch.float32)

        final_class_weights = resolve_final_class_weights(self.loss_cfg, labels, num_classes=4)
        loss_terms["loss_final_ce"] = probability_ce_loss(outputs["class_probabilities"], labels, class_weights=final_class_weights)

        if use_global_corn_aux and outputs.get("global_corn_aux_logits") is not None:
            loss_terms["loss_global_corn_aux"] = corn_loss(outputs["global_corn_aux_logits"], labels)
        else:
            loss_terms["loss_global_corn_aux"] = labels.new_tensor(0.0, dtype=torch.float32)

        loss_terms["loss_minor_no_aux"] = labels.new_tensor(0.0, dtype=torch.float32)
        if enable_minor_boundary_aux and outputs.get("minor_no_aux_logit") is not None:
            minor_no_mask = (labels == 0) | (labels == 1)
            if minor_no_mask.any():
                minor_no_target = (labels[minor_no_mask] == 1).float()
                loss_terms["loss_minor_no_aux"] = F.binary_cross_entropy_with_logits(
                    outputs["minor_no_aux_logit"][minor_no_mask].float(),
                    minor_no_target,
                    reduction="mean",
                )

        loss_terms["loss_minor_major_aux"] = labels.new_tensor(0.0, dtype=torch.float32)
        if enable_minor_boundary_aux and outputs.get("minor_major_aux_logit") is not None:
            minor_major_mask = (labels == 1) | (labels == 2)
            if minor_major_mask.any():
                minor_major_target = (labels[minor_major_mask] == 2).float()
                loss_terms["loss_minor_major_aux"] = F.binary_cross_entropy_with_logits(
                    outputs["minor_major_aux_logit"][minor_major_mask].float(),
                    minor_major_target,
                    reduction="mean",
                )

        loss_terms["loss_structural_binary"] = labels.new_tensor(0.0, dtype=torch.float32)
        if use_structural_two_stage_head and outputs.get("structural_binary_logit") is not None:
            structural_target = (labels >= 2).float()
            loss_terms["loss_structural_binary"] = F.binary_cross_entropy_with_logits(
                outputs["structural_binary_logit"].float(),
                structural_target,
                reduction="mean",
            )

        loss_terms["loss_low_stage"] = labels.new_tensor(0.0, dtype=torch.float32)
        if use_structural_two_stage_head and outputs.get("low_stage_logit") is not None:
            low_mask = (labels == 0) | (labels == 1)
            if low_mask.any():
                low_target = (labels[low_mask] == 1).float()
                loss_terms["loss_low_stage"] = F.binary_cross_entropy_with_logits(
                    outputs["low_stage_logit"][low_mask].float(),
                    low_target,
                    reduction="mean",
                )

        loss_terms["loss_high_stage"] = labels.new_tensor(0.0, dtype=torch.float32)
        if use_structural_two_stage_head and outputs.get("high_stage_logit") is not None:
            high_mask = (labels == 2) | (labels == 3)
            if high_mask.any():
                high_target = (labels[high_mask] == 3).float()
                loss_terms["loss_high_stage"] = F.binary_cross_entropy_with_logits(
                    outputs["high_stage_logit"][high_mask].float(),
                    high_target,
                    reduction="mean",
                )

        loss_terms["loss_scale_router_ce"] = labels.new_tensor(0.0, dtype=torch.float32)
        if use_severity_aware_scale_router and outputs.get("scale_router_logits") is not None:
            loss_terms["loss_scale_router_ce"] = F.cross_entropy(outputs["scale_router_logits"].float(), labels, reduction="mean")

        loss_terms["loss_conditional_review_ce"] = labels.new_tensor(0.0, dtype=torch.float32)
        if use_conditional_review_head and outputs.get("conditional_review_logits") is not None:
            loss_terms["loss_conditional_review_ce"] = F.cross_entropy(
                outputs["conditional_review_logits"].float(),
                labels,
                reduction="mean",
            )

        loss_terms["loss_conditional_review_low_ce"] = labels.new_tensor(0.0, dtype=torch.float32)
        low_review_logits = outputs.get("conditional_review_low_logits")
        if use_conditional_review_head and low_review_logits is not None:
            per_sample_ce = F.cross_entropy(low_review_logits.float(), labels, reduction="none")
            sample_weight = torch.ones_like(per_sample_ce)
            low_focus_mask = labels <= 1
            sample_weight[low_focus_mask] = float(self.loss_cfg.get("conditional_review_low_focus_weight", 2.0))
            loss_terms["loss_conditional_review_low_ce"] = (per_sample_ce * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)

        loss_terms["loss_conditional_review_high_ce"] = labels.new_tensor(0.0, dtype=torch.float32)
        high_review_logits = outputs.get("conditional_review_high_logits")
        if use_conditional_review_head and high_review_logits is not None:
            per_sample_ce = F.cross_entropy(high_review_logits.float(), labels, reduction="none")
            sample_weight = torch.ones_like(per_sample_ce)
            high_focus_mask = labels >= 2
            sample_weight[high_focus_mask] = float(self.loss_cfg.get("conditional_review_high_focus_weight", 2.0))
            loss_terms["loss_conditional_review_high_ce"] = (per_sample_ce * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)

        loss_terms["loss_conditional_review_delta_l2"] = labels.new_tensor(0.0, dtype=torch.float32)
        if use_conditional_review_head and low_review_logits is not None and high_review_logits is not None:
            loss_terms["loss_conditional_review_delta_l2"] = (
                low_review_logits.float().pow(2).mean() + high_review_logits.float().pow(2).mean()
            )

        loss_terms["loss_scale_aux_tight"] = labels.new_tensor(0.0, dtype=torch.float32)
        if use_scale_aux_fusion_head and outputs.get("scale_aux_tight_logits") is not None:
            loss_terms["loss_scale_aux_tight"] = corn_loss(outputs["scale_aux_tight_logits"], labels)

        loss_terms["loss_scale_aux_context"] = labels.new_tensor(0.0, dtype=torch.float32)
        if use_scale_aux_fusion_head and outputs.get("scale_aux_context_logits") is not None:
            loss_terms["loss_scale_aux_context"] = corn_loss(outputs["scale_aux_context_logits"], labels)

        loss_terms["loss_scale_aux_neighborhood"] = labels.new_tensor(0.0, dtype=torch.float32)
        if use_scale_aux_fusion_head and outputs.get("scale_aux_neighborhood_logits") is not None:
            loss_terms["loss_scale_aux_neighborhood"] = corn_loss(outputs["scale_aux_neighborhood_logits"], labels)

        loss_terms["loss_scale_aux_fusion_ce"] = labels.new_tensor(0.0, dtype=torch.float32)
        if use_scale_aux_fusion_head and outputs.get("scale_aux_fused_class_probabilities") is not None:
            loss_terms["loss_scale_aux_fusion_ce"] = probability_ce_loss(outputs["scale_aux_fused_class_probabilities"], labels)

        scale_aux_weights = outputs.get("scale_aux_fusion_weights")
        loss_terms["loss_scale_aux_weight_entropy"] = labels.new_tensor(0.0, dtype=torch.float32)
        loss_terms["loss_scale_aux_weight_prior_kl"] = labels.new_tensor(0.0, dtype=torch.float32)
        if use_scale_aux_fusion_head and scale_aux_weights is not None:
            eps = 1.0e-8
            weight_log = torch.log(scale_aux_weights.clamp_min(eps))
            loss_terms["loss_scale_aux_weight_entropy"] = -(
                scale_aux_weights * weight_log
            ).sum(dim=1).mean()
            prior = torch.as_tensor(
                self.loss_cfg.get("scale_aux_weight_prior", [0.85, 0.06, 0.06, 0.03]),
                dtype=scale_aux_weights.dtype,
                device=scale_aux_weights.device,
            ).clamp_min(eps)
            prior = prior / prior.sum().clamp_min(eps)
            loss_terms["loss_scale_aux_weight_prior_kl"] = (
                scale_aux_weights * (weight_log - torch.log(prior.clamp_min(eps)))
            ).sum(dim=1).mean()

        evidence_pixel_weight_eff = compute_effective_evidence_weight(
            base_weight=float(self.loss_cfg.get("evidence_pixel_weight", 0.0)),
            loss_cfg=self.loss_cfg,
            epoch_index=epoch_index,
        ) if use_evidence_head else 0.0
        evidence_mil_weight_eff = compute_effective_evidence_weight(
            base_weight=float(self.loss_cfg.get("evidence_mil_weight", 0.0)),
            loss_cfg=self.loss_cfg,
            epoch_index=epoch_index,
        ) if use_evidence_head else 0.0
        severity_map_weight_eff = compute_effective_evidence_weight(
            base_weight=float(self.loss_cfg.get("severity_map_weight", 0.0)),
            loss_cfg=self.loss_cfg,
            epoch_index=epoch_index,
        ) if use_evidence_head else 0.0
        loss_terms["evidence_pixel_weight_eff"] = labels.new_tensor(evidence_pixel_weight_eff, dtype=torch.float32)
        loss_terms["evidence_mil_weight_eff"] = labels.new_tensor(evidence_mil_weight_eff, dtype=torch.float32)
        loss_terms["severity_map_weight_eff"] = labels.new_tensor(severity_map_weight_eff, dtype=torch.float32)

        ignore_index = int(self.loss_cfg.get("ignore_index", 255))
        evidence_pixel_losses = []
        evidence_mil_losses = []
        severity_map_losses = []
        damage_aux_losses = []
        gate_bg_losses = []
        gate_contrast_losses = []
        unchanged_losses = []
        evidence_pixel_class_mode = str(self.loss_cfg.get("evidence_pixel_class_mode", "dense_all"))
        evidence_pixel_every_n_steps = max(1, int(self.loss_cfg.get("evidence_pixel_every_n_steps", 1)))
        should_compute_evidence_pixel = (not is_train) or (batch_idx % evidence_pixel_every_n_steps == 0)
        for scale_name in active_scales:
            logits = outputs.get(f"evidence_logits_{scale_name}")
            severity_map = outputs.get(f"severity_map_{scale_name}")
            target = batch.get(f"post_target_{scale_name}")
            mask = batch.get(f"mask_{scale_name}")
            if use_evidence_head and should_compute_evidence_pixel:
                evidence_pixel_losses.append(masked_evidence_pixel_loss(logits, target, labels, ignore_index, evidence_pixel_class_mode))
            else:
                evidence_pixel_losses.append(labels.new_tensor(0.0, dtype=torch.float32))
            if use_evidence_head and logits is not None and mask is not None:
                evidence_mil_losses.append(
                    evidence_mil_loss(logits, mask, labels, topk_ratio=float(self.model_cfg.get("evidence_topk_ratio", 0.1)))
                )
            if use_evidence_head and severity_map is not None and mask is not None:
                severity_map_losses.append(severity_map_regression_loss(severity_map, mask, labels))
            damage_aux = outputs["damage_aux_scores"].get(scale_name)
            if enable_damage_aware_block and damage_aux is not None:
                damage_aux_prob = damage_aux.float().clamp(1e-6, 1.0 - 1e-6)
                damage_aux_losses.append(
                    F.binary_cross_entropy_with_logits(
                        torch.logit(damage_aux_prob),
                        damaged_target,
                        reduction="mean",
                    )
                )
            if enable_change_gate:
                gate_bg, gate_contrast, unchanged = compute_gate_losses(
                    outputs.get("change_gates", {}).get(scale_name),
                    mask,
                    labels,
                    margin=float(self.loss_cfg.get("gate_contrast_margin", 0.15)),
                )
            else:
                gate_bg = labels.new_tensor(0.0, dtype=torch.float32)
                gate_contrast = labels.new_tensor(0.0, dtype=torch.float32)
                unchanged = labels.new_tensor(0.0, dtype=torch.float32)
            gate_bg_losses.append(gate_bg)
            gate_contrast_losses.append(gate_contrast)
            unchanged_losses.append(unchanged)

        loss_terms["loss_evidence_pixel"] = torch.stack(evidence_pixel_losses).mean() if evidence_pixel_losses else labels.new_tensor(0.0, dtype=torch.float32)
        loss_terms["loss_evidence_mil"] = torch.stack(evidence_mil_losses).mean() if evidence_mil_losses else labels.new_tensor(0.0, dtype=torch.float32)
        loss_terms["loss_severity_map"] = torch.stack(severity_map_losses).mean() if severity_map_losses else labels.new_tensor(0.0, dtype=torch.float32)
        loss_terms["loss_damage_aux"] = torch.stack(damage_aux_losses).mean() if damage_aux_losses else labels.new_tensor(0.0, dtype=torch.float32)
        loss_terms["loss_gate_bg"] = torch.stack(gate_bg_losses).mean() if gate_bg_losses else labels.new_tensor(0.0, dtype=torch.float32)
        loss_terms["loss_gate_contrast"] = torch.stack(gate_contrast_losses).mean() if gate_contrast_losses else labels.new_tensor(0.0, dtype=torch.float32)
        loss_terms["loss_unchanged"] = torch.stack(unchanged_losses).mean() if unchanged_losses else labels.new_tensor(0.0, dtype=torch.float32)

        pixel_valid_mask = outputs.get("pixel_valid_mask")
        if pixel_valid_mask is not None:
            pixel_valid_mask = pixel_valid_mask.to(device=labels.device)
        pixel_enabled_in_outputs = bool(outputs.get("pixel_line_enabled", False))
        normalize_pixel_by_mask = bool(pixel_loss_cfg.get("normalize_by_mask_area", True))
        loss_terms["loss_pixel_binary_dense"] = labels.new_tensor(0.0, dtype=torch.float32)
        loss_terms["loss_pixel_severity_dense"] = labels.new_tensor(0.0, dtype=torch.float32)
        if pixel_line_loss_enabled and pixel_enabled_in_outputs and pixel_valid_mask is not None:
            loss_binary_dense, loss_severity_dense = pixel_two_stage_dense_loss(
                damage_binary_logit=outputs.get("pixel_damage_binary_logit"),
                severity_corn_logits=outputs.get("pixel_severity_corn_logits"),
                valid_mask=pixel_valid_mask,
                labels=labels,
                binary_weight=float(pixel_loss_cfg.get("binary_weight", 1.0)),
                severity_weight=float(pixel_loss_cfg.get("severity_weight", 1.0)),
                normalize_by_mask_area=normalize_pixel_by_mask,
            )
            loss_terms["loss_pixel_binary_dense"] = loss_binary_dense
            loss_terms["loss_pixel_severity_dense"] = loss_severity_dense
            if outputs.get("pixel_corn_logits") is not None:
                loss_terms["loss_pixel_binary_dense"] = labels.new_tensor(0.0, dtype=torch.float32)
                loss_terms["loss_pixel_severity_dense"] = pixel_flat_corn_dense_loss(
                    corn_logits=outputs.get("pixel_corn_logits"),
                    valid_mask=pixel_valid_mask,
                    labels=labels,
                    normalize_by_mask_area=normalize_pixel_by_mask,
                )
        loss_terms["loss_pixel_dense"] = loss_terms["loss_pixel_binary_dense"] + loss_terms["loss_pixel_severity_dense"]
        loss_terms["loss_pixel_agg"] = (
            probability_ce_loss(outputs["pixel_instance_probabilities"], labels)
            if pixel_line_loss_enabled and pixel_enabled_in_outputs and outputs.get("pixel_instance_probabilities") is not None
            else labels.new_tensor(0.0, dtype=torch.float32)
        )
        loss_terms["loss_pixel_entropy"] = (
            pixel_entropy_regularization(
                outputs.get("pixel_class_probabilities"),
                pixel_valid_mask,
                normalize_by_mask_area=normalize_pixel_by_mask,
            )
            if pixel_line_loss_enabled and pixel_enabled_in_outputs and outputs.get("pixel_class_probabilities") is not None
            else labels.new_tensor(0.0, dtype=torch.float32)
        )

        instance_total = (
            float(self.loss_cfg.get("corn_weight", 0.0)) * loss_terms["loss_corn"]
            + float(self.loss_cfg["binary_damage_weight"]) * loss_terms["loss_binary_damage"]
            + float(self.loss_cfg["severity_corn_weight"]) * loss_terms["loss_severity_corn"]
            + float(self.loss_cfg["final_ce_weight"]) * loss_terms["loss_final_ce"]
            + float(self.loss_cfg["global_corn_aux_weight"]) * loss_terms["loss_global_corn_aux"]
            + evidence_pixel_weight_eff * loss_terms["loss_evidence_pixel"]
            + evidence_mil_weight_eff * loss_terms["loss_evidence_mil"]
            + severity_map_weight_eff * loss_terms["loss_severity_map"]
            + float(self.loss_cfg["damage_aux_weight"]) * loss_terms["loss_damage_aux"]
            + float(self.loss_cfg["unchanged_weight"]) * loss_terms["loss_unchanged"]
            + float(self.loss_cfg["gate_bg_weight"]) * loss_terms["loss_gate_bg"]
            + float(self.loss_cfg["gate_contrast_weight"]) * loss_terms["loss_gate_contrast"]
            + float(self.loss_cfg.get("minor_no_aux_weight", 0.0)) * loss_terms["loss_minor_no_aux"]
            + float(self.loss_cfg.get("minor_major_aux_weight", 0.0)) * loss_terms["loss_minor_major_aux"]
            + float(self.loss_cfg.get("structural_binary_weight", 0.0)) * loss_terms["loss_structural_binary"]
            + float(self.loss_cfg.get("low_stage_weight", 0.0)) * loss_terms["loss_low_stage"]
            + float(self.loss_cfg.get("high_stage_weight", 0.0)) * loss_terms["loss_high_stage"]
            + float(self.loss_cfg.get("scale_router_ce_weight", 0.0)) * loss_terms["loss_scale_router_ce"]
            + float(self.loss_cfg.get("conditional_review_ce_weight", 0.0)) * loss_terms["loss_conditional_review_ce"]
            + float(self.loss_cfg.get("conditional_review_low_ce_weight", 0.0)) * loss_terms["loss_conditional_review_low_ce"]
            + float(self.loss_cfg.get("conditional_review_high_ce_weight", 0.0)) * loss_terms["loss_conditional_review_high_ce"]
            + float(self.loss_cfg.get("conditional_review_delta_l2_weight", 0.0)) * loss_terms["loss_conditional_review_delta_l2"]
            + float(self.loss_cfg.get("scale_aux_tight_weight", 0.0)) * loss_terms["loss_scale_aux_tight"]
            + float(self.loss_cfg.get("scale_aux_context_weight", 0.0)) * loss_terms["loss_scale_aux_context"]
            + float(self.loss_cfg.get("scale_aux_neighborhood_weight", 0.0)) * loss_terms["loss_scale_aux_neighborhood"]
            + float(self.loss_cfg.get("scale_aux_fusion_ce_weight", 0.0)) * loss_terms["loss_scale_aux_fusion_ce"]
            # scale_aux_weight_entropy_weight > 0 encourages less peaky scale weights.
            - float(self.loss_cfg.get("scale_aux_weight_entropy_weight", 0.0)) * loss_terms["loss_scale_aux_weight_entropy"]
            + float(self.loss_cfg.get("scale_aux_weight_prior_kl_weight", 0.0)) * loss_terms["loss_scale_aux_weight_prior_kl"]
        )
        pixel_total = (
            float(pixel_loss_cfg.get("agg_weight", 0.5)) * loss_terms["loss_pixel_agg"]
            + float(pixel_loss_cfg.get("dense_weight", 0.05)) * loss_terms["loss_pixel_dense"]
            - float(pixel_loss_cfg.get("entropy_weight", 0.0)) * loss_terms["loss_pixel_entropy"]
        )
        if train_target == "instance":
            total = instance_total
        elif train_target == "pixel":
            total = pixel_total
        elif train_target == "joint":
            total = instance_total + pixel_total
        else:
            raise ValueError(f"Unsupported experiment.train_target='{train_target}'.")
        loss_terms["loss_total"] = total
        return loss_terms


def build_loss(config: dict[str, Any]) -> HierarchicalLoss:
    return HierarchicalLoss(config)
