from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.classifier import decode_corn_logits


def summarize_tensor_for_finite_check(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach().float()
    finite_mask = torch.isfinite(detached)
    summary: dict[str, Any] = {
        "shape": list(detached.shape),
        "numel": int(detached.numel()),
        "num_finite": int(finite_mask.sum().item()),
        "num_nonfinite": int((~finite_mask).sum().item()),
    }
    if finite_mask.any():
        finite_values = detached[finite_mask]
        summary.update(
            {
                "mean": float(finite_values.mean().item()),
                "std": float(finite_values.std(unbiased=False).item()) if finite_values.numel() > 1 else 0.0,
                "min": float(finite_values.min().item()),
                "max": float(finite_values.max().item()),
            }
        )
    else:
        summary.update({"mean": None, "std": None, "min": None, "max": None})
    return summary


def assert_finite_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    context: dict[str, Any] | None = None,
) -> None:
    summary = summarize_tensor_for_finite_check(tensor)
    if summary["num_nonfinite"] == 0:
        return
    message = [f"Non-finite tensor detected: {name}", f"summary: {summary}"]
    if context:
        message.append(f"context: {context}")
    raise RuntimeError(" | ".join(message))


def compute_expected_severity_from_probabilities(
    probabilities: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    class_probabilities = probabilities.float()
    if positions is None:
        positions = torch.linspace(
            0.0,
            1.0,
            steps=class_probabilities.size(1),
            device=class_probabilities.device,
            dtype=class_probabilities.dtype,
        )
    else:
        positions = positions.to(device=class_probabilities.device, dtype=class_probabilities.dtype)
    return (class_probabilities * positions.unsqueeze(0)).sum(dim=1)


def normalize_probability_distribution(probabilities: torch.Tensor, eps: float) -> torch.Tensor:
    safe_probabilities = torch.nan_to_num(probabilities.float(), nan=eps, posinf=1.0 - eps, neginf=eps)
    safe_probabilities = safe_probabilities.clamp(min=eps, max=1.0 - eps)
    safe_probabilities = safe_probabilities / safe_probabilities.sum(dim=1, keepdim=True).clamp_min(eps)
    return safe_probabilities


def resize_mask_to_feature_map(
    mask: torch.Tensor | None,
    spatial_size: tuple[int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4:
        raise ValueError(f"Expected mask with 3 or 4 dims, got shape={tuple(mask.shape)}.")
    resized_mask = F.interpolate(mask.float(), size=spatial_size, mode="nearest")
    return resized_mask.to(device=device, dtype=dtype)


def masked_average_pool_2d(feature_map: torch.Tensor, mask: torch.Tensor | None, eps: float = 1e-6) -> torch.Tensor:
    if feature_map.ndim != 4:
        raise ValueError(f"masked_average_pool_2d expects [B, C, H, W], got shape={tuple(feature_map.shape)}.")

    batch_size, channels, _, _ = feature_map.shape
    pooled_full = feature_map.float().flatten(2).mean(dim=-1)
    resized_mask = resize_mask_to_feature_map(
        mask,
        feature_map.shape[-2:],
        device=feature_map.device,
        dtype=feature_map.dtype,
    )
    if resized_mask is None:
        return pooled_full

    flat_mask = resized_mask.flatten(2)
    flat_feature = feature_map.float().flatten(2)
    denominator = flat_mask.sum(dim=-1)
    pooled_masked = (flat_feature * flat_mask).sum(dim=-1) / denominator.clamp_min(eps)
    valid_rows = (denominator >= eps).expand(batch_size, channels)
    return torch.where(valid_rows, pooled_masked, pooled_full)


def compute_single_channel_region_statistics(
    single_channel_map: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    if single_channel_map.ndim != 4 or single_channel_map.size(1) != 1:
        raise ValueError(
            "compute_single_channel_region_statistics expects [B, 1, H, W], "
            f"got shape={tuple(single_channel_map.shape)}."
        )

    values = single_channel_map.float().flatten(1)
    map_mean = values.mean(dim=1)
    map_std = values.std(dim=1, unbiased=False)
    if mask is None:
        valid = torch.ones_like(map_mean, dtype=torch.bool)
        invalid = torch.zeros_like(map_mean, dtype=torch.bool)
        return {
            "inside_mean": map_mean,
            "outside_mean": map_mean,
            "gap": torch.zeros_like(map_mean),
            "map_mean": map_mean,
            "map_std": map_std,
            "inside_valid": valid,
            "outside_valid": invalid,
        }

    resized_mask = resize_mask_to_feature_map(
        mask,
        single_channel_map.shape[-2:],
        device=single_channel_map.device,
        dtype=single_channel_map.dtype,
    )
    if resized_mask is None:
        valid = torch.ones_like(map_mean, dtype=torch.bool)
        invalid = torch.zeros_like(map_mean, dtype=torch.bool)
        return {
            "inside_mean": map_mean,
            "outside_mean": map_mean,
            "gap": torch.zeros_like(map_mean),
            "map_mean": map_mean,
            "map_std": map_std,
            "inside_valid": valid,
            "outside_valid": invalid,
        }

    flat_mask = resized_mask.float().flatten(1)
    inside_denominator = flat_mask.sum(dim=1)
    outside_weight = 1.0 - flat_mask
    outside_denominator = outside_weight.sum(dim=1)

    inside_mean = (values * flat_mask).sum(dim=1) / inside_denominator.clamp_min(eps)
    outside_mean = (values * outside_weight).sum(dim=1) / outside_denominator.clamp_min(eps)

    inside_valid = inside_denominator >= eps
    outside_valid = outside_denominator >= eps
    inside_mean = torch.where(inside_valid, inside_mean, map_mean)
    outside_mean = torch.where(outside_valid, outside_mean, map_mean)

    return {
        "inside_mean": inside_mean,
        "outside_mean": outside_mean,
        "gap": inside_mean - outside_mean,
        "map_mean": map_mean,
        "map_std": map_std,
        "inside_valid": inside_valid,
        "outside_valid": outside_valid,
    }


def compute_gate_regularization_losses(
    change_gate: torch.Tensor | None,
    instance_mask: torch.Tensor | None,
    *,
    contrast_margin: float,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
    if change_gate is None:
        return torch.tensor(0.0), torch.tensor(0.0), None

    stats = compute_single_channel_region_statistics(change_gate, instance_mask, eps=eps)
    zero = change_gate.new_tensor(0.0)

    outside_valid = stats["outside_valid"]
    if outside_valid.any():
        loss_gate_bg = stats["outside_mean"][outside_valid].mean()
    else:
        loss_gate_bg = zero

    both_valid = stats["inside_valid"] & stats["outside_valid"]
    if both_valid.any():
        contrast_terms = F.relu(float(contrast_margin) - stats["inside_mean"] + stats["outside_mean"])
        loss_gate_contrast = contrast_terms[both_valid].mean()
    else:
        loss_gate_contrast = zero

    return loss_gate_bg, loss_gate_contrast, stats


def resolve_damage_pixel_target(
    *,
    post_target: torch.Tensor | None,
    damage_mask_gt: torch.Tensor | None,
    no_damage_label_id: int,
) -> torch.Tensor | None:
    pixel_target = damage_mask_gt if damage_mask_gt is not None else post_target
    if pixel_target is None:
        return None
    if pixel_target.ndim == 3:
        pixel_target = pixel_target.unsqueeze(1)
    if pixel_target.ndim != 4:
        return None

    pixel_target = pixel_target.float()
    if damage_mask_gt is not None:
        return (pixel_target > 0.5).to(dtype=torch.float32)

    max_value = float(pixel_target.max().item()) if pixel_target.numel() > 0 else 0.0
    damage_threshold = 1.0 if max_value > 3.0 else float(no_damage_label_id)
    return (pixel_target > damage_threshold).to(dtype=torch.float32)


def masked_binary_map_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    eps: float = 1e-6,
) -> torch.Tensor:
    logits = logits.float()
    targets = resize_mask_to_feature_map(targets, logits.shape[-2:], device=logits.device, dtype=logits.dtype)
    if targets is None:
        return logits.new_tensor(0.0)
    per_pixel = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    resized_mask = resize_mask_to_feature_map(mask, logits.shape[-2:], device=logits.device, dtype=logits.dtype)
    if resized_mask is None:
        return per_pixel.mean()

    valid_weight = resized_mask.expand_as(per_pixel)
    denominator = valid_weight.flatten(1).sum(dim=1)
    masked_loss = (per_pixel * valid_weight).flatten(1).sum(dim=1) / denominator.clamp_min(eps)
    fallback_loss = per_pixel.flatten(1).mean(dim=1)
    return torch.where(denominator >= eps, masked_loss, fallback_loss).mean()


class CORNClassificationLoss(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int = 4,
        probability_eps: float = 1e-6,
        loss_damage_tight_weight: float = 0.0,
        loss_damage_context_weight: float = 0.0,
        loss_damage_neighborhood_weight: float = 0.0,
        loss_severity_aux_weight: float = 0.0,
        loss_unchanged_weight: float = 0.0,
        tight_gate_bg_weight: float = 0.0,
        tight_gate_contrast_weight: float = 0.0,
        context_gate_bg_weight: float = 0.0,
        context_gate_contrast_weight: float = 0.0,
        neighborhood_gate_bg_weight: float = 0.0,
        neighborhood_gate_contrast_weight: float = 0.0,
        loss_corn_mono_weight: float = 0.0,
        tight_gate_contrast_margin: float = 0.15,
        context_gate_contrast_margin: float = 0.08,
        neighborhood_gate_contrast_margin: float = 0.03,
        damage_aux_mode: str = "instance_weak",
        no_damage_label_id: int = 0,
        graph_consistency_weight: float = 0.0,
        enable_ordinal_adjacent_smoothing: bool = False,
        ordinal_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.probability_eps = float(probability_eps)
        self.damage_aux_weights = {
            "tight": float(loss_damage_tight_weight),
            "context": float(loss_damage_context_weight),
            "neighborhood": float(loss_damage_neighborhood_weight),
        }
        self.loss_severity_aux_weight = float(loss_severity_aux_weight)
        self.loss_unchanged_weight = float(loss_unchanged_weight)
        self.gate_bg_weights = {
            "tight": float(tight_gate_bg_weight),
            "context": float(context_gate_bg_weight),
            "neighborhood": float(neighborhood_gate_bg_weight),
        }
        self.gate_contrast_weights = {
            "tight": float(tight_gate_contrast_weight),
            "context": float(context_gate_contrast_weight),
            "neighborhood": float(neighborhood_gate_contrast_weight),
        }
        self.gate_margins = {
            "tight": float(tight_gate_contrast_margin),
            "context": float(context_gate_contrast_margin),
            "neighborhood": float(neighborhood_gate_contrast_margin),
        }
        self.loss_corn_mono_weight = float(loss_corn_mono_weight)
        self.damage_aux_mode = str(damage_aux_mode).lower()
        self.no_damage_label_id = int(no_damage_label_id)
        self.graph_consistency_weight = float(graph_consistency_weight)
        self.enable_ordinal_adjacent_smoothing = bool(enable_ordinal_adjacent_smoothing)
        self.ordinal_smoothing = float(ordinal_smoothing)

    def _build_adjacent_smoothed_targets(self, targets: torch.Tensor) -> torch.Tensor:
        num_classes = self.num_classes
        smoothing = max(min(self.ordinal_smoothing, 0.49), 0.0)
        smoothed = torch.zeros(targets.size(0), num_classes, device=targets.device, dtype=torch.float32)
        smoothed.scatter_(1, targets.unsqueeze(1), 1.0)
        if smoothing <= 0.0:
            return smoothed
        for class_index in range(num_classes):
            class_mask = targets == class_index
            if not class_mask.any():
                continue
            neighbors = []
            if class_index > 0:
                neighbors.append(class_index - 1)
            if class_index < num_classes - 1:
                neighbors.append(class_index + 1)
            if not neighbors:
                continue
            share = smoothing / float(len(neighbors))
            smoothed[class_mask, class_index] = 1.0 - smoothing
            for neighbor_index in neighbors:
                smoothed[class_mask, neighbor_index] = share
        return smoothed

    def _compute_corn_loss(
        self,
        corn_logits: torch.Tensor,
        targets: torch.Tensor,
        threshold_probabilities: torch.Tensor | None,
        class_probabilities: torch.Tensor | None,
        debug_context: dict[str, Any] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if corn_logits.size(1) != self.num_classes - 1:
            raise ValueError(
                f"CORNClassificationLoss expects {self.num_classes - 1} logits, got {tuple(corn_logits.shape)}."
            )

        logits = corn_logits.float()
        total_loss = logits.new_tensor(0.0)
        num_examples = 0
        task_losses = logits.new_zeros(self.num_classes - 1)
        task_counts = logits.new_zeros(self.num_classes - 1)

        for threshold_idx in range(self.num_classes - 1):
            label_mask = torch.ones_like(targets, dtype=torch.bool) if threshold_idx == 0 else targets > (threshold_idx - 1)
            if not label_mask.any():
                continue
            binary_targets = (targets[label_mask] > threshold_idx).to(dtype=logits.dtype)
            threshold_logits = logits[label_mask, threshold_idx]
            task_loss = F.binary_cross_entropy_with_logits(threshold_logits, binary_targets, reduction="sum")
            total_loss = total_loss + task_loss
            task_count = int(binary_targets.numel())
            num_examples += task_count
            task_losses[threshold_idx] = task_loss / max(task_count, 1)
            task_counts[threshold_idx] = float(task_count)

        resolved_threshold_probabilities = torch.sigmoid(logits) if threshold_probabilities is None else threshold_probabilities.float()
        resolved_class_probabilities = decode_corn_logits(logits) if class_probabilities is None else class_probabilities.float()
        resolved_class_probabilities = normalize_probability_distribution(resolved_class_probabilities, self.probability_eps)
        assert_finite_tensor("corn_logits", logits, context=debug_context)
        assert_finite_tensor("threshold_probabilities", resolved_threshold_probabilities, context=debug_context)
        assert_finite_tensor("class_probabilities", resolved_class_probabilities, context=debug_context)
        return (
            total_loss / max(num_examples, 1),
            resolved_threshold_probabilities,
            resolved_class_probabilities,
            task_losses,
            task_counts,
        )

    def _compute_scale_damage_loss(
        self,
        damage_map_logits: torch.Tensor | None,
        mask: torch.Tensor | None,
        damage_binary: torch.Tensor,
        pixel_damage_target: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if damage_map_logits is None:
            return damage_binary.new_tensor(0.0), None
        damage_score_logit = masked_average_pool_2d(damage_map_logits, mask).squeeze(1)
        if self.damage_aux_mode == "pixel_if_available" and pixel_damage_target is not None:
            return masked_binary_map_loss(damage_map_logits, pixel_damage_target, mask=mask), damage_score_logit
        return F.binary_cross_entropy_with_logits(damage_score_logit.float(), damage_binary.float()), damage_score_logit

    def _compute_unchanged_loss(
        self,
        scale_outputs: dict[str, Any],
        batch: dict[str, Any],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        no_damage_rows = targets == self.no_damage_label_id
        if not no_damage_rows.any():
            return targets.new_tensor(0.0, dtype=torch.float32)
        losses = []
        for scale_name in ("tight", "context", "neighborhood"):
            feat_pre = scale_outputs[scale_name].get("feat_pre_refined")
            feat_post = scale_outputs[scale_name].get("feat_post_refined")
            if feat_pre is None or feat_post is None:
                continue
            mask = batch.get(f"mask_{scale_name}")
            pre_pooled = masked_average_pool_2d(feat_pre, mask)[no_damage_rows]
            post_pooled = masked_average_pool_2d(feat_post, mask)[no_damage_rows]
            cosine_similarity = F.cosine_similarity(pre_pooled.float(), post_pooled.float(), dim=1, eps=1e-6)
            losses.append((1.0 - cosine_similarity).mean())
        if not losses:
            return targets.new_tensor(0.0, dtype=torch.float32)
        return torch.stack(losses).mean()

    def forward(
        self,
        *,
        outputs: dict[str, Any],
        targets: torch.Tensor,
        batch: dict[str, Any],
        debug_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        loss_corn, resolved_threshold_probabilities, resolved_class_probabilities, task_losses, task_counts = self._compute_corn_loss(
            outputs["corn_logits"],
            targets,
            outputs.get("threshold_probabilities"),
            outputs.get("class_probabilities"),
            debug_context,
        )
        zero = loss_corn.new_tensor(0.0)
        damage_binary = (targets != self.no_damage_label_id).to(dtype=loss_corn.dtype)
        pixel_damage_target = resolve_damage_pixel_target(
            post_target=batch.get("post_target"),
            damage_mask_gt=batch.get("damage_mask_gt"),
            no_damage_label_id=self.no_damage_label_id,
        )

        scale_outputs = outputs["scale_outputs"]
        damage_losses: dict[str, torch.Tensor] = {}
        damage_scores: dict[str, torch.Tensor | None] = {}
        gate_bg_losses: dict[str, torch.Tensor] = {}
        gate_contrast_losses: dict[str, torch.Tensor] = {}
        loss_total = loss_corn

        loss_ordinal_adjacent_smoothing = zero
        if self.enable_ordinal_adjacent_smoothing and self.ordinal_smoothing > 0.0:
            smoothed_targets = self._build_adjacent_smoothed_targets(targets)
            log_probs = resolved_class_probabilities.float().clamp_min(self.probability_eps).log()
            loss_ordinal_adjacent_smoothing = F.kl_div(log_probs, smoothed_targets, reduction="batchmean")
            loss_total = loss_total + (self.ordinal_smoothing * loss_ordinal_adjacent_smoothing)

        for scale_name in ("tight", "context", "neighborhood"):
            damage_loss, damage_score = self._compute_scale_damage_loss(
                scale_outputs[scale_name].get("damage_map_logits"),
                batch.get(f"mask_{scale_name}"),
                damage_binary,
                pixel_damage_target,
            )
            damage_losses[scale_name] = damage_loss
            damage_scores[scale_name] = damage_score
            loss_total = loss_total + (self.damage_aux_weights[scale_name] * damage_loss)

            change_gate = scale_outputs[scale_name].get("change_gate")
            if change_gate is not None and (
                self.gate_bg_weights[scale_name] > 0.0 or self.gate_contrast_weights[scale_name] > 0.0
            ):
                gate_bg, gate_contrast, _ = compute_gate_regularization_losses(
                    change_gate,
                    batch.get(f"mask_{scale_name}"),
                    contrast_margin=self.gate_margins[scale_name],
                )
            else:
                gate_bg, gate_contrast = zero, zero
            gate_bg_losses[scale_name] = gate_bg
            gate_contrast_losses[scale_name] = gate_contrast
            loss_total = loss_total + (self.gate_bg_weights[scale_name] * gate_bg)
            loss_total = loss_total + (self.gate_contrast_weights[scale_name] * gate_contrast)

        severity_score = outputs.get("severity_score")
        severity_target = targets.float() / max(self.num_classes - 1, 1)
        loss_severity_aux = zero
        if severity_score is not None and self.loss_severity_aux_weight > 0.0:
            loss_severity_aux = F.smooth_l1_loss(severity_score.float(), severity_target)
            loss_total = loss_total + (self.loss_severity_aux_weight * loss_severity_aux)

        loss_unchanged = zero
        if self.loss_unchanged_weight > 0.0:
            loss_unchanged = self._compute_unchanged_loss(scale_outputs, batch, targets)
            loss_total = loss_total + (self.loss_unchanged_weight * loss_unchanged)

        loss_corn_mono = zero
        if self.loss_corn_mono_weight > 0.0 and resolved_threshold_probabilities.size(1) >= 2:
            monotonic_terms = []
            threshold_probs = resolved_threshold_probabilities.float()
            for threshold_idx in range(1, threshold_probs.size(1)):
                monotonic_terms.append(F.relu(threshold_probs[:, threshold_idx] - threshold_probs[:, threshold_idx - 1]))
            if monotonic_terms:
                loss_corn_mono = torch.stack(monotonic_terms, dim=0).sum(dim=0).mean()
                loss_total = loss_total + (self.loss_corn_mono_weight * loss_corn_mono)

        loss_graph_consistency = zero
        graph_feature = outputs.get("neighborhood_graph_feature")
        if graph_feature is not None and self.graph_consistency_weight > 0.0:
            reference = outputs["instance_feature"]
            cosine = F.cosine_similarity(reference.float(), graph_feature.float(), dim=1, eps=1e-6)
            valid_neighbors = outputs["diagnostics"].get("graph_valid_neighbor_count")
            if valid_neighbors is not None:
                valid_rows = valid_neighbors > 0
                if valid_rows.any():
                    loss_graph_consistency = (1.0 - cosine[valid_rows]).mean()
            else:
                loss_graph_consistency = (1.0 - cosine).mean()
            loss_total = loss_total + (self.graph_consistency_weight * loss_graph_consistency)

        loss_terms: dict[str, torch.Tensor] = {
            "loss_corn": loss_corn,
            "loss_damage_tight": damage_losses["tight"],
            "loss_damage_context": damage_losses["context"],
            "loss_damage_neighborhood": damage_losses["neighborhood"],
            "loss_damage_aux": damage_losses["tight"] + damage_losses["context"] + damage_losses["neighborhood"],
            "loss_severity_aux": loss_severity_aux,
            "loss_unchanged": loss_unchanged,
            "loss_gate_bg_tight": gate_bg_losses["tight"],
            "loss_gate_bg_context": gate_bg_losses["context"],
            "loss_gate_bg_neighborhood": gate_bg_losses["neighborhood"],
            "loss_gate_contrast_tight": gate_contrast_losses["tight"],
            "loss_gate_contrast_context": gate_contrast_losses["context"],
            "loss_gate_contrast_neighborhood": gate_contrast_losses["neighborhood"],
            "loss_graph_consistency": loss_graph_consistency,
            "loss_corn_mono": loss_corn_mono,
            "loss_ordinal_adjacent_smoothing": loss_ordinal_adjacent_smoothing,
        }
        loss_terms["loss_gate_bg"] = (
            gate_bg_losses["tight"] + gate_bg_losses["context"] + gate_bg_losses["neighborhood"]
        )
        loss_terms["loss_gate_contrast"] = (
            gate_contrast_losses["tight"] + gate_contrast_losses["context"] + gate_contrast_losses["neighborhood"]
        )

        return {
            "loss": loss_total,
            "loss_main": loss_corn,
            "loss_terms": loss_terms,
            "class_probabilities": resolved_class_probabilities,
            "threshold_probabilities": resolved_threshold_probabilities,
            "corn_task_losses": task_losses,
            "corn_task_counts": task_counts,
            "aux_metrics": {
                "damage_aux_scores": damage_scores,
                "damage_aux_targets": damage_binary.detach().float(),
                "severity_scores": None if severity_score is None else severity_score.detach().float(),
                "severity_targets": severity_target.detach().float(),
                "severity_labels": targets.detach().long(),
            },
        }


def build_loss(config: dict[str, Any]) -> CORNClassificationLoss:
    loss_cfg = config["loss"]
    return CORNClassificationLoss(
        num_classes=4,
        probability_eps=float(loss_cfg["probability_eps"]),
        loss_damage_tight_weight=float(loss_cfg["loss_damage_tight_weight"]),
        loss_damage_context_weight=float(loss_cfg["loss_damage_context_weight"]),
        loss_damage_neighborhood_weight=float(loss_cfg["loss_damage_neighborhood_weight"]),
        loss_severity_aux_weight=float(loss_cfg["loss_severity_aux_weight"]),
        loss_unchanged_weight=float(loss_cfg["loss_unchanged_weight"]),
        tight_gate_bg_weight=float(loss_cfg["tight_gate_bg_weight"]),
        tight_gate_contrast_weight=float(loss_cfg["tight_gate_contrast_weight"]),
        context_gate_bg_weight=float(loss_cfg["context_gate_bg_weight"]),
        context_gate_contrast_weight=float(loss_cfg["context_gate_contrast_weight"]),
        neighborhood_gate_bg_weight=float(loss_cfg["neighborhood_gate_bg_weight"]),
        neighborhood_gate_contrast_weight=float(loss_cfg["neighborhood_gate_contrast_weight"]),
        loss_corn_mono_weight=float(loss_cfg.get("loss_corn_mono_weight", 0.0)),
        tight_gate_contrast_margin=float(loss_cfg["tight_gate_contrast_margin"]),
        context_gate_contrast_margin=float(loss_cfg["context_gate_contrast_margin"]),
        neighborhood_gate_contrast_margin=float(loss_cfg["neighborhood_gate_contrast_margin"]),
        damage_aux_mode=str(loss_cfg["damage_aux_mode"]),
        no_damage_label_id=int(loss_cfg["no_damage_label_id"]),
        graph_consistency_weight=float(loss_cfg.get("graph_consistency_weight", 0.0)),
        enable_ordinal_adjacent_smoothing=bool(loss_cfg.get("enable_ordinal_adjacent_smoothing", False)),
        ordinal_smoothing=float(loss_cfg.get("ordinal_smoothing", 0.05)),
    )
