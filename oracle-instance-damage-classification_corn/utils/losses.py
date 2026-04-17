from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def _prepare_soft_target_distribution(
    target_distribution: torch.Tensor | list[list[float]] | tuple[tuple[float, ...], ...],
    *,
    num_classes: int,
) -> torch.Tensor:
    distribution = torch.as_tensor(target_distribution, dtype=torch.float32)
    if distribution.ndim != 2 or distribution.size(0) != int(num_classes) or distribution.size(1) != int(num_classes):
        raise ValueError(
            "Soft target distribution must be a square matrix with shape "
            f"({int(num_classes)}, {int(num_classes)}), got {tuple(distribution.shape)}."
        )
    row_sums = distribution.sum(dim=1, keepdim=True)
    if torch.any(row_sums <= 0.0):
        raise ValueError("Each row in the soft target distribution must sum to a positive value.")
    return distribution / row_sums


def build_batch_soft_targets(
    targets: torch.Tensor,
    target_distribution: torch.Tensor,
) -> torch.Tensor:
    safe_targets = targets.reshape(-1).to(device=target_distribution.device, dtype=torch.long)
    return target_distribution.index_select(0, safe_targets)


def _soft_cross_entropy_from_probabilities(
    probabilities: torch.Tensor,
    soft_targets: torch.Tensor,
) -> torch.Tensor:
    safe_probabilities = probabilities.float().clamp_min(1e-8)
    safe_soft_targets = soft_targets.float()
    loss = -(safe_soft_targets * safe_probabilities.log()).sum(dim=1).mean()
    return loss.to(device=probabilities.device, dtype=safe_probabilities.dtype)


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-6)
    return math.log(math.expm1(value))


def _inverse_sigmoid(value: float) -> float:
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


def _compute_tau_fraction_scalar(
    tau_value: float,
    tau_min: float,
    tau_max: float,
) -> float:
    tau_range = max(float(tau_max) - float(tau_min), 1e-6)
    fraction = (float(tau_value) - float(tau_min)) / tau_range
    return min(max(fraction, 1e-6), 1.0 - 1e-6)


def _compute_raw_tau_center(
    tau_target: float,
    tau_min: float,
    tau_max: float,
) -> float:
    return _inverse_sigmoid(_compute_tau_fraction_scalar(tau_target, tau_min, tau_max))


def _tau_to_fraction_tensor(
    tau: torch.Tensor,
    tau_min: float,
    tau_max: float,
) -> torch.Tensor:
    tau_range = max(float(tau_max) - float(tau_min), 1e-6)
    return ((tau.float() - float(tau_min)) / tau_range).clamp(1e-6, 1.0 - 1e-6)


def _tau_to_raw_tau_logit(
    tau: torch.Tensor,
    tau_min: float,
    tau_max: float,
) -> torch.Tensor:
    tau_fraction = _tau_to_fraction_tensor(tau, tau_min, tau_max)
    return torch.log(tau_fraction / (1.0 - tau_fraction))


def _compute_scalar_statistics(values: torch.Tensor) -> dict[str, torch.Tensor]:
    flattened = values.detach().float().reshape(-1)
    if flattened.numel() == 0:
        zero = values.new_tensor(0.0)
        return {
            "mean": zero,
            "std": zero,
            "min": zero,
            "max": zero,
            "p10": zero,
            "p50": zero,
            "p90": zero,
        }

    quantiles = torch.quantile(
        flattened,
        torch.tensor([0.10, 0.50, 0.90], device=flattened.device, dtype=flattened.dtype),
    )
    std = flattened.std(unbiased=False) if flattened.numel() > 1 else flattened.new_tensor(0.0)
    return {
        "mean": flattened.mean(),
        "std": std,
        "min": flattened.min(),
        "max": flattened.max(),
        "p10": quantiles[0],
        "p50": quantiles[1],
        "p90": quantiles[2],
    }


def _compute_pearson_correlation(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first_flat = first.detach().float().reshape(-1)
    second_flat = second.detach().float().reshape(-1)
    if first_flat.numel() == 0 or second_flat.numel() == 0 or first_flat.numel() != second_flat.numel():
        return first.new_tensor(0.0)
    if first_flat.numel() < 2:
        return first.new_tensor(0.0)

    first_centered = first_flat - first_flat.mean()
    second_centered = second_flat - second_flat.mean()
    denominator = torch.sqrt(first_centered.pow(2).sum() * second_centered.pow(2).sum()).clamp_min(1e-12)
    correlation = (first_centered * second_centered).sum() / denominator
    return correlation.to(device=first.device, dtype=first.dtype)


def compute_sample_difficulty_from_probabilities(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = probabilities.detach().float()
    gt_probabilities = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
    difficulty = 1.0 - gt_probabilities
    return difficulty.to(device=probabilities.device, dtype=probabilities.dtype), gt_probabilities.to(
        device=probabilities.device,
        dtype=probabilities.dtype,
    )


def compute_sample_difficulty(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.softmax(logits.detach().float(), dim=1)
    difficulty, gt_probabilities = compute_sample_difficulty_from_probabilities(probabilities, targets)
    return difficulty.to(device=logits.device, dtype=logits.dtype), gt_probabilities.to(device=logits.device, dtype=logits.dtype)


def build_tau_reference_from_difficulty(
    tau: torch.Tensor,
    difficulty: torch.Tensor,
    tau_easy: float,
    tau_hard: float,
) -> torch.Tensor:
    return tau.new_tensor(float(tau_easy)) + (tau.new_tensor(float(tau_hard - tau_easy)) * difficulty)


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


def compute_symmetric_kl_from_probabilities(
    first_probabilities: torch.Tensor,
    second_probabilities: torch.Tensor,
) -> torch.Tensor:
    first = first_probabilities.float().clamp_min(1e-8)
    second = second_probabilities.float().clamp_min(1e-8)
    kl_first_second = (first * (first.log() - second.log())).sum(dim=1)
    kl_second_first = (second * (second.log() - first.log())).sum(dim=1)
    return 0.5 * (kl_first_second + kl_second_first).mean()


def compute_js_divergence_from_probabilities(
    first_probabilities: torch.Tensor,
    second_probabilities: torch.Tensor,
) -> torch.Tensor:
    first = first_probabilities.float().clamp_min(1e-8)
    second = second_probabilities.float().clamp_min(1e-8)
    mixture = 0.5 * (first + second)
    return 0.5 * (
        (first * (first.log() - mixture.log())).sum(dim=1).mean()
        + (second * (second.log() - mixture.log())).sum(dim=1).mean()
    )


def compute_ordinal_cdf_distance(
    first_probabilities: torch.Tensor,
    second_probabilities: torch.Tensor,
) -> torch.Tensor:
    first = first_probabilities.float()
    second = second_probabilities.float()
    first_cdf = torch.cumsum(first, dim=1)
    second_cdf = torch.cumsum(second, dim=1)
    if first_cdf.size(1) <= 1:
        return first.new_tensor(0.0)
    return torch.abs(first_cdf[:, :-1] - second_cdf[:, :-1]).mean()


class WeightedCrossEntropyLossWrapper(nn.Module):
    def __init__(self, class_weights: torch.Tensor | None = None, smoothing: float = 0.0) -> None:
        super().__init__()
        self.smoothing = float(smoothing)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        if self.smoothing > 0:
            num_classes = logits.size(1)
            with torch.no_grad():
                true_dist = torch.full_like(log_probs, self.smoothing / max(num_classes - 1, 1))
                true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
            loss = -(true_dist * log_probs).sum(dim=1)
        else:
            loss = F.nll_loss(log_probs, target, reduction="none")

        if self.class_weights is not None:
            loss = loss * self.class_weights[target]
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        gamma: float = 2.0,
        smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.smoothing = float(smoothing)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        focal_weight = (1.0 - pt).pow(self.gamma)

        if self.smoothing > 0:
            num_classes = logits.size(1)
            with torch.no_grad():
                true_dist = torch.full_like(log_probs, self.smoothing / max(num_classes - 1, 1))
                true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
            ce = -(true_dist * log_probs).sum(dim=1)
        else:
            ce = -log_pt

        loss = focal_weight * ce
        if self.class_weights is not None:
            loss = loss * self.class_weights[target]
        return loss.mean()


class LearnableSeverityAxis(nn.Module):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        if int(num_classes) != 4:
            raise ValueError("The current oracle damage setup expects exactly 4 ordered damage classes.")

        self.num_classes = int(num_classes)
        gap_init = _inverse_softplus(1.0)
        self.raw_gap_01 = nn.Parameter(torch.tensor(gap_init, dtype=torch.float32))
        self.raw_gap_12 = nn.Parameter(torch.tensor(gap_init, dtype=torch.float32))
        self.raw_gap_23 = nn.Parameter(torch.tensor(gap_init, dtype=torch.float32))

    def _compute_gaps(self) -> torch.Tensor:
        raw_gaps = torch.stack([self.raw_gap_01, self.raw_gap_12, self.raw_gap_23], dim=0)
        return F.softplus(raw_gaps) + 1e-4

    def _compute_positions(self) -> tuple[torch.Tensor, torch.Tensor]:
        gaps = self._compute_gaps()
        positions = torch.zeros(self.num_classes, device=gaps.device, dtype=gaps.dtype)
        positions[1] = gaps[0]
        positions[2] = gaps[0] + gaps[1]
        positions[3] = gaps.sum()
        positions = positions / (positions[-1] + 1e-6)
        return gaps, positions

    def build_soft_targets(
        self,
        targets: torch.Tensor,
        tau: torch.Tensor,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gaps, positions = self._compute_positions()
        if device is not None or dtype is not None:
            positions = positions.to(
                device=device if device is not None else positions.device,
                dtype=dtype if dtype is not None else positions.dtype,
            )
            gaps = gaps.to(
                device=device if device is not None else gaps.device,
                dtype=dtype if dtype is not None else gaps.dtype,
            )

        tau = tau.to(device=positions.device, dtype=positions.dtype).clamp_min(1e-6)
        gt_positions = positions[targets].unsqueeze(1)
        distances = torch.abs(gt_positions - positions.unsqueeze(0))
        if tau.ndim == 0:
            soft_targets = torch.softmax(-distances / tau, dim=1)
        else:
            soft_targets = torch.softmax(-distances / tau.unsqueeze(1), dim=1)
        return gaps, soft_targets

    def get_current_gaps(self) -> torch.Tensor:
        return self._compute_gaps().detach().clone()

    def get_current_positions(self) -> torch.Tensor:
        _, positions = self._compute_positions()
        return positions.detach().clone()

    def get_gap_parameters(self) -> list[nn.Parameter]:
        return [self.raw_gap_01, self.raw_gap_12, self.raw_gap_23]


class FixedCDALoss(nn.Module):
    def __init__(self, num_classes: int = 4, alpha: float = 0.3) -> None:
        super().__init__()
        if alpha <= 0:
            raise ValueError("fixed_cda alpha must be > 0.")
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.register_buffer("base_positions", torch.linspace(0.0, 1.0, steps=self.num_classes))

    def _build_soft_targets(
        self,
        targets: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        class_indices = torch.arange(self.num_classes, device=device, dtype=targets.dtype)
        distances = (targets.unsqueeze(1) - class_indices.unsqueeze(0)).abs().to(dtype=dtype)
        weights = torch.pow(
            torch.full_like(distances, self.alpha, dtype=dtype, device=device),
            distances,
        )
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        soft_targets = self._build_soft_targets(targets, device=logits.device, dtype=logits.dtype)
        return {
            "loss_ord": _soft_cross_entropy(logits, soft_targets),
            "soft_targets": soft_targets,
            "positions": self.base_positions.to(device=logits.device, dtype=logits.dtype),
        }

    def get_current_positions(self) -> torch.Tensor:
        return self.base_positions.detach().clone()

    def get_current_soft_target_matrix(self) -> torch.Tensor:
        targets = torch.arange(self.num_classes, device=self.base_positions.device, dtype=torch.long)
        return self._build_soft_targets(
            targets,
            device=self.base_positions.device,
            dtype=self.base_positions.dtype,
        ).detach()


class LearnableOrdinalCDALoss(LearnableSeverityAxis):
    def __init__(self, num_classes: int = 4, tau_init: float = 0.35) -> None:
        super().__init__(num_classes=num_classes)
        tau_fraction = (float(tau_init) - 0.05) / 0.95
        self.raw_tau = nn.Parameter(torch.tensor(_inverse_sigmoid(tau_fraction), dtype=torch.float32))

    def _compute_tau(self) -> torch.Tensor:
        return 0.05 + 0.95 * torch.sigmoid(self.raw_tau)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        gaps, soft_targets = self.build_soft_targets(
            targets,
            self._compute_tau(),
            device=logits.device,
            dtype=logits.dtype,
        )
        positions = self.get_current_positions().to(device=logits.device, dtype=logits.dtype)
        tau = self._compute_tau().to(device=logits.device, dtype=logits.dtype)
        return {
            "loss_ord": _soft_cross_entropy(logits, soft_targets),
            "soft_targets": soft_targets,
            "positions": positions,
            "tau": tau,
            "gaps": gaps,
        }

    def get_current_tau(self) -> torch.Tensor:
        return self._compute_tau().detach().clone()

    def get_current_soft_target_matrix(self) -> torch.Tensor:
        positions = self.get_current_positions()
        tau = self.get_current_tau().clamp_min(1e-6)
        distances = torch.abs(positions.unsqueeze(1) - positions.unsqueeze(0))
        return torch.softmax(-distances / tau, dim=1).detach()


class UnimodalityRegularizer(nn.Module):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.num_classes = int(num_classes)

    def forward(self, probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        penalties: list[torch.Tensor] = []

        for boundary in range(1, self.num_classes):
            mask = targets >= boundary
            if mask.any():
                penalties.append(F.relu(probabilities[mask, boundary - 1] - probabilities[mask, boundary]))

        for boundary in range(self.num_classes - 1):
            mask = targets <= boundary
            if mask.any():
                penalties.append(F.relu(probabilities[mask, boundary + 1] - probabilities[mask, boundary]))

        if not penalties:
            return probabilities.new_tensor(0.0)
        return torch.cat([item.reshape(-1) for item in penalties], dim=0).mean()


class ConcentrationRegularizer(nn.Module):
    def __init__(self, num_classes: int = 4, margin: float = 0.05) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.margin = float(margin)

    def forward(self, probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        gt_probs = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
        penalties: list[torch.Tensor] = []

        left_neighbors = targets - 1
        left_mask = left_neighbors >= 0
        if left_mask.any():
            left_probs = probabilities[left_mask].gather(1, left_neighbors[left_mask].unsqueeze(1)).squeeze(1)
            penalties.append(F.relu(self.margin - (gt_probs[left_mask] - left_probs)))

        right_neighbors = targets + 1
        right_mask = right_neighbors < self.num_classes
        if right_mask.any():
            right_probs = probabilities[right_mask].gather(1, right_neighbors[right_mask].unsqueeze(1)).squeeze(1)
            penalties.append(F.relu(self.margin - (gt_probs[right_mask] - right_probs)))

        if not penalties:
            return probabilities.new_tensor(0.0)
        return torch.cat([item.reshape(-1) for item in penalties], dim=0).mean()


class DecodedUnimodalityRegularizer(nn.Module):
    def __init__(self, margin: float = 0.0) -> None:
        super().__init__()
        self.margin = float(margin)

    def forward(self, probabilities: torch.Tensor) -> torch.Tensor:
        class_probabilities = probabilities.float()
        if class_probabilities.ndim != 2 or class_probabilities.size(1) < 3:
            return class_probabilities.new_tensor(0.0)

        penalties: list[torch.Tensor] = []
        peak_indices = class_probabilities.argmax(dim=1)
        num_classes = class_probabilities.size(1)
        for boundary in range(num_classes - 1):
            left_mask = peak_indices > boundary
            if left_mask.any():
                penalties.append(
                    F.relu(
                        class_probabilities[left_mask, boundary] - class_probabilities[left_mask, boundary + 1] - self.margin
                    )
                )
            right_mask = peak_indices <= boundary
            if right_mask.any():
                penalties.append(
                    F.relu(
                        class_probabilities[right_mask, boundary + 1] - class_probabilities[right_mask, boundary] - self.margin
                    )
                )
        if not penalties:
            return class_probabilities.new_tensor(0.0)
        return torch.cat([item.reshape(-1) for item in penalties], dim=0).mean()


class OrdinalSupConLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.10,
        adjacent_weight: float = 1.0,
        far_weight: float = 0.5,
        minor_major_weight: float = 1.75,
        min_positives: int = 1,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.adjacent_weight = float(adjacent_weight)
        self.far_weight = float(far_weight)
        self.min_positives = int(max(min_positives, 1))
        self.minor_major_weight = float(minor_major_weight)

    def _build_pair_weights(self, labels: torch.Tensor) -> torch.Tensor:
        label_distance = torch.abs(labels.unsqueeze(1) - labels.unsqueeze(0))
        pair_weights = torch.ones_like(label_distance, dtype=torch.float32)
        # Adjacent mistakes matter most for ordinal ranking, and minor<->major gets an extra bump.
        pair_weights = torch.where(label_distance == 1, pair_weights.new_tensor(self.adjacent_weight), pair_weights)
        pair_weights = torch.where(label_distance >= 2, pair_weights.new_tensor(self.far_weight), pair_weights)
        minor_major_mask = ((labels.unsqueeze(1) == 1) & (labels.unsqueeze(0) == 2)) | (
            (labels.unsqueeze(1) == 2) & (labels.unsqueeze(0) == 1)
        )
        pair_weights = torch.where(minor_major_mask, pair_weights.new_tensor(self.minor_major_weight), pair_weights)
        return pair_weights

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2 or embeddings.size(0) <= 1:
            return embeddings.new_tensor(0.0)

        normalized_embeddings = F.normalize(embeddings.float(), dim=1)
        label_tensor = labels.reshape(-1).to(device=normalized_embeddings.device)
        if label_tensor.numel() != normalized_embeddings.size(0):
            raise ValueError(
                "OrdinalSupConLoss expects embeddings and labels to agree on the batch dimension, "
                f"got embeddings={tuple(embeddings.shape)} labels={tuple(labels.shape)}."
            )

        logits = torch.matmul(normalized_embeddings, normalized_embeddings.t()) / max(self.temperature, 1e-6)
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        valid_mask = ~torch.eye(logits.size(0), device=logits.device, dtype=torch.bool)
        positive_mask = (label_tensor.unsqueeze(1) == label_tensor.unsqueeze(0)) & valid_mask
        pair_weights = self._build_pair_weights(label_tensor).to(device=logits.device, dtype=logits.dtype)
        pair_weights = pair_weights * valid_mask.to(dtype=logits.dtype)

        positive_counts = positive_mask.sum(dim=1)
        valid_anchors = positive_counts >= self.min_positives
        if not valid_anchors.any():
            return embeddings.new_tensor(0.0)

        exp_logits = torch.exp(logits) * pair_weights
        denominator = exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8)
        log_prob = logits - denominator.log()
        positive_log_prob = (log_prob * positive_mask.to(dtype=logits.dtype)).sum(dim=1) / positive_counts.clamp_min(1)
        loss = -positive_log_prob[valid_anchors]
        if loss.numel() == 0:
            return embeddings.new_tensor(0.0)
        return loss.mean().to(device=embeddings.device, dtype=embeddings.dtype)


class OrderAwareContrastiveLoss(nn.Module):
    def __init__(
        self,
        distance: str = "cosine",
        margin_adjacent: float = 0.45,
        margin_far: float = 0.90,
        far_weight: float = 1.25,
        margin_gap1: float | None = None,
        margin_gap2: float | None = None,
        margin_gap3: float = 1.10,
        pair_weight_same: float = 1.0,
        pair_weight_gap1: float = 1.2,
        pair_weight_gap2: float | None = None,
        pair_weight_gap3: float | None = None,
        minor_major_pair_boost: float = 1.1,
    ) -> None:
        super().__init__()
        if distance not in {"cosine", "euclidean"}:
            raise ValueError(f"Unsupported contrastive distance='{distance}'.")
        self.margin_gap1 = float(margin_adjacent if margin_gap1 is None else margin_gap1)
        self.margin_gap2 = float(margin_far if margin_gap2 is None else margin_gap2)
        self.margin_gap3 = float(margin_gap3)
        if self.margin_gap2 <= self.margin_gap1:
            raise ValueError("margin_gap2 must be larger than margin_gap1.")
        if self.margin_gap3 <= self.margin_gap2:
            raise ValueError("margin_gap3 must be larger than margin_gap2.")
        self.distance = str(distance)
        self.margin_adjacent = float(self.margin_gap1)
        self.margin_far = float(self.margin_gap2)
        self.far_weight = float(far_weight)
        self.pair_weight_same = float(pair_weight_same)
        self.pair_weight_gap1 = float(pair_weight_gap1)
        self.pair_weight_gap2 = float(self.far_weight if pair_weight_gap2 is None else pair_weight_gap2)
        self.pair_weight_gap3 = float(self.far_weight if pair_weight_gap3 is None else pair_weight_gap3)
        self.minor_major_pair_boost = float(minor_major_pair_boost)

    def _pairwise_distance(self, embeddings: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(embeddings.float(), dim=1)
        if self.distance == "euclidean":
            return torch.cdist(normalized, normalized, p=2)
        cosine_similarity = torch.matmul(normalized, normalized.t()).clamp(-1.0, 1.0)
        return (1.0 - cosine_similarity).clamp_min(0.0)

    def _empty_output(self, zero: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "loss": zero,
            "mean_same_distance": zero,
            "mean_gap1_distance": zero,
            "mean_gap2_distance": zero,
            "mean_gap3_distance": zero,
            "mean_adjacent_distance": zero,
            "mean_far_distance": zero,
            "gap0_pair_count": zero,
            "gap1_pair_count": zero,
            "gap2_pair_count": zero,
            "gap3_pair_count": zero,
            "same_pair_count": zero,
            "adjacent_pair_count": zero,
            "far_pair_count": zero,
            "pair_count_by_gap": {
                "gap0": zero,
                "gap1": zero,
                "gap2": zero,
                "gap3": zero,
            },
        }

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        zero = embeddings.new_tensor(0.0)
        if embeddings.ndim != 2 or embeddings.size(0) <= 1:
            return self._empty_output(zero)

        label_tensor = labels.reshape(-1).to(device=embeddings.device, dtype=torch.long)
        if label_tensor.numel() != embeddings.size(0):
            raise ValueError(
                "OrderAwareContrastiveLoss expects embeddings and labels to agree on the batch dimension, "
                f"got embeddings={tuple(embeddings.shape)} labels={tuple(labels.shape)}."
            )

        pairwise_distance = self._pairwise_distance(embeddings)
        upper_indices = torch.triu_indices(pairwise_distance.size(0), pairwise_distance.size(1), offset=1, device=embeddings.device)
        if upper_indices.numel() == 0:
            return self._empty_output(zero)

        pair_distances = pairwise_distance[upper_indices[0], upper_indices[1]]
        pair_distances = pair_distances.float()
        label_gaps = (label_tensor[upper_indices[0]] - label_tensor[upper_indices[1]]).abs()
        left_labels = label_tensor[upper_indices[0]]
        right_labels = label_tensor[upper_indices[1]]
        finite_mask = torch.isfinite(pair_distances)
        if not finite_mask.any():
            return self._empty_output(zero)

        pair_distances = pair_distances[finite_mask]
        label_gaps = label_gaps[finite_mask]
        left_labels = left_labels[finite_mask]
        right_labels = right_labels[finite_mask]

        gap0_mask = label_gaps == 0
        gap1_mask = label_gaps == 1
        gap2_mask = label_gaps == 2
        gap3_mask = label_gaps >= 3
        far_mask = gap2_mask | gap3_mask

        per_pair_loss = pair_distances.new_zeros(pair_distances.shape)
        per_pair_weight = pair_distances.new_zeros(pair_distances.shape)

        if gap0_mask.any():
            per_pair_loss[gap0_mask] = pair_distances[gap0_mask].pow(2)
            per_pair_weight[gap0_mask] = float(self.pair_weight_same)
        if gap1_mask.any():
            per_pair_loss[gap1_mask] = F.relu(pair_distances.new_tensor(self.margin_gap1) - pair_distances[gap1_mask]).pow(2)
            per_pair_weight[gap1_mask] = float(self.pair_weight_gap1)
        if gap2_mask.any():
            per_pair_loss[gap2_mask] = F.relu(pair_distances.new_tensor(self.margin_gap2) - pair_distances[gap2_mask]).pow(2)
            per_pair_weight[gap2_mask] = float(self.pair_weight_gap2)
        if gap3_mask.any():
            per_pair_loss[gap3_mask] = F.relu(pair_distances.new_tensor(self.margin_gap3) - pair_distances[gap3_mask]).pow(2)
            per_pair_weight[gap3_mask] = float(self.pair_weight_gap3)

        minor_major_mask = ((left_labels == 1) & (right_labels == 2)) | ((left_labels == 2) & (right_labels == 1))
        if self.minor_major_pair_boost > 0.0 and minor_major_mask.any():
            per_pair_weight[minor_major_mask] = per_pair_weight[minor_major_mask] * float(self.minor_major_pair_boost)

        valid_loss_mask = per_pair_weight > 0.0
        if valid_loss_mask.any():
            weighted_loss = per_pair_loss[valid_loss_mask] * per_pair_weight[valid_loss_mask]
            loss = (
                weighted_loss.sum() / per_pair_weight[valid_loss_mask].sum().clamp_min(1e-8)
            ).to(device=embeddings.device, dtype=embeddings.dtype)
        else:
            loss = zero

        def _mean_distance(mask: torch.Tensor) -> torch.Tensor:
            if not mask.any():
                return zero
            return pair_distances[mask].mean().to(device=embeddings.device, dtype=embeddings.dtype)

        def _count(mask: torch.Tensor) -> torch.Tensor:
            return pair_distances.new_tensor(float(mask.sum().item())).to(device=embeddings.device, dtype=embeddings.dtype)

        return {
            "loss": loss,
            "mean_same_distance": _mean_distance(gap0_mask),
            "mean_gap1_distance": _mean_distance(gap1_mask),
            "mean_gap2_distance": _mean_distance(gap2_mask),
            "mean_gap3_distance": _mean_distance(gap3_mask),
            "mean_adjacent_distance": _mean_distance(gap1_mask),
            "mean_far_distance": _mean_distance(far_mask),
            "gap0_pair_count": _count(gap0_mask),
            "gap1_pair_count": _count(gap1_mask),
            "gap2_pair_count": _count(gap2_mask),
            "gap3_pair_count": _count(gap3_mask),
            "same_pair_count": _count(gap0_mask),
            "adjacent_pair_count": _count(gap1_mask),
            "far_pair_count": _count(far_mask),
            "pair_count_by_gap": {
                "gap0": _count(gap0_mask),
                "gap1": _count(gap1_mask),
                "gap2": _count(gap2_mask),
                "gap3": _count(gap3_mask),
            },
        }


class TauRegularizer(nn.Module):
    def __init__(
        self,
        tau_target: float = 0.22,
        variance_weight: float = 1e-3,
        std_floor: float = 0.03,
    ) -> None:
        super().__init__()
        self.tau_target = float(tau_target)
        self.variance_weight = float(variance_weight)
        self.std_floor = float(std_floor)

    def forward(self, tau: torch.Tensor) -> dict[str, torch.Tensor]:
        tau_values = tau.float().reshape(-1)
        tau_mean = tau_values.mean() if tau_values.numel() > 0 else tau.new_tensor(self.tau_target).float()
        tau_std = tau_values.std(unbiased=False) if tau_values.numel() > 1 else tau_values.new_tensor(0.0)
        loss_mean = (tau_mean - self.tau_target).pow(2)
        loss_var = tau_values.new_tensor(0.0)
        if tau_values.numel() > 1 and self.variance_weight > 0.0:
            loss_var = self.variance_weight * F.relu(self.std_floor - tau_std).pow(2)
        loss_tau = loss_mean + loss_var
        return {
            "loss_tau_reg": loss_tau.to(device=tau.device, dtype=tau.dtype),
            "tau_mean": tau_mean.to(device=tau.device, dtype=tau.dtype),
            "tau_std": tau_std.to(device=tau.device, dtype=tau.dtype),
            "loss_tau_mean": loss_mean.to(device=tau.device, dtype=tau.dtype),
            "loss_tau_var": loss_var.to(device=tau.device, dtype=tau.dtype),
        }


class DifficultyGuidedTauRegularizer(nn.Module):
    def __init__(
        self,
        tau_easy: float = 0.16,
        tau_hard: float = 0.32,
    ) -> None:
        super().__init__()
        if tau_hard <= tau_easy:
            raise ValueError("tau_hard must be larger than tau_easy.")
        self.tau_easy = float(tau_easy)
        self.tau_hard = float(tau_hard)

    def forward(
        self,
        tau: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        difficulty, gt_probability = compute_sample_difficulty(logits, targets)
        tau_ref = build_tau_reference_from_difficulty(tau, difficulty, self.tau_easy, self.tau_hard)
        loss_tau_diff = F.smooth_l1_loss(tau.float(), tau_ref.float())
        difficulty_stats = _compute_scalar_statistics(difficulty)
        corr_tau_difficulty = _compute_pearson_correlation(tau, difficulty)
        return {
            "loss_tau_diff": loss_tau_diff.to(device=tau.device, dtype=tau.dtype),
            "tau_ref": tau_ref.to(device=tau.device, dtype=tau.dtype),
            "difficulty": difficulty.to(device=tau.device, dtype=tau.dtype),
            "gt_probability": gt_probability.to(device=tau.device, dtype=tau.dtype),
            "difficulty_stats": {
                key: value.to(device=tau.device, dtype=tau.dtype)
                for key, value in difficulty_stats.items()
            },
            "corr_tau_difficulty": corr_tau_difficulty.to(device=tau.device, dtype=tau.dtype),
        }


class PairwiseTauRankRegularizer(nn.Module):
    def __init__(
        self,
        margin_difficulty: float = 0.10,
        margin_value: float = 0.01,
    ) -> None:
        super().__init__()
        self.margin_difficulty = float(margin_difficulty)
        self.margin_value = float(margin_value)

    def forward(self, tau: torch.Tensor, difficulty: torch.Tensor) -> torch.Tensor:
        tau_values = tau.float().reshape(-1)
        difficulty_values = difficulty.detach().float().reshape(-1)
        if tau_values.numel() < 2 or tau_values.numel() != difficulty_values.numel():
            return tau.new_tensor(0.0)

        sorted_indices = torch.argsort(difficulty_values, descending=True)
        tau_sorted = tau_values[sorted_indices]
        difficulty_sorted = difficulty_values[sorted_indices]

        difficulty_gap = difficulty_sorted[:-1] - difficulty_sorted[1:]
        valid_mask = difficulty_gap > self.margin_difficulty
        if not valid_mask.any():
            return tau.new_tensor(0.0)

        harder_tau = tau_sorted[:-1][valid_mask]
        easier_tau = tau_sorted[1:][valid_mask]
        penalties = F.relu(easier_tau - harder_tau + self.margin_value)
        if penalties.numel() == 0:
            return tau.new_tensor(0.0)
        return penalties.mean().to(device=tau.device, dtype=tau.dtype)


class AdaptiveUCLCDALoss(LearnableSeverityAxis):
    def __init__(
        self,
        num_classes: int = 4,
        tau_init: float = 0.35,
        tau_min: float = 0.05,
        tau_max: float = 1.0,
        concentration_margin: float = 0.05,
    ) -> None:
        super().__init__(num_classes=num_classes)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.reference_tau_init = float(min(max(tau_init, tau_min), tau_max))
        self.unimodality = UnimodalityRegularizer(num_classes=num_classes)
        self.concentration = ConcentrationRegularizer(num_classes=num_classes, margin=concentration_margin)

        self.register_buffer("last_tau_mean", torch.tensor(self.reference_tau_init, dtype=torch.float32))
        self.register_buffer("last_tau_std", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("last_tau_min", torch.tensor(self.reference_tau_init, dtype=torch.float32))
        self.register_buffer("last_tau_max", torch.tensor(self.reference_tau_init, dtype=torch.float32))
        self.register_buffer("last_tau_p10", torch.tensor(self.reference_tau_init, dtype=torch.float32))
        self.register_buffer("last_tau_p50", torch.tensor(self.reference_tau_init, dtype=torch.float32))
        self.register_buffer("last_tau_p90", torch.tensor(self.reference_tau_init, dtype=torch.float32))
        self.register_buffer("last_difficulty_mean", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("last_difficulty_std", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("last_corr_tau_difficulty", torch.tensor(0.0, dtype=torch.float32))

    def _update_tau_stats(self, tau: torch.Tensor) -> None:
        stats = _compute_scalar_statistics(tau)
        self.last_tau_mean.copy_(stats["mean"])
        self.last_tau_std.copy_(stats["std"])
        self.last_tau_min.copy_(stats["min"])
        self.last_tau_max.copy_(stats["max"])
        self.last_tau_p10.copy_(stats["p10"])
        self.last_tau_p50.copy_(stats["p50"])
        self.last_tau_p90.copy_(stats["p90"])

    def _current_tau_stats_tensor(self, *, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        return {
            "mean": self.last_tau_mean.to(device=device, dtype=dtype),
            "std": self.last_tau_std.to(device=device, dtype=dtype),
            "min": self.last_tau_min.to(device=device, dtype=dtype),
            "max": self.last_tau_max.to(device=device, dtype=dtype),
            "p10": self.last_tau_p10.to(device=device, dtype=dtype),
            "p50": self.last_tau_p50.to(device=device, dtype=dtype),
            "p90": self.last_tau_p90.to(device=device, dtype=dtype),
        }

    def _update_difficulty_stats(self, tau: torch.Tensor, difficulty: torch.Tensor | None) -> None:
        if difficulty is None or difficulty.numel() == 0:
            self.last_difficulty_mean.zero_()
            self.last_difficulty_std.zero_()
            self.last_corr_tau_difficulty.zero_()
            return

        stats = _compute_scalar_statistics(difficulty)
        self.last_difficulty_mean.copy_(stats["mean"])
        self.last_difficulty_std.copy_(stats["std"])
        self.last_corr_tau_difficulty.copy_(_compute_pearson_correlation(tau, difficulty))

    def _current_difficulty_stats_tensor(self, *, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        return {
            "mean": self.last_difficulty_mean.to(device=device, dtype=dtype),
            "std": self.last_difficulty_std.to(device=device, dtype=dtype),
        }

    def _compute_outputs(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        tau: torch.Tensor,
        *,
        raw_tau: torch.Tensor | None = None,
        difficulty_values: torch.Tensor | None = None,
        tau_ref: torch.Tensor | None = None,
        raw_tau_ref: torch.Tensor | None = None,
        loss_tau_reg: torch.Tensor | None = None,
        loss_tau_mean: torch.Tensor | None = None,
        loss_tau_diff: torch.Tensor | None = None,
        loss_tau_var: torch.Tensor | None = None,
        loss_tau_rank: torch.Tensor | None = None,
        loss_raw_tau_diff: torch.Tensor | None = None,
        loss_raw_tau_center: torch.Tensor | None = None,
        loss_raw_tau_bound: torch.Tensor | None = None,
        corr_raw_tau_difficulty: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        gaps, soft_targets = self.build_soft_targets(targets, tau, device=logits.device, dtype=logits.dtype)
        positions = self.get_current_positions().to(device=logits.device, dtype=logits.dtype)
        probabilities = F.softmax(logits, dim=1)
        loss_unimodal = self.unimodality(probabilities, targets)
        loss_concentration = self.concentration(probabilities, targets)
        soft_target_entropy = -(soft_targets * soft_targets.clamp_min(1e-8).log()).sum(dim=1).mean()
        soft_target_gt_mass = soft_targets.gather(1, targets.unsqueeze(1)).mean()
        self._update_tau_stats(tau)
        self._update_difficulty_stats(tau, difficulty_values)

        zero = logits.new_tensor(0.0)
        resolved_loss_tau_var = zero if loss_tau_var is None else loss_tau_var.to(device=logits.device, dtype=logits.dtype)
        resolved_loss_tau_mean = zero if loss_tau_mean is None else loss_tau_mean.to(device=logits.device, dtype=logits.dtype)
        resolved_loss_tau_diff = zero if loss_tau_diff is None else loss_tau_diff.to(device=logits.device, dtype=logits.dtype)
        resolved_loss_tau_rank = zero if loss_tau_rank is None else loss_tau_rank.to(device=logits.device, dtype=logits.dtype)
        resolved_loss_raw_tau_diff = (
            zero if loss_raw_tau_diff is None else loss_raw_tau_diff.to(device=logits.device, dtype=logits.dtype)
        )
        resolved_loss_raw_tau_center = (
            zero if loss_raw_tau_center is None else loss_raw_tau_center.to(device=logits.device, dtype=logits.dtype)
        )
        resolved_loss_raw_tau_bound = (
            zero if loss_raw_tau_bound is None else loss_raw_tau_bound.to(device=logits.device, dtype=logits.dtype)
        )
        if loss_tau_reg is None:
            resolved_loss_tau_reg = resolved_loss_tau_mean + resolved_loss_tau_var
        else:
            resolved_loss_tau_reg = loss_tau_reg.to(device=logits.device, dtype=logits.dtype)
        resolved_corr_raw_tau_difficulty = (
            None
            if corr_raw_tau_difficulty is None
            else corr_raw_tau_difficulty.to(device=logits.device, dtype=logits.dtype)
        )

        return {
            "loss_ord": _soft_cross_entropy(logits, soft_targets),
            "loss_unimodal": loss_unimodal,
            "loss_concentration": loss_concentration,
            "loss_tau_reg": resolved_loss_tau_reg,
            "loss_tau_mean": resolved_loss_tau_mean,
            "loss_tau_var": resolved_loss_tau_var,
            "loss_tau_diff": resolved_loss_tau_diff,
            "loss_tau_rank": resolved_loss_tau_rank,
            "loss_raw_tau_diff": resolved_loss_raw_tau_diff,
            "loss_raw_tau_center": resolved_loss_raw_tau_center,
            "loss_raw_tau_bound": resolved_loss_raw_tau_bound,
            "soft_targets": soft_targets,
            "soft_target_stats": {
                "entropy_mean": soft_target_entropy,
                "gt_mass_mean": soft_target_gt_mass,
            },
            "positions": positions,
            "tau": tau,
            "raw_tau": raw_tau,
            "tau_ref": tau_ref,
            "raw_tau_ref": raw_tau_ref,
            "gaps": gaps,
            "tau_stats": self._current_tau_stats_tensor(device=logits.device, dtype=logits.dtype),
            "difficulty": difficulty_values,
            "difficulty_values": difficulty_values,
            "difficulty_stats": self._current_difficulty_stats_tensor(device=logits.device, dtype=logits.dtype),
            "corr_tau_difficulty": self.last_corr_tau_difficulty.to(device=logits.device, dtype=logits.dtype),
            "corr_raw_tau_difficulty": resolved_corr_raw_tau_difficulty,
        }

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_tau: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if sample_tau is None:
            raise ValueError("AdaptiveUCLCDALoss requires per-sample tau from the ambiguity head.")

        if sample_tau.ndim == 0:
            sample_tau = sample_tau.unsqueeze(0)
        tau = sample_tau.to(device=logits.device, dtype=logits.dtype).clamp(self.tau_min, self.tau_max)
        difficulty_values, _ = compute_sample_difficulty(logits, targets)
        return self._compute_outputs(logits, targets, tau, difficulty_values=difficulty_values)

    def get_current_tau(self) -> torch.Tensor:
        return self.last_tau_mean.detach().clone()

    def get_last_tau_statistics(self) -> dict[str, float]:
        return {
            "mean": float(self.last_tau_mean.detach().cpu().item()),
            "std": float(self.last_tau_std.detach().cpu().item()),
            "min": float(self.last_tau_min.detach().cpu().item()),
            "max": float(self.last_tau_max.detach().cpu().item()),
            "p10": float(self.last_tau_p10.detach().cpu().item()),
            "p50": float(self.last_tau_p50.detach().cpu().item()),
            "p90": float(self.last_tau_p90.detach().cpu().item()),
        }

    def get_last_difficulty_statistics(self) -> dict[str, float]:
        return {
            "mean": float(self.last_difficulty_mean.detach().cpu().item()),
            "std": float(self.last_difficulty_std.detach().cpu().item()),
        }

    def get_last_tau_difficulty_correlation(self) -> float:
        return float(self.last_corr_tau_difficulty.detach().cpu().item())

    def get_current_soft_target_matrix(self) -> torch.Tensor:
        positions = self.get_current_positions()
        tau = self.get_current_tau().clamp_min(1e-6)
        distances = torch.abs(positions.unsqueeze(1) - positions.unsqueeze(0))
        return torch.softmax(-distances / tau, dim=1).detach()


class AdaptiveUCLCDALossV2(AdaptiveUCLCDALoss):
    def __init__(
        self,
        num_classes: int = 4,
        tau_init: float = 0.27,
        tau_min: float = 0.10,
        tau_max: float = 0.60,
        concentration_margin: float = 0.05,
        tau_target: float = 0.22,
        tau_variance_weight: float = 1e-3,
        tau_std_floor: float = 0.03,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            tau_init=tau_init,
            tau_min=tau_min,
            tau_max=tau_max,
            concentration_margin=concentration_margin,
        )
        self.tau_regularizer = TauRegularizer(
            tau_target=tau_target,
            variance_weight=tau_variance_weight,
            std_floor=tau_std_floor,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_tau: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        outputs = super().forward(logits, targets, sample_tau=sample_tau)
        tau_reg_outputs = self.tau_regularizer(outputs["tau"])
        outputs["loss_tau_reg"] = tau_reg_outputs["loss_tau_reg"]
        outputs["loss_tau_mean"] = tau_reg_outputs["loss_tau_mean"]
        outputs["loss_tau_diff"] = logits.new_tensor(0.0)
        outputs["loss_tau_var"] = tau_reg_outputs["loss_tau_var"]
        outputs["loss_tau_rank"] = logits.new_tensor(0.0)
        return outputs


class AdaptiveUCLCDALossV3(AdaptiveUCLCDALoss):
    def __init__(
        self,
        num_classes: int = 4,
        tau_init: float = 0.22,
        tau_min: float = 0.12,
        tau_max: float = 0.45,
        concentration_margin: float = 0.05,
        tau_target: float = 0.22,
        tau_easy: float = 0.16,
        tau_hard: float = 0.32,
        tau_variance_weight: float = 1e-2,
        tau_std_floor: float = 0.03,
        tau_rank_margin_difficulty: float = 0.10,
        tau_rank_margin_value: float = 0.01,
        raw_tau_soft_margin: float = 1.5,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            tau_init=tau_init,
            tau_min=tau_min,
            tau_max=tau_max,
            concentration_margin=concentration_margin,
        )
        self.tau_target = float(tau_target)
        self.tau_diff_regularizer = DifficultyGuidedTauRegularizer(
            tau_easy=tau_easy,
            tau_hard=tau_hard,
        )
        self.tau_regularizer = TauRegularizer(
            tau_target=tau_target,
            variance_weight=tau_variance_weight,
            std_floor=tau_std_floor,
        )
        self.raw_tau_center = _compute_raw_tau_center(self.tau_target, self.tau_min, self.tau_max)
        self.raw_tau_soft_margin = float(raw_tau_soft_margin)
        self.tau_rank_regularizer = PairwiseTauRankRegularizer(
            margin_difficulty=tau_rank_margin_difficulty,
            margin_value=tau_rank_margin_value,
        )
        self.register_buffer("last_corr_raw_tau_difficulty", torch.tensor(0.0, dtype=torch.float32))

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_tau: torch.Tensor | None,
        raw_tau: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if sample_tau is None:
            raise ValueError("AdaptiveUCLCDALossV3 requires per-sample tau from the ambiguity head.")
        if raw_tau is None:
            raise ValueError("AdaptiveUCLCDALossV3 requires raw_tau from the ambiguity head for logit-space regularization.")

        if sample_tau.ndim == 0:
            sample_tau = sample_tau.unsqueeze(0)
        if raw_tau.ndim == 0:
            raw_tau = raw_tau.unsqueeze(0)

        tau = sample_tau.to(device=logits.device, dtype=logits.dtype).clamp(self.tau_min, self.tau_max)
        raw_tau = raw_tau.to(device=logits.device, dtype=logits.dtype)
        if raw_tau.numel() != tau.numel():
            raise ValueError(
                "AdaptiveUCLCDALossV3 expects raw_tau and sample_tau to have the same number of elements, "
                f"got raw_tau={tuple(raw_tau.shape)} and sample_tau={tuple(tau.shape)}."
            )

        tau_reg_outputs = self.tau_regularizer(tau)
        tau_diff_outputs = self.tau_diff_regularizer(tau, logits, targets)
        loss_tau_rank = self.tau_rank_regularizer(tau, tau_diff_outputs["difficulty"])
        raw_tau_ref = _tau_to_raw_tau_logit(tau_diff_outputs["tau_ref"], self.tau_min, self.tau_max).to(
            device=logits.device,
            dtype=logits.dtype,
        )
        loss_raw_tau_diff = F.smooth_l1_loss(raw_tau.float(), raw_tau_ref.float())
        raw_tau_values = raw_tau.float().reshape(-1)
        raw_tau_center = raw_tau.new_tensor(self.raw_tau_center).float()
        raw_tau_mean = raw_tau_values.mean() if raw_tau_values.numel() > 0 else raw_tau_center
        loss_raw_tau_center = (raw_tau_mean - raw_tau_center).pow(2)
        raw_tau_soft_margin = raw_tau.new_tensor(self.raw_tau_soft_margin).float()
        raw_tau_margin_excess = F.relu((raw_tau_values - raw_tau_center).abs() - raw_tau_soft_margin)
        loss_raw_tau_bound = (
            raw_tau_margin_excess.pow(2).mean()
            if raw_tau_margin_excess.numel() > 0
            else raw_tau.new_tensor(0.0).float()
        )
        corr_raw_tau_difficulty = _compute_pearson_correlation(raw_tau, tau_diff_outputs["difficulty"])
        self.last_corr_raw_tau_difficulty.copy_(corr_raw_tau_difficulty.detach().float())

        outputs = self._compute_outputs(
            logits,
            targets,
            tau,
            raw_tau=raw_tau,
            difficulty_values=tau_diff_outputs["difficulty"],
            tau_ref=tau_diff_outputs["tau_ref"],
            raw_tau_ref=raw_tau_ref,
            loss_tau_reg=tau_reg_outputs["loss_tau_reg"],
            loss_tau_mean=tau_reg_outputs["loss_tau_mean"],
            loss_tau_var=tau_reg_outputs["loss_tau_var"],
            loss_tau_diff=tau_diff_outputs["loss_tau_diff"],
            loss_tau_rank=loss_tau_rank,
            loss_raw_tau_diff=loss_raw_tau_diff.to(device=logits.device, dtype=logits.dtype),
            loss_raw_tau_center=loss_raw_tau_center.to(device=logits.device, dtype=logits.dtype),
            loss_raw_tau_bound=loss_raw_tau_bound.to(device=logits.device, dtype=logits.dtype),
            corr_raw_tau_difficulty=corr_raw_tau_difficulty,
        )
        outputs["difficulty_stats"] = tau_diff_outputs["difficulty_stats"]
        outputs["corr_tau_difficulty"] = tau_diff_outputs["corr_tau_difficulty"]
        outputs["corr_raw_tau_difficulty"] = corr_raw_tau_difficulty.to(device=logits.device, dtype=logits.dtype)
        outputs["gt_probability"] = tau_diff_outputs["gt_probability"]
        outputs["difficulty"] = tau_diff_outputs["difficulty"]
        return outputs

    def get_last_raw_tau_difficulty_correlation(self) -> float:
        return float(self.last_corr_raw_tau_difficulty.detach().cpu().item())


def corn_logits_to_threshold_probabilities(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits.float())


def decode_corn_probabilities(
    threshold_probabilities: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_thresholds = threshold_probabilities.shape
    num_classes = num_thresholds + 1
    conditional = threshold_probabilities.float().clamp(1e-6, 1.0 - 1e-6)
    survival = torch.ones(
        batch_size,
        num_classes,
        device=threshold_probabilities.device,
        dtype=conditional.dtype,
    )
    survival[:, 1:] = torch.cumprod(conditional, dim=1)
    class_probabilities = torch.zeros_like(survival)
    class_probabilities[:, :-1] = survival[:, :-1] - survival[:, 1:]
    class_probabilities[:, -1] = survival[:, -1]
    return class_probabilities.clamp_min(1e-8)


def decode_corn_logits(corn_logits: torch.Tensor) -> torch.Tensor:
    return decode_corn_probabilities(corn_logits_to_threshold_probabilities(corn_logits))


class CORNLoss(nn.Module):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        if int(num_classes) < 2:
            raise ValueError("CORNLoss expects at least 2 ordered classes.")
        self.num_classes = int(num_classes)

    @staticmethod
    def logits_to_threshold_probabilities(logits: torch.Tensor) -> torch.Tensor:
        return corn_logits_to_threshold_probabilities(logits)

    @classmethod
    def threshold_probabilities_to_class_probabilities(cls, threshold_probabilities: torch.Tensor) -> torch.Tensor:
        return decode_corn_probabilities(threshold_probabilities)

    @classmethod
    def logits_to_class_probabilities(cls, logits: torch.Tensor) -> torch.Tensor:
        return decode_corn_logits(logits)

    @classmethod
    def logits_to_predictions(cls, logits: torch.Tensor) -> torch.Tensor:
        probabilities = cls.logits_to_class_probabilities(logits)
        return probabilities.argmax(dim=1)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        if logits.size(1) != self.num_classes - 1:
            raise ValueError(
                f"CORNLoss expects {self.num_classes - 1} ordinal logits, got shape={tuple(logits.shape)}."
            )

        total_loss = logits.new_tensor(0.0)
        num_examples = 0
        task_losses = logits.new_zeros(self.num_classes - 1)
        task_counts = logits.new_zeros(self.num_classes - 1)

        for threshold_idx in range(self.num_classes - 1):
            if threshold_idx == 0:
                label_mask = torch.ones_like(targets, dtype=torch.bool)
            else:
                label_mask = targets > (threshold_idx - 1)

            if not label_mask.any():
                continue

            binary_targets = (targets[label_mask] > threshold_idx).to(dtype=logits.dtype)
            threshold_logits = logits[label_mask, threshold_idx]
            task_loss = F.binary_cross_entropy_with_logits(
                threshold_logits,
                binary_targets,
                reduction="sum",
            )
            total_loss = total_loss + task_loss
            task_count = int(binary_targets.numel())
            num_examples += task_count
            task_losses[threshold_idx] = task_loss / max(task_count, 1)
            task_counts[threshold_idx] = float(task_count)

        threshold_probabilities = self.logits_to_threshold_probabilities(logits)
        class_probabilities = self.logits_to_class_probabilities(logits)
        normalized_loss = total_loss / max(num_examples, 1)
        return {
            "loss_ord": normalized_loss,
            "loss_corn_main": normalized_loss,
            "positions": torch.linspace(0.0, 1.0, steps=self.num_classes, device=logits.device, dtype=logits.dtype),
            "threshold_probabilities": threshold_probabilities,
            "class_probabilities": class_probabilities,
            "task_losses": task_losses,
            "task_counts": task_counts,
        }


class DamageLossModule(nn.Module):
    def __init__(
        self,
        loss_mode: str,
        class_weights: torch.Tensor | None,
        num_classes: int = 4,
        label_smoothing: float = 0.05,
        lambda_ord: float = 0.3,
        lambda_uni: float = 0.10,
        lambda_conc: float = 0.05,
        lambda_gap_reg: float = 1e-3,
        lambda_tau: float = 0.01,
        lambda_tau_mean: float = 0.05,
        lambda_tau_diff: float = 0.20,
        lambda_tau_rank: float = 0.05,
        lambda_raw_tau_diff: float = 0.10,
        lambda_raw_tau_center: float = 0.02,
        lambda_raw_tau_bound: float = 0.02,
        lambda_corn_soft: float = 0.03,
        enable_corn_soft_emd: bool = False,
        lambda_corn_soft_emd: float = 0.02,
        enable_decoded_unimodal_reg: bool = False,
        lambda_decoded_unimodal: float = 0.01,
        decoded_unimodal_margin: float = 0.0,
        fixed_cda_alpha: float = 0.3,
        tau_init: float = 0.35,
        tau_min: float = 0.05,
        tau_max: float = 1.0,
        tau_base: float = 0.22,
        delta_scale: float = 0.12,
        tau_parameterization: str = "sigmoid",
        tau_logit_scale: float = 2.0,
        tau_target: float = 0.22,
        tau_easy: float = 0.16,
        tau_hard: float = 0.32,
        tau_variance_weight: float = 1e-2,
        tau_std_floor: float = 0.03,
        tau_rank_margin_difficulty: float = 0.10,
        tau_rank_margin_value: float = 0.01,
        raw_tau_soft_margin: float = 1.5,
        concentration_margin: float = 0.05,
        lambda_aux: float = 0.2,
        aux_soft_label_enabled: bool = False,
        aux_soft_label_weight: float = 0.0,
        aux_soft_label_target_distribution: torch.Tensor | list[list[float]] | tuple[tuple[float, ...], ...] | None = None,
        lambda_distribution: float = 0.18,
        lambda_severity_reg: float = 0.08,
        lambda_consistency_dist: float = 0.03,
        lambda_consistency_severity: float = 0.03,
        distribution_head_enabled: bool = False,
        distribution_target_distribution: torch.Tensor | list[list[float]] | tuple[tuple[float, ...], ...] | None = None,
        distribution_class_weights: torch.Tensor | list[float] | tuple[float, ...] | None = None,
        severity_regression_enabled: bool = False,
        severity_regression_class_weights: torch.Tensor | list[float] | tuple[float, ...] | None = None,
        severity_regression_loss: str = "smooth_l1",
        consistency_enabled: bool = False,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.loss_mode = str(loss_mode)
        self.num_classes = int(num_classes)
        self.lambda_ord = float(lambda_ord)
        self.lambda_uni = float(lambda_uni)
        self.lambda_conc = float(lambda_conc)
        self.lambda_gap_reg = float(lambda_gap_reg)
        self.lambda_tau = float(lambda_tau)
        self.lambda_tau_mean = float(lambda_tau_mean)
        self.lambda_tau_diff = float(lambda_tau_diff)
        self.lambda_tau_rank = float(lambda_tau_rank)
        self.lambda_raw_tau_diff = float(lambda_raw_tau_diff)
        self.lambda_raw_tau_center = float(lambda_raw_tau_center)
        self.lambda_raw_tau_bound = float(lambda_raw_tau_bound)
        self.lambda_corn_soft = float(lambda_corn_soft)
        self.enable_corn_soft_emd = bool(enable_corn_soft_emd)
        self.lambda_corn_soft_emd = float(lambda_corn_soft_emd)
        self.enable_decoded_unimodal_reg = bool(enable_decoded_unimodal_reg)
        self.lambda_decoded_unimodal = float(lambda_decoded_unimodal)
        self.decoded_unimodal_margin = float(decoded_unimodal_margin)
        self.fixed_cda_alpha = float(fixed_cda_alpha)
        self.lambda_aux = float(lambda_aux)
        self.aux_soft_label_enabled = bool(aux_soft_label_enabled)
        self.aux_soft_label_weight = float(aux_soft_label_weight)
        self.lambda_distribution = float(lambda_distribution)
        self.lambda_severity_reg = float(lambda_severity_reg)
        self.lambda_consistency_dist = float(lambda_consistency_dist)
        self.lambda_consistency_severity = float(lambda_consistency_severity)
        self.distribution_head_enabled = bool(distribution_head_enabled)
        self.severity_regression_enabled = bool(severity_regression_enabled)
        self.consistency_enabled = bool(consistency_enabled)
        self.severity_regression_loss = str(severity_regression_loss)
        self.use_focal = bool(use_focal)
        self.focal_gamma = float(focal_gamma)
        self.label_smoothing = float(label_smoothing)
        self.corn_soft_detach_target = True
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.tau_base = float(tau_base)
        self.delta_scale = float(delta_scale)
        self.tau_parameterization = str(tau_parameterization)
        self.tau_logit_scale = float(tau_logit_scale)
        self.tau_target = float(tau_target)
        self.tau_easy = float(tau_easy)
        self.tau_hard = float(tau_hard)
        self.tau_variance_weight = float(tau_variance_weight)
        self.tau_std_floor = float(tau_std_floor)
        self.tau_rank_margin_difficulty = float(tau_rank_margin_difficulty)
        self.tau_rank_margin_value = float(tau_rank_margin_value)
        self.raw_tau_soft_margin = float(raw_tau_soft_margin)
        self.raw_tau_center = _compute_raw_tau_center(self.tau_target, self.tau_min, self.tau_max)
        self.concentration_margin = float(concentration_margin)

        if self.loss_mode not in {
            "weighted_ce",
            "fixed_cda",
            "learnable_cda",
            "adaptive_ucl_cda",
            "adaptive_ucl_cda_v2",
            "adaptive_ucl_cda_v3",
            "corn",
            "corn_adaptive_tau_safe",
            "corn_ordinal_multitask_v1",
        }:
            raise ValueError(f"Unsupported loss_mode='{self.loss_mode}'.")
        if self.loss_mode != "weighted_ce" and self.use_focal:
            raise ValueError("Focal loss is only supported for the weighted_ce baseline.")

        if self.use_focal:
            self.ce_loss = FocalLoss(
                class_weights=class_weights,
                gamma=self.focal_gamma,
                smoothing=self.label_smoothing,
            )
        else:
            self.ce_loss = WeightedCrossEntropyLossWrapper(
                class_weights=class_weights,
                smoothing=self.label_smoothing,
            )

        self.corn_loss: CORNLoss | None = None
        self.decoded_unimodality_regularizer = DecodedUnimodalityRegularizer(margin=self.decoded_unimodal_margin)

        if self.loss_mode == "fixed_cda":
            self.ordinal_loss: nn.Module | None = FixedCDALoss(
                num_classes=self.num_classes,
                alpha=self.fixed_cda_alpha,
            )
        elif self.loss_mode == "learnable_cda":
            self.ordinal_loss = LearnableOrdinalCDALoss(
                num_classes=self.num_classes,
                tau_init=tau_init,
            )
        elif self.loss_mode == "adaptive_ucl_cda":
            self.ordinal_loss = AdaptiveUCLCDALoss(
                num_classes=self.num_classes,
                tau_init=tau_init,
                tau_min=tau_min,
                tau_max=tau_max,
                concentration_margin=concentration_margin,
            )
        elif self.loss_mode == "adaptive_ucl_cda_v2":
            self.ordinal_loss = AdaptiveUCLCDALossV2(
                num_classes=self.num_classes,
                tau_init=tau_init,
                tau_min=tau_min,
                tau_max=tau_max,
                concentration_margin=concentration_margin,
                tau_target=tau_target,
                tau_variance_weight=tau_variance_weight,
                tau_std_floor=tau_std_floor,
            )
        elif self.loss_mode == "adaptive_ucl_cda_v3":
            self.ordinal_loss = AdaptiveUCLCDALossV3(
                num_classes=self.num_classes,
                tau_init=tau_init,
                tau_min=tau_min,
                tau_max=tau_max,
                concentration_margin=concentration_margin,
                tau_target=tau_target,
                tau_easy=tau_easy,
                tau_hard=tau_hard,
                tau_variance_weight=tau_variance_weight,
                tau_std_floor=tau_std_floor,
                tau_rank_margin_difficulty=tau_rank_margin_difficulty,
                tau_rank_margin_value=tau_rank_margin_value,
                raw_tau_soft_margin=raw_tau_soft_margin,
            )
        elif self.loss_mode == "corn":
            self.corn_loss = CORNLoss(num_classes=self.num_classes)
            self.ordinal_loss = self.corn_loss
        elif self.loss_mode in {"corn_adaptive_tau_safe", "corn_ordinal_multitask_v1"}:
            self.corn_loss = CORNLoss(num_classes=self.num_classes)
            self.ordinal_loss = AdaptiveUCLCDALossV3(
                num_classes=self.num_classes,
                tau_init=tau_init,
                tau_min=tau_min,
                tau_max=tau_max,
                concentration_margin=concentration_margin,
                tau_target=tau_target,
                tau_easy=tau_easy,
                tau_hard=tau_hard,
                tau_variance_weight=tau_variance_weight,
                tau_std_floor=tau_std_floor,
                tau_rank_margin_difficulty=tau_rank_margin_difficulty,
                tau_rank_margin_value=tau_rank_margin_value,
                raw_tau_soft_margin=raw_tau_soft_margin,
            )
        else:
            self.ordinal_loss = None

        self.register_buffer("canonical_positions", torch.linspace(0.0, 1.0, steps=self.num_classes))
        aux_soft_distribution = _prepare_soft_target_distribution(
            self.canonical_positions.new_tensor(
                aux_soft_label_target_distribution
                if aux_soft_label_target_distribution is not None
                else torch.eye(self.num_classes, dtype=torch.float32)
            ),
            num_classes=self.num_classes,
        )
        self.register_buffer("aux_soft_target_distribution", aux_soft_distribution)

        dist_target_distribution = _prepare_soft_target_distribution(
            self.canonical_positions.new_tensor(
                distribution_target_distribution
                if distribution_target_distribution is not None
                else torch.eye(self.num_classes, dtype=torch.float32)
            ),
            num_classes=self.num_classes,
        )
        self.register_buffer("distribution_target_distribution", dist_target_distribution)

        if distribution_class_weights is None:
            distribution_class_weights = torch.ones(self.num_classes, dtype=torch.float32)
        dist_class_weights = torch.as_tensor(distribution_class_weights, dtype=torch.float32)
        if dist_class_weights.numel() != self.num_classes:
            raise ValueError(
                f"distribution_class_weights must have {self.num_classes} values, got {tuple(dist_class_weights.shape)}."
            )
        self.register_buffer("distribution_class_weights", dist_class_weights.reshape(self.num_classes))

        if severity_regression_class_weights is None:
            severity_regression_class_weights = torch.ones(self.num_classes, dtype=torch.float32)
        reg_class_weights = torch.as_tensor(severity_regression_class_weights, dtype=torch.float32)
        if reg_class_weights.numel() != self.num_classes:
            raise ValueError(
                "severity_regression_class_weights must have "
                f"{self.num_classes} values, got {tuple(reg_class_weights.shape)}."
            )
        self.register_buffer("severity_regression_class_weights", reg_class_weights.reshape(self.num_classes))
        self.register_buffer(
            "severity_class_values",
            torch.arange(self.num_classes, dtype=torch.float32),
        )

    def _zero(self, reference: torch.Tensor) -> torch.Tensor:
        return reference.new_tensor(0.0)

    def _weighted_soft_cross_entropy(
        self,
        logits: torch.Tensor,
        soft_targets: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        log_probs = F.log_softmax(logits.float(), dim=1)
        per_sample = -(soft_targets.float() * log_probs).sum(dim=1)
        if sample_weights is not None:
            weights = sample_weights.to(device=logits.device, dtype=per_sample.dtype).reshape(-1)
            loss = (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)
        else:
            loss = per_sample.mean()
        return loss.to(device=logits.device, dtype=logits.dtype)

    def _weighted_smooth_l1(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        per_sample = F.smooth_l1_loss(
            predictions.float().reshape(-1),
            targets.float().reshape(-1),
            reduction="none",
        )
        if sample_weights is not None:
            weights = sample_weights.to(device=predictions.device, dtype=per_sample.dtype).reshape(-1)
            loss = (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)
        else:
            loss = per_sample.mean()
        return loss.to(device=predictions.device, dtype=predictions.dtype)

    def compute_gap_regularization(self) -> torch.Tensor:
        if not isinstance(self.ordinal_loss, LearnableSeverityAxis):
            return self._zero(self.canonical_positions)
        gaps = self.ordinal_loss._compute_gaps()
        return (gaps.mean() - 1.0).pow(2)

    def get_gap_parameters(self) -> list[nn.Parameter]:
        if isinstance(self.ordinal_loss, LearnableSeverityAxis):
            return self.ordinal_loss.get_gap_parameters()
        return []

    def get_non_gap_trainable_parameters(self) -> list[nn.Parameter]:
        gap_param_ids = {id(param) for param in self.get_gap_parameters()}
        return [param for param in self.parameters() if param.requires_grad and id(param) not in gap_param_ids]

    def _forward_corn_adaptive_tau_safe(
        self,
        corn_logits: torch.Tensor,
        targets: torch.Tensor,
        sample_tau: torch.Tensor | None,
        raw_tau: torch.Tensor | None,
        *,
        corn_soft_enabled: bool,
    ) -> dict[str, Any]:
        assert self.corn_loss is not None
        assert isinstance(self.ordinal_loss, AdaptiveUCLCDALossV3)
        if sample_tau is None:
            raise ValueError("corn_adaptive_tau_safe requires per-sample tau from the ambiguity head.")
        if raw_tau is None:
            raise ValueError("corn_adaptive_tau_safe requires raw_tau from the ambiguity head.")

        corn_outputs = self.corn_loss(corn_logits, targets)
        loss_corn_main = corn_outputs["loss_corn_main"]
        threshold_probabilities = corn_outputs["threshold_probabilities"]
        class_probabilities = corn_outputs["class_probabilities"]
        task_losses = corn_outputs["task_losses"]
        task_counts = corn_outputs["task_counts"]

        if sample_tau.ndim == 0:
            sample_tau = sample_tau.unsqueeze(0)
        if raw_tau.ndim == 0:
            raw_tau = raw_tau.unsqueeze(0)

        tau = sample_tau.to(device=corn_logits.device, dtype=corn_logits.dtype).clamp(self.tau_min, self.tau_max)
        raw_tau = raw_tau.to(device=corn_logits.device, dtype=corn_logits.dtype)
        if raw_tau.numel() != tau.numel():
            raise ValueError(
                "corn_adaptive_tau_safe expects raw_tau and sample_tau to have the same number of elements, "
                f"got raw_tau={tuple(raw_tau.shape)} and sample_tau={tuple(tau.shape)}."
            )

        tau_reg_outputs = self.ordinal_loss.tau_regularizer(tau)
        difficulty_values, gt_probability = compute_sample_difficulty_from_probabilities(class_probabilities, targets)
        difficulty_values = difficulty_values.to(device=corn_logits.device, dtype=corn_logits.dtype)
        gt_probability = gt_probability.to(device=corn_logits.device, dtype=corn_logits.dtype)
        tau_ref = build_tau_reference_from_difficulty(tau, difficulty_values, self.tau_easy, self.tau_hard)
        loss_tau_diff = F.smooth_l1_loss(tau.float(), tau_ref.float())
        loss_tau_rank = self.ordinal_loss.tau_rank_regularizer(tau, difficulty_values)

        raw_tau_ref = _tau_to_raw_tau_logit(tau_ref, self.tau_min, self.tau_max).to(
            device=corn_logits.device,
            dtype=corn_logits.dtype,
        )
        loss_raw_tau_diff = F.smooth_l1_loss(raw_tau.float(), raw_tau_ref.float())
        raw_tau_values = raw_tau.float().reshape(-1)
        raw_tau_center = raw_tau.new_tensor(self.raw_tau_center).float()
        raw_tau_mean = raw_tau_values.mean() if raw_tau_values.numel() > 0 else raw_tau_center
        loss_raw_tau_center = (raw_tau_mean - raw_tau_center).pow(2)
        raw_tau_soft_margin = raw_tau.new_tensor(self.raw_tau_soft_margin).float()
        raw_tau_margin_excess = F.relu((raw_tau_values - raw_tau_center).abs() - raw_tau_soft_margin)
        loss_raw_tau_bound = (
            raw_tau_margin_excess.pow(2).mean()
            if raw_tau_margin_excess.numel() > 0
            else raw_tau.new_tensor(0.0).float()
        )
        corr_tau_difficulty = _compute_pearson_correlation(tau, difficulty_values)
        corr_raw_tau_difficulty = _compute_pearson_correlation(raw_tau, difficulty_values)
        self.ordinal_loss.last_corr_raw_tau_difficulty.copy_(corr_raw_tau_difficulty.detach().float())

        gaps, soft_targets = self.ordinal_loss.build_soft_targets(
            targets,
            tau,
            device=corn_logits.device,
            dtype=corn_logits.dtype,
        )
        soft_target_entropy = -(soft_targets * soft_targets.clamp_min(1e-8).log()).sum(dim=1).mean()
        soft_target_gt_mass = soft_targets.gather(1, targets.unsqueeze(1)).mean()
        self.ordinal_loss._update_tau_stats(tau)
        self.ordinal_loss._update_difficulty_stats(tau, difficulty_values)

        detached_soft_targets = soft_targets.detach() if self.corn_soft_detach_target else soft_targets
        loss_corn_soft = _soft_cross_entropy_from_probabilities(class_probabilities, detached_soft_targets)
        loss_corn_soft_emd = compute_ordinal_cdf_distance(class_probabilities, detached_soft_targets)
        loss_corn_unimodal = self.decoded_unimodality_regularizer(class_probabilities)
        if not corn_soft_enabled:
            loss_corn_soft = self._zero(corn_logits)
            loss_corn_soft_emd = self._zero(corn_logits)
            loss_corn_unimodal = self._zero(corn_logits)
        elif not self.enable_corn_soft_emd:
            loss_corn_soft_emd = self._zero(corn_logits)
        if not self.enable_decoded_unimodal_reg:
            loss_corn_unimodal = self._zero(corn_logits)

        loss_gap_reg = self.compute_gap_regularization().to(device=corn_logits.device, dtype=corn_logits.dtype)
        loss_tau_reg = tau_reg_outputs["loss_tau_reg"].to(device=corn_logits.device, dtype=corn_logits.dtype)
        loss_tau_mean = tau_reg_outputs["loss_tau_mean"].to(device=corn_logits.device, dtype=corn_logits.dtype)
        loss_tau_var = tau_reg_outputs["loss_tau_var"].to(device=corn_logits.device, dtype=corn_logits.dtype)
        loss_tau_diff = loss_tau_diff.to(device=corn_logits.device, dtype=corn_logits.dtype)
        loss_tau_rank = loss_tau_rank.to(device=corn_logits.device, dtype=corn_logits.dtype)
        loss_raw_tau_diff = loss_raw_tau_diff.to(device=corn_logits.device, dtype=corn_logits.dtype)
        loss_raw_tau_center = loss_raw_tau_center.to(device=corn_logits.device, dtype=corn_logits.dtype)
        loss_raw_tau_bound = loss_raw_tau_bound.to(device=corn_logits.device, dtype=corn_logits.dtype)

        loss_without_soft = (
            loss_corn_main
            + self.lambda_tau_mean * loss_tau_reg
            + self.lambda_tau_diff * loss_tau_diff
            + self.lambda_tau_rank * loss_tau_rank
            + self.lambda_raw_tau_diff * loss_raw_tau_diff
            + self.lambda_raw_tau_center * loss_raw_tau_center
            + self.lambda_raw_tau_bound * loss_raw_tau_bound
            + self.lambda_gap_reg * loss_gap_reg
        )
        # Keep decoded soft losses head-only in safe mode so they do not dominate trunk learning.
        loss_safe_head_only = (
            (self.lambda_corn_soft * loss_corn_soft)
            + (self.lambda_corn_soft_emd * loss_corn_soft_emd)
            + (self.lambda_decoded_unimodal * loss_corn_unimodal)
        )
        total_loss = loss_without_soft + loss_safe_head_only

        return {
            "loss": total_loss,
            "loss_backward": loss_without_soft,
            "loss_safe_head_only": loss_safe_head_only,
            "loss_ce": self._zero(corn_logits),
            "loss_ord": loss_corn_main,
            "loss_corn_main": loss_corn_main,
            "loss_corn_soft": loss_corn_soft,
            "loss_corn_soft_emd": loss_corn_soft_emd,
            "loss_corn_unimodal": loss_corn_unimodal,
            "loss_gap_reg": loss_gap_reg,
            "loss_aux": self._zero(corn_logits),
            "loss_uni": self._zero(corn_logits),
            "loss_conc": self._zero(corn_logits),
            "loss_unimodal": self._zero(corn_logits),
            "loss_concentration": self._zero(corn_logits),
            "loss_tau_reg": loss_tau_reg,
            "loss_tau_mean": loss_tau_mean,
            "loss_tau_var": loss_tau_var,
            "loss_tau_diff": loss_tau_diff,
            "loss_tau_rank": loss_tau_rank,
            "loss_raw_tau_diff": loss_raw_tau_diff,
            "loss_raw_tau_center": loss_raw_tau_center,
            "loss_raw_tau_bound": loss_raw_tau_bound,
            "soft_targets": detached_soft_targets,
            "soft_target_stats": {
                "entropy_mean": soft_target_entropy,
                "gt_mass_mean": soft_target_gt_mass,
            },
            "positions": self.ordinal_loss.get_current_positions().to(device=corn_logits.device, dtype=corn_logits.dtype),
            "tau": tau,
            "tau_ref": tau_ref.to(device=corn_logits.device, dtype=corn_logits.dtype),
            "raw_tau_ref": raw_tau_ref,
            "tau_stats": self.ordinal_loss._current_tau_stats_tensor(device=corn_logits.device, dtype=corn_logits.dtype),
            "difficulty": difficulty_values,
            "difficulty_values": difficulty_values,
            "difficulty_stats": {
                key: value.to(device=corn_logits.device, dtype=corn_logits.dtype)
                for key, value in _compute_scalar_statistics(difficulty_values).items()
            },
            "corr_tau_difficulty": corr_tau_difficulty.to(device=corn_logits.device, dtype=corn_logits.dtype),
            "corr_raw_tau_difficulty": corr_raw_tau_difficulty.to(device=corn_logits.device, dtype=corn_logits.dtype),
            "threshold_probabilities": threshold_probabilities,
            "class_probabilities": class_probabilities,
            "corn_task_losses": task_losses,
            "corn_task_counts": task_counts,
            "corn_soft_enabled": bool(corn_soft_enabled),
            "corn_mode": "corn_adaptive_tau_safe",
            "gt_probability": gt_probability,
            "gaps": gaps,
            "adaptive_soft_target_detached": self.corn_soft_detach_target,
        }

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        aux_logits: torch.Tensor | None = None,
        distribution_logits: torch.Tensor | None = None,
        severity_score: torch.Tensor | None = None,
        sample_tau: torch.Tensor | None = None,
        raw_tau: torch.Tensor | None = None,
        corn_soft_enabled: bool | None = None,
    ) -> dict[str, Any]:
        positions = self.canonical_positions.to(device=logits.device, dtype=logits.dtype)
        loss_ce = self._zero(logits)
        loss_ord = self._zero(logits)
        loss_corn_main = self._zero(logits)
        loss_corn_soft = self._zero(logits)
        loss_corn_soft_emd = self._zero(logits)
        loss_corn_unimodal = self._zero(logits)
        loss_gap_reg = self._zero(logits)
        loss_aux = self._zero(logits)
        loss_aux_soft = self._zero(logits)
        loss_distribution = self._zero(logits)
        loss_severity_reg = self._zero(logits)
        loss_consistency_dist = self._zero(logits)
        loss_consistency_severity = self._zero(logits)
        loss_unimodal = self._zero(logits)
        loss_concentration = self._zero(logits)
        loss_tau_reg = self._zero(logits)
        loss_tau_mean = self._zero(logits)
        loss_tau_var = self._zero(logits)
        loss_tau_diff = self._zero(logits)
        loss_tau_rank = self._zero(logits)
        loss_raw_tau_diff = self._zero(logits)
        loss_raw_tau_center = self._zero(logits)
        loss_raw_tau_bound = self._zero(logits)
        soft_targets = None
        tau_tensor = None
        tau_stats = None
        soft_target_stats = None
        difficulty_stats = None
        corr_tau_difficulty = None
        corr_raw_tau_difficulty = None
        difficulty = None
        difficulty_values = None
        tau_ref = None
        raw_tau_ref = None
        threshold_probabilities = None
        class_probabilities = None
        corn_task_losses = None
        corn_task_counts = None
        corn_mode = self.loss_mode if self.loss_mode in {"corn", "corn_adaptive_tau_safe", "corn_ordinal_multitask_v1"} else None
        adaptive_soft_target_detached = None
        corn_soft_enabled_flag = bool(corn_soft_enabled) if corn_soft_enabled is not None else False
        loss_safe_head_only = self._zero(logits)
        distribution_probabilities = None
        distribution_expected_severity = None
        distribution_head_acc = None
        severity_target = None
        pred_severity_mean = None
        pred_severity_std = None
        total_loss: torch.Tensor

        if self.loss_mode == "corn":
            assert self.corn_loss is not None
            ordinal_outputs = self.corn_loss(logits, targets)
            loss_ord = ordinal_outputs["loss_ord"]
            loss_corn_main = ordinal_outputs["loss_corn_main"]
            threshold_probabilities = ordinal_outputs["threshold_probabilities"]
            class_probabilities = ordinal_outputs["class_probabilities"]
            positions = ordinal_outputs["positions"]
            corn_task_losses = ordinal_outputs["task_losses"]
            corn_task_counts = ordinal_outputs["task_counts"]
            total_loss = loss_ord
            loss_backward = total_loss
        elif self.loss_mode == "corn_adaptive_tau_safe":
            safe_outputs = self._forward_corn_adaptive_tau_safe(
                logits,
                targets,
                sample_tau=sample_tau,
                raw_tau=raw_tau,
                corn_soft_enabled=corn_soft_enabled_flag,
            )
            total_loss = safe_outputs["loss"]
            loss_backward = safe_outputs["loss_backward"]
            loss_ord = safe_outputs["loss_ord"]
            loss_corn_main = safe_outputs["loss_corn_main"]
            loss_corn_soft = safe_outputs["loss_corn_soft"]
            loss_corn_soft_emd = safe_outputs["loss_corn_soft_emd"]
            loss_corn_unimodal = safe_outputs["loss_corn_unimodal"]
            loss_gap_reg = safe_outputs["loss_gap_reg"]
            loss_aux = safe_outputs["loss_aux"]
            loss_unimodal = safe_outputs["loss_unimodal"]
            loss_concentration = safe_outputs["loss_concentration"]
            loss_tau_reg = safe_outputs["loss_tau_reg"]
            loss_tau_mean = safe_outputs["loss_tau_mean"]
            loss_tau_var = safe_outputs["loss_tau_var"]
            loss_tau_diff = safe_outputs["loss_tau_diff"]
            loss_tau_rank = safe_outputs["loss_tau_rank"]
            loss_raw_tau_diff = safe_outputs["loss_raw_tau_diff"]
            loss_raw_tau_center = safe_outputs["loss_raw_tau_center"]
            loss_raw_tau_bound = safe_outputs["loss_raw_tau_bound"]
            soft_targets = safe_outputs["soft_targets"]
            soft_target_stats = safe_outputs["soft_target_stats"]
            positions = safe_outputs["positions"]
            tau_tensor = safe_outputs["tau"]
            tau_ref = safe_outputs["tau_ref"]
            raw_tau_ref = safe_outputs["raw_tau_ref"]
            tau_stats = safe_outputs["tau_stats"]
            difficulty = safe_outputs["difficulty"]
            difficulty_values = safe_outputs["difficulty_values"]
            difficulty_stats = safe_outputs["difficulty_stats"]
            corr_tau_difficulty = safe_outputs["corr_tau_difficulty"]
            corr_raw_tau_difficulty = safe_outputs["corr_raw_tau_difficulty"]
            threshold_probabilities = safe_outputs["threshold_probabilities"]
            class_probabilities = safe_outputs["class_probabilities"]
            corn_task_losses = safe_outputs["corn_task_losses"]
            corn_task_counts = safe_outputs["corn_task_counts"]
            corn_soft_enabled_flag = bool(safe_outputs["corn_soft_enabled"])
            adaptive_soft_target_detached = safe_outputs["adaptive_soft_target_detached"]
            loss_safe_head_only = safe_outputs["loss_safe_head_only"]
        elif self.loss_mode == "corn_ordinal_multitask_v1":
            safe_outputs = self._forward_corn_adaptive_tau_safe(
                logits,
                targets,
                sample_tau=sample_tau,
                raw_tau=raw_tau,
                corn_soft_enabled=corn_soft_enabled_flag,
            )
            total_loss = safe_outputs["loss"]
            loss_backward = safe_outputs["loss_backward"]
            loss_ord = safe_outputs["loss_ord"]
            loss_corn_main = safe_outputs["loss_corn_main"]
            loss_corn_soft = safe_outputs["loss_corn_soft"]
            loss_corn_soft_emd = safe_outputs["loss_corn_soft_emd"]
            loss_corn_unimodal = safe_outputs["loss_corn_unimodal"]
            loss_gap_reg = safe_outputs["loss_gap_reg"]
            loss_aux = safe_outputs["loss_aux"]
            loss_unimodal = safe_outputs["loss_unimodal"]
            loss_concentration = safe_outputs["loss_concentration"]
            loss_tau_reg = safe_outputs["loss_tau_reg"]
            loss_tau_mean = safe_outputs["loss_tau_mean"]
            loss_tau_var = safe_outputs["loss_tau_var"]
            loss_tau_diff = safe_outputs["loss_tau_diff"]
            loss_tau_rank = safe_outputs["loss_tau_rank"]
            loss_raw_tau_diff = safe_outputs["loss_raw_tau_diff"]
            loss_raw_tau_center = safe_outputs["loss_raw_tau_center"]
            loss_raw_tau_bound = safe_outputs["loss_raw_tau_bound"]
            soft_targets = safe_outputs["soft_targets"]
            soft_target_stats = safe_outputs["soft_target_stats"]
            positions = safe_outputs["positions"]
            tau_tensor = safe_outputs["tau"]
            tau_ref = safe_outputs["tau_ref"]
            raw_tau_ref = safe_outputs["raw_tau_ref"]
            tau_stats = safe_outputs["tau_stats"]
            difficulty = safe_outputs["difficulty"]
            difficulty_values = safe_outputs["difficulty_values"]
            difficulty_stats = safe_outputs["difficulty_stats"]
            corr_tau_difficulty = safe_outputs["corr_tau_difficulty"]
            corr_raw_tau_difficulty = safe_outputs["corr_raw_tau_difficulty"]
            threshold_probabilities = safe_outputs["threshold_probabilities"]
            class_probabilities = safe_outputs["class_probabilities"]
            corn_task_losses = safe_outputs["corn_task_losses"]
            corn_task_counts = safe_outputs["corn_task_counts"]
            corn_soft_enabled_flag = bool(safe_outputs["corn_soft_enabled"])
            adaptive_soft_target_detached = safe_outputs["adaptive_soft_target_detached"]
            loss_safe_head_only = safe_outputs["loss_safe_head_only"]
            corn_mode = "corn_ordinal_multitask_v1"

            severity_positions = self.severity_class_values.to(device=logits.device, dtype=logits.dtype)
            sample_distribution_weights = self.distribution_class_weights.to(device=logits.device, dtype=logits.dtype)[targets]
            sample_regression_weights = self.severity_regression_class_weights.to(device=logits.device, dtype=logits.dtype)[targets]

            if distribution_logits is not None and self.distribution_head_enabled:
                distribution_target = build_batch_soft_targets(
                    targets,
                    self.distribution_target_distribution.to(device=logits.device, dtype=logits.dtype),
                )
                distribution_probabilities = torch.softmax(distribution_logits, dim=1)
                loss_distribution = self._weighted_soft_cross_entropy(
                    distribution_logits,
                    distribution_target,
                    sample_weights=sample_distribution_weights,
                )
                distribution_expected_severity = compute_expected_severity_from_probabilities(
                    distribution_probabilities,
                    severity_positions,
                )
                distribution_head_acc = (distribution_probabilities.argmax(dim=1) == targets).float().mean()
                total_loss = total_loss + (self.lambda_distribution * loss_distribution)
                loss_backward = loss_backward + (self.lambda_distribution * loss_distribution)

            if severity_score is not None and self.severity_regression_enabled:
                severity_target = severity_positions.index_select(0, targets)
                loss_severity_reg = self._weighted_smooth_l1(
                    severity_score,
                    severity_target,
                    sample_weights=sample_regression_weights,
                )
                severity_values = severity_score.detach().float().reshape(-1)
                pred_severity_mean = severity_values.mean()
                pred_severity_std = (
                    severity_values.std(unbiased=False)
                    if severity_values.numel() > 1
                    else severity_values.new_tensor(0.0)
                )
                total_loss = total_loss + (self.lambda_severity_reg * loss_severity_reg)
                loss_backward = loss_backward + (self.lambda_severity_reg * loss_severity_reg)

            if self.consistency_enabled:
                if distribution_probabilities is not None and class_probabilities is not None:
                    loss_consistency_dist = compute_symmetric_kl_from_probabilities(class_probabilities, distribution_probabilities)
                    total_loss = total_loss + (self.lambda_consistency_dist * loss_consistency_dist)
                    loss_backward = loss_backward + (self.lambda_consistency_dist * loss_consistency_dist)
                if severity_score is not None and class_probabilities is not None:
                    corn_expected_severity = compute_expected_severity_from_probabilities(class_probabilities, severity_positions)
                    loss_consistency_severity = F.smooth_l1_loss(
                        corn_expected_severity.float(),
                        severity_score.float().reshape(-1),
                    ).to(device=logits.device, dtype=logits.dtype)
                    total_loss = total_loss + (self.lambda_consistency_severity * loss_consistency_severity)
                    loss_backward = loss_backward + (self.lambda_consistency_severity * loss_consistency_severity)
        else:
            loss_ce = self.ce_loss(logits, targets)
            total_loss = loss_ce

            if self.loss_mode in {
                "fixed_cda",
                "learnable_cda",
                "adaptive_ucl_cda",
                "adaptive_ucl_cda_v2",
                "adaptive_ucl_cda_v3",
            }:
                assert self.ordinal_loss is not None
                if self.loss_mode in {"adaptive_ucl_cda", "adaptive_ucl_cda_v2", "adaptive_ucl_cda_v3"}:
                    assert isinstance(self.ordinal_loss, AdaptiveUCLCDALoss)
                    if self.loss_mode == "adaptive_ucl_cda_v3":
                        assert isinstance(self.ordinal_loss, AdaptiveUCLCDALossV3)
                        ordinal_outputs = self.ordinal_loss(logits, targets, sample_tau=sample_tau, raw_tau=raw_tau)
                    else:
                        ordinal_outputs = self.ordinal_loss(logits, targets, sample_tau=sample_tau)
                    loss_unimodal = ordinal_outputs["loss_unimodal"]
                    loss_concentration = ordinal_outputs["loss_concentration"]
                    loss_tau_reg = ordinal_outputs["loss_tau_reg"]
                    loss_tau_mean = ordinal_outputs.get("loss_tau_mean", self._zero(logits))
                    loss_tau_var = ordinal_outputs.get("loss_tau_var", self._zero(logits))
                    loss_tau_diff = ordinal_outputs.get("loss_tau_diff", self._zero(logits))
                    loss_tau_rank = ordinal_outputs.get("loss_tau_rank", self._zero(logits))
                    loss_raw_tau_diff = ordinal_outputs.get("loss_raw_tau_diff", self._zero(logits))
                    loss_raw_tau_center = ordinal_outputs.get("loss_raw_tau_center", self._zero(logits))
                    loss_raw_tau_bound = ordinal_outputs.get("loss_raw_tau_bound", self._zero(logits))
                    tau_stats = ordinal_outputs["tau_stats"]
                    soft_target_stats = ordinal_outputs.get("soft_target_stats")
                    difficulty_stats = ordinal_outputs.get("difficulty_stats")
                    corr_tau_difficulty = ordinal_outputs.get("corr_tau_difficulty")
                    corr_raw_tau_difficulty = ordinal_outputs.get("corr_raw_tau_difficulty")
                    difficulty = ordinal_outputs.get("difficulty")
                    difficulty_values = ordinal_outputs.get("difficulty_values")
                    tau_ref = ordinal_outputs.get("tau_ref")
                    raw_tau_ref = ordinal_outputs.get("raw_tau_ref")
                    total_loss = total_loss + self.lambda_uni * loss_unimodal + self.lambda_conc * loss_concentration
                    if self.loss_mode == "adaptive_ucl_cda_v2":
                        total_loss = total_loss + self.lambda_tau * loss_tau_reg
                    elif self.loss_mode == "adaptive_ucl_cda_v3":
                        total_loss = (
                            total_loss
                            + self.lambda_tau_mean * loss_tau_reg
                            + self.lambda_tau_diff * loss_tau_diff
                            + self.lambda_tau_rank * loss_tau_rank
                            + self.lambda_raw_tau_diff * loss_raw_tau_diff
                            + self.lambda_raw_tau_center * loss_raw_tau_center
                            + self.lambda_raw_tau_bound * loss_raw_tau_bound
                        )
                else:
                    ordinal_outputs = self.ordinal_loss(logits, targets)

                loss_ord = ordinal_outputs["loss_ord"]
                soft_targets = ordinal_outputs["soft_targets"]
                positions = ordinal_outputs["positions"]
                tau_tensor = ordinal_outputs.get("tau")
                total_loss = total_loss + self.lambda_ord * loss_ord

                if isinstance(self.ordinal_loss, LearnableSeverityAxis):
                    loss_gap_reg = self.compute_gap_regularization().to(device=logits.device, dtype=logits.dtype)
                    total_loss = total_loss + self.lambda_gap_reg * loss_gap_reg
            loss_backward = total_loss

        if aux_logits is not None and self.aux_soft_label_enabled and self.aux_soft_label_weight > 0.0:
            aux_soft_targets = build_batch_soft_targets(
                targets,
                self.aux_soft_target_distribution.to(device=logits.device, dtype=logits.dtype),
            )
            loss_aux_soft = _soft_cross_entropy(aux_logits, aux_soft_targets)
            total_loss = total_loss + self.aux_soft_label_weight * loss_aux_soft
            loss_backward = loss_backward + self.aux_soft_label_weight * loss_aux_soft

        return {
            "loss": total_loss,
            "loss_backward": loss_backward,
            "loss_ce": loss_ce,
            "loss_ord": loss_ord,
            "loss_corn_main": loss_corn_main,
            "loss_corn_soft": loss_corn_soft,
            "loss_corn_soft_emd": loss_corn_soft_emd,
            "loss_corn_unimodal": loss_corn_unimodal,
            "loss_gap_reg": loss_gap_reg,
            "loss_aux": loss_aux,
            "loss_aux_soft": loss_aux_soft,
            "loss_distribution": loss_distribution,
            "loss_severity_reg": loss_severity_reg,
            "loss_consistency_dist": loss_consistency_dist,
            "loss_consistency_severity": loss_consistency_severity,
            "loss_uni": loss_unimodal,
            "loss_conc": loss_concentration,
            "loss_unimodal": loss_unimodal,
            "loss_concentration": loss_concentration,
            "loss_tau_reg": loss_tau_reg,
            "loss_tau_mean": loss_tau_mean,
            "loss_tau_var": loss_tau_var,
            "loss_tau_diff": loss_tau_diff,
            "loss_tau_rank": loss_tau_rank,
            "loss_raw_tau_diff": loss_raw_tau_diff,
            "loss_raw_tau_center": loss_raw_tau_center,
            "loss_raw_tau_bound": loss_raw_tau_bound,
            "soft_targets": soft_targets,
            "soft_target_stats": soft_target_stats,
            "positions": positions,
            "tau": tau_tensor,
            "tau_ref": tau_ref,
            "raw_tau_ref": raw_tau_ref,
            "tau_stats": tau_stats,
            "difficulty": difficulty,
            "difficulty_values": difficulty_values,
            "difficulty_stats": difficulty_stats,
            "corr_tau_difficulty": corr_tau_difficulty,
            "corr_raw_tau_difficulty": corr_raw_tau_difficulty,
            "threshold_probabilities": threshold_probabilities,
            "class_probabilities": class_probabilities,
            "corn_task_losses": corn_task_losses,
            "corn_task_counts": corn_task_counts,
            "corn_soft_enabled": corn_soft_enabled_flag,
            "corn_mode": corn_mode,
            "adaptive_soft_target_detached": adaptive_soft_target_detached,
            "loss_safe_head_only": loss_safe_head_only,
            "distribution_probabilities": distribution_probabilities,
            "distribution_expected_severity": distribution_expected_severity,
            "distribution_head_acc": distribution_head_acc,
            "severity_target": severity_target,
            "pred_severity_mean": pred_severity_mean,
            "pred_severity_std": pred_severity_std,
        }

    def get_current_positions(self) -> torch.Tensor:
        if self.ordinal_loss is None:
            return self.canonical_positions.detach().clone()
        if isinstance(self.ordinal_loss, FixedCDALoss):
            return self.ordinal_loss.get_current_positions()
        if isinstance(self.ordinal_loss, LearnableSeverityAxis):
            return self.ordinal_loss.get_current_positions()
        return self.canonical_positions.detach().clone()

    def get_current_soft_target_matrix(self) -> torch.Tensor:
        if self.ordinal_loss is None or isinstance(self.ordinal_loss, CORNLoss):
            return torch.eye(
                self.num_classes,
                device=self.canonical_positions.device,
                dtype=self.canonical_positions.dtype,
            )
        if isinstance(self.ordinal_loss, FixedCDALoss):
            return self.ordinal_loss.get_current_soft_target_matrix()
        return self.ordinal_loss.get_current_soft_target_matrix()

    def get_current_tau(self) -> torch.Tensor | None:
        if isinstance(self.ordinal_loss, LearnableOrdinalCDALoss):
            return self.ordinal_loss.get_current_tau()
        if isinstance(self.ordinal_loss, AdaptiveUCLCDALoss):
            return self.ordinal_loss.get_current_tau()
        return None

    def get_current_gaps(self) -> torch.Tensor:
        if isinstance(self.ordinal_loss, LearnableSeverityAxis):
            return self.ordinal_loss.get_current_gaps()
        return torch.ones(3, device=self.canonical_positions.device, dtype=self.canonical_positions.dtype)

    def get_tau_statistics(self) -> dict[str, float] | None:
        if isinstance(self.ordinal_loss, AdaptiveUCLCDALoss):
            return self.ordinal_loss.get_last_tau_statistics()
        tau = self.get_current_tau()
        if tau is None:
            return None
        tau_value = float(tau.detach().cpu().item())
        return {
            "mean": tau_value,
            "std": 0.0,
            "min": tau_value,
            "max": tau_value,
            "p10": tau_value,
            "p50": tau_value,
            "p90": tau_value,
        }

    def get_difficulty_statistics(self) -> dict[str, float] | None:
        if isinstance(self.ordinal_loss, AdaptiveUCLCDALoss):
            return self.ordinal_loss.get_last_difficulty_statistics()
        return None

    def get_tau_difficulty_correlation(self) -> float | None:
        if isinstance(self.ordinal_loss, AdaptiveUCLCDALoss):
            return self.ordinal_loss.get_last_tau_difficulty_correlation()
        return None

    def get_raw_tau_difficulty_correlation(self) -> float | None:
        if isinstance(self.ordinal_loss, AdaptiveUCLCDALossV3):
            return self.ordinal_loss.get_last_raw_tau_difficulty_correlation()
        return None

    def export_state(self, class_names: list[str] | None = None) -> dict[str, Any]:
        positions = self.get_current_positions().detach().cpu().tolist()
        soft_target_matrix = self.get_current_soft_target_matrix().detach().cpu().tolist()
        gaps = self.get_current_gaps().detach().cpu().tolist()
        tau = self.get_current_tau()
        tau_value = None if tau is None else float(tau.detach().cpu().item())

        state: dict[str, Any] = {
            "loss_mode": self.loss_mode,
            "lambda_ord": self.lambda_ord,
            "lambda_uni": self.lambda_uni,
            "lambda_conc": self.lambda_conc,
            "lambda_gap_reg": self.lambda_gap_reg,
            "lambda_tau": self.lambda_tau,
            "lambda_tau_mean": self.lambda_tau_mean,
            "lambda_tau_diff": self.lambda_tau_diff,
            "lambda_tau_rank": self.lambda_tau_rank,
            "lambda_raw_tau_diff": self.lambda_raw_tau_diff,
            "lambda_raw_tau_center": self.lambda_raw_tau_center,
            "lambda_raw_tau_bound": self.lambda_raw_tau_bound,
            "lambda_corn_soft": self.lambda_corn_soft,
            "enable_corn_soft_emd": self.enable_corn_soft_emd,
            "lambda_corn_soft_emd": self.lambda_corn_soft_emd,
            "enable_decoded_unimodal_reg": self.enable_decoded_unimodal_reg,
            "lambda_decoded_unimodal": self.lambda_decoded_unimodal,
            "decoded_unimodal_margin": self.decoded_unimodal_margin,
            "lambda_aux": self.lambda_aux,
            "aux_soft_label_enabled": self.aux_soft_label_enabled,
            "aux_soft_label_weight": self.aux_soft_label_weight,
            "aux_soft_label_target_distribution": self.aux_soft_target_distribution.detach().cpu().tolist(),
            "lambda_distribution": self.lambda_distribution,
            "lambda_severity_reg": self.lambda_severity_reg,
            "lambda_consistency_dist": self.lambda_consistency_dist,
            "lambda_consistency_severity": self.lambda_consistency_severity,
            "distribution_head_enabled": self.distribution_head_enabled,
            "distribution_target_distribution": self.distribution_target_distribution.detach().cpu().tolist(),
            "distribution_class_weights": self.distribution_class_weights.detach().cpu().tolist(),
            "severity_regression_enabled": self.severity_regression_enabled,
            "severity_regression_class_weights": self.severity_regression_class_weights.detach().cpu().tolist(),
            "severity_regression_loss": self.severity_regression_loss,
            "consistency_enabled": self.consistency_enabled,
            "fixed_cda_alpha": self.fixed_cda_alpha if self.loss_mode == "fixed_cda" else None,
            "label_smoothing": self.label_smoothing,
            "use_focal": self.use_focal,
            "focal_gamma": self.focal_gamma if self.use_focal else None,
            "positions": positions,
            "soft_target_matrix": soft_target_matrix,
            "tau": tau_value,
            "tau_parameterization": self.tau_parameterization,
            "tau_logit_scale": self.tau_logit_scale,
            "tau_bounds": {
                "tau_min": self.tau_min,
                "tau_max": self.tau_max,
                "tau_base": self.tau_base,
                "delta_scale": self.delta_scale,
            },
            "tau_statistics": self.get_tau_statistics(),
            "difficulty_statistics": self.get_difficulty_statistics(),
            "corr_tau_difficulty": self.get_tau_difficulty_correlation(),
            "corr_raw_tau_difficulty": self.get_raw_tau_difficulty_correlation(),
            "tau_regularizer": {
                "tau_target": self.tau_target,
                "tau_variance_weight": self.tau_variance_weight,
                "tau_std_floor": self.tau_std_floor,
            } if self.loss_mode == "adaptive_ucl_cda_v2" else (
                {
                    "tau_target": self.tau_target,
                    "tau_easy": self.tau_easy,
                    "tau_hard": self.tau_hard,
                    "tau_variance_weight": self.tau_variance_weight,
                    "tau_std_floor": self.tau_std_floor,
                    "lambda_tau_mean": self.lambda_tau_mean,
                    "lambda_tau_diff": self.lambda_tau_diff,
                    "lambda_tau_rank": self.lambda_tau_rank,
                    "tau_rank_margin_difficulty": self.tau_rank_margin_difficulty,
                    "tau_rank_margin_value": self.tau_rank_margin_value,
                    "raw_tau_center": self.raw_tau_center,
                    "raw_tau_soft_margin": self.raw_tau_soft_margin,
                    "lambda_raw_tau_diff": self.lambda_raw_tau_diff,
                    "lambda_raw_tau_center": self.lambda_raw_tau_center,
                    "lambda_raw_tau_bound": self.lambda_raw_tau_bound,
                    "lambda_corn_soft": self.lambda_corn_soft if self.loss_mode in {"corn_adaptive_tau_safe", "corn_ordinal_multitask_v1"} else None,
                    "enable_corn_soft_emd": self.enable_corn_soft_emd if self.loss_mode in {"corn_adaptive_tau_safe", "corn_ordinal_multitask_v1"} else None,
                    "lambda_corn_soft_emd": self.lambda_corn_soft_emd if self.loss_mode in {"corn_adaptive_tau_safe", "corn_ordinal_multitask_v1"} else None,
                    "enable_decoded_unimodal_reg": self.enable_decoded_unimodal_reg if self.loss_mode in {"corn_adaptive_tau_safe", "corn_ordinal_multitask_v1"} else None,
                    "lambda_decoded_unimodal": self.lambda_decoded_unimodal if self.loss_mode in {"corn_adaptive_tau_safe", "corn_ordinal_multitask_v1"} else None,
                    "decoded_unimodal_margin": self.decoded_unimodal_margin if self.loss_mode in {"corn_adaptive_tau_safe", "corn_ordinal_multitask_v1"} else None,
                    "corn_soft_target_detached": self.corn_soft_detach_target if self.loss_mode in {"corn_adaptive_tau_safe", "corn_ordinal_multitask_v1"} else None,
                } if self.loss_mode in {"adaptive_ucl_cda_v3", "corn_adaptive_tau_safe", "corn_ordinal_multitask_v1"} else None
            ),
            "concentration_margin": self.concentration_margin,
            "gaps": {
                "gap_01": float(gaps[0]),
                "gap_12": float(gaps[1]),
                "gap_23": float(gaps[2]),
            },
        }
        if class_names is not None:
            state["positions_by_class"] = {name: float(pos) for name, pos in zip(class_names, positions)}
        return state


def build_loss_function(
    class_weights: torch.Tensor | None,
    loss_mode: str = "learnable_cda",
    label_smoothing: float = 0.05,
    lambda_ord: float = 0.3,
    lambda_uni: float = 0.10,
    lambda_conc: float = 0.05,
    lambda_gap_reg: float = 1e-3,
    lambda_tau: float = 0.01,
    lambda_tau_mean: float = 0.05,
    lambda_tau_diff: float = 0.20,
    lambda_tau_rank: float = 0.05,
    lambda_raw_tau_diff: float = 0.10,
    lambda_raw_tau_center: float = 0.02,
    lambda_raw_tau_bound: float = 0.02,
    lambda_corn_soft: float = 0.03,
    enable_corn_soft_emd: bool = False,
    lambda_corn_soft_emd: float = 0.02,
    enable_decoded_unimodal_reg: bool = False,
    lambda_decoded_unimodal: float = 0.01,
    decoded_unimodal_margin: float = 0.0,
    fixed_cda_alpha: float = 0.3,
    tau_init: float = 0.35,
    tau_min: float = 0.05,
    tau_max: float = 1.0,
    tau_base: float = 0.22,
    delta_scale: float = 0.12,
    tau_parameterization: str = "sigmoid",
    tau_logit_scale: float = 2.0,
    tau_target: float = 0.22,
    tau_easy: float = 0.16,
    tau_hard: float = 0.32,
    tau_variance_weight: float = 1e-2,
    tau_std_floor: float = 0.03,
    tau_rank_margin_difficulty: float = 0.10,
    tau_rank_margin_value: float = 0.01,
    raw_tau_soft_margin: float = 1.5,
    concentration_margin: float = 0.05,
    lambda_aux: float = 0.2,
    aux_soft_label_enabled: bool = False,
    aux_soft_label_weight: float = 0.0,
    aux_soft_label_target_distribution: torch.Tensor | list[list[float]] | tuple[tuple[float, ...], ...] | None = None,
    lambda_distribution: float = 0.18,
    lambda_severity_reg: float = 0.08,
    lambda_consistency_dist: float = 0.03,
    lambda_consistency_severity: float = 0.03,
    distribution_head_enabled: bool = False,
    distribution_target_distribution: torch.Tensor | list[list[float]] | tuple[tuple[float, ...], ...] | None = None,
    distribution_class_weights: torch.Tensor | list[float] | tuple[float, ...] | None = None,
    severity_regression_enabled: bool = False,
    severity_regression_class_weights: torch.Tensor | list[float] | tuple[float, ...] | None = None,
    severity_regression_loss: str = "smooth_l1",
    consistency_enabled: bool = False,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
    num_classes: int = 4,
) -> DamageLossModule:
    return DamageLossModule(
        loss_mode=loss_mode,
        class_weights=class_weights,
        num_classes=num_classes,
        label_smoothing=label_smoothing,
        lambda_ord=lambda_ord,
        lambda_uni=lambda_uni,
        lambda_conc=lambda_conc,
        lambda_gap_reg=lambda_gap_reg,
        lambda_tau=lambda_tau,
        lambda_tau_mean=lambda_tau_mean,
        lambda_tau_diff=lambda_tau_diff,
        lambda_tau_rank=lambda_tau_rank,
        lambda_raw_tau_diff=lambda_raw_tau_diff,
        lambda_raw_tau_center=lambda_raw_tau_center,
        lambda_raw_tau_bound=lambda_raw_tau_bound,
        lambda_corn_soft=lambda_corn_soft,
        enable_corn_soft_emd=enable_corn_soft_emd,
        lambda_corn_soft_emd=lambda_corn_soft_emd,
        enable_decoded_unimodal_reg=enable_decoded_unimodal_reg,
        lambda_decoded_unimodal=lambda_decoded_unimodal,
        decoded_unimodal_margin=decoded_unimodal_margin,
        fixed_cda_alpha=fixed_cda_alpha,
        tau_init=tau_init,
        tau_min=tau_min,
        tau_max=tau_max,
        tau_base=tau_base,
        delta_scale=delta_scale,
        tau_parameterization=tau_parameterization,
        tau_logit_scale=tau_logit_scale,
        tau_target=tau_target,
        tau_easy=tau_easy,
        tau_hard=tau_hard,
        tau_variance_weight=tau_variance_weight,
        tau_std_floor=tau_std_floor,
        tau_rank_margin_difficulty=tau_rank_margin_difficulty,
        tau_rank_margin_value=tau_rank_margin_value,
        raw_tau_soft_margin=raw_tau_soft_margin,
        concentration_margin=concentration_margin,
        lambda_aux=lambda_aux,
        aux_soft_label_enabled=aux_soft_label_enabled,
        aux_soft_label_weight=aux_soft_label_weight,
        aux_soft_label_target_distribution=aux_soft_label_target_distribution,
        lambda_distribution=lambda_distribution,
        lambda_severity_reg=lambda_severity_reg,
        lambda_consistency_dist=lambda_consistency_dist,
        lambda_consistency_severity=lambda_consistency_severity,
        distribution_head_enabled=distribution_head_enabled,
        distribution_target_distribution=distribution_target_distribution,
        distribution_class_weights=distribution_class_weights,
        severity_regression_enabled=severity_regression_enabled,
        severity_regression_class_weights=severity_regression_class_weights,
        severity_regression_loss=severity_regression_loss,
        consistency_enabled=consistency_enabled,
        use_focal=use_focal,
        focal_gamma=focal_gamma,
    )


def build_loss(
    class_weights: torch.Tensor | None,
    label_smoothing: float = 0.05,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
) -> DamageLossModule:
    return build_loss_function(
        class_weights=class_weights,
        loss_mode="weighted_ce",
        label_smoothing=label_smoothing,
        use_focal=use_focal,
        focal_gamma=focal_gamma,
    )


def compute_class_weights(class_counts: list[int] | tuple[int, ...]) -> torch.Tensor:
    counts = torch.tensor(class_counts, dtype=torch.float32)
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return weights
