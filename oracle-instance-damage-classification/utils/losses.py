from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-6)
    return math.log(math.expm1(value))


def _inverse_sigmoid(value: float) -> float:
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


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


def compute_sample_difficulty(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.softmax(logits.detach().float(), dim=1)
    gt_probabilities = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
    difficulty = 1.0 - gt_probabilities
    return difficulty.to(device=logits.device, dtype=logits.dtype), gt_probabilities.to(
        device=logits.device,
        dtype=logits.dtype,
    )


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
        tau_ref = tau.new_tensor(self.tau_easy) + (tau.new_tensor(self.tau_hard - self.tau_easy) * difficulty)
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
        difficulty_values: torch.Tensor | None = None,
        tau_ref: torch.Tensor | None = None,
        loss_tau_reg: torch.Tensor | None = None,
        loss_tau_mean: torch.Tensor | None = None,
        loss_tau_diff: torch.Tensor | None = None,
        loss_tau_var: torch.Tensor | None = None,
        loss_tau_rank: torch.Tensor | None = None,
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
        if loss_tau_reg is None:
            resolved_loss_tau_reg = resolved_loss_tau_mean + resolved_loss_tau_var
        else:
            resolved_loss_tau_reg = loss_tau_reg.to(device=logits.device, dtype=logits.dtype)

        return {
            "loss_ord": _soft_cross_entropy(logits, soft_targets),
            "loss_unimodal": loss_unimodal,
            "loss_concentration": loss_concentration,
            "loss_tau_reg": resolved_loss_tau_reg,
            "loss_tau_mean": resolved_loss_tau_mean,
            "loss_tau_var": resolved_loss_tau_var,
            "loss_tau_diff": resolved_loss_tau_diff,
            "loss_tau_rank": resolved_loss_tau_rank,
            "soft_targets": soft_targets,
            "soft_target_stats": {
                "entropy_mean": soft_target_entropy,
                "gt_mass_mean": soft_target_gt_mass,
            },
            "positions": positions,
            "tau": tau,
            "tau_ref": tau_ref,
            "gaps": gaps,
            "tau_stats": self._current_tau_stats_tensor(device=logits.device, dtype=logits.dtype),
            "difficulty": difficulty_values,
            "difficulty_values": difficulty_values,
            "difficulty_stats": self._current_difficulty_stats_tensor(device=logits.device, dtype=logits.dtype),
            "corr_tau_difficulty": self.last_corr_tau_difficulty.to(device=logits.device, dtype=logits.dtype),
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
        self.tau_rank_regularizer = PairwiseTauRankRegularizer(
            margin_difficulty=tau_rank_margin_difficulty,
            margin_value=tau_rank_margin_value,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_tau: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if sample_tau is None:
            raise ValueError("AdaptiveUCLCDALossV3 requires per-sample tau from the ambiguity head.")

        if sample_tau.ndim == 0:
            sample_tau = sample_tau.unsqueeze(0)
        tau = sample_tau.to(device=logits.device, dtype=logits.dtype).clamp_min(1e-6)

        tau_reg_outputs = self.tau_regularizer(tau)
        tau_diff_outputs = self.tau_diff_regularizer(tau, logits, targets)
        loss_tau_rank = self.tau_rank_regularizer(tau, tau_diff_outputs["difficulty"])

        outputs = self._compute_outputs(
            logits,
            targets,
            tau,
            difficulty_values=tau_diff_outputs["difficulty"],
            tau_ref=tau_diff_outputs["tau_ref"],
            loss_tau_reg=tau_reg_outputs["loss_tau_reg"],
            loss_tau_mean=tau_reg_outputs["loss_tau_mean"],
            loss_tau_var=tau_reg_outputs["loss_tau_var"],
            loss_tau_diff=tau_diff_outputs["loss_tau_diff"],
            loss_tau_rank=loss_tau_rank,
        )
        outputs["difficulty_stats"] = tau_diff_outputs["difficulty_stats"]
        outputs["corr_tau_difficulty"] = tau_diff_outputs["corr_tau_difficulty"]
        outputs["gt_probability"] = tau_diff_outputs["gt_probability"]
        outputs["difficulty"] = tau_diff_outputs["difficulty"]
        return outputs


class CORNLoss(nn.Module):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        if int(num_classes) < 2:
            raise ValueError("CORNLoss expects at least 2 ordered classes.")
        self.num_classes = int(num_classes)

    @staticmethod
    def logits_to_threshold_probabilities(logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)

    @classmethod
    def threshold_probabilities_to_class_probabilities(cls, threshold_probabilities: torch.Tensor) -> torch.Tensor:
        batch_size, num_thresholds = threshold_probabilities.shape
        num_classes = num_thresholds + 1
        conditional = threshold_probabilities.clamp(1e-6, 1.0 - 1e-6)
        survival = torch.ones(
            batch_size,
            num_classes,
            device=threshold_probabilities.device,
            dtype=threshold_probabilities.dtype,
        )
        survival[:, 1:] = torch.cumprod(conditional, dim=1)
        class_probabilities = torch.zeros_like(survival)
        class_probabilities[:, :-1] = survival[:, :-1] - survival[:, 1:]
        class_probabilities[:, -1] = survival[:, -1]
        return class_probabilities.clamp_min(1e-8)

    @classmethod
    def logits_to_class_probabilities(cls, logits: torch.Tensor) -> torch.Tensor:
        return cls.threshold_probabilities_to_class_probabilities(cls.logits_to_threshold_probabilities(logits))

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
        per_task_losses: list[torch.Tensor] = []

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
            num_examples += int(binary_targets.numel())
            per_task_losses.append(task_loss / max(int(binary_targets.numel()), 1))

        threshold_probabilities = self.logits_to_threshold_probabilities(logits)
        class_probabilities = self.logits_to_class_probabilities(logits)
        normalized_loss = total_loss / max(num_examples, 1)
        return {
            "loss_ord": normalized_loss,
            "positions": torch.linspace(0.0, 1.0, steps=self.num_classes, device=logits.device, dtype=logits.dtype),
            "threshold_probabilities": threshold_probabilities,
            "class_probabilities": class_probabilities,
            "task_losses": torch.stack(per_task_losses) if per_task_losses else logits.new_zeros(0),
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
        fixed_cda_alpha: float = 0.3,
        tau_init: float = 0.35,
        tau_min: float = 0.05,
        tau_max: float = 1.0,
        tau_base: float = 0.22,
        delta_scale: float = 0.12,
        tau_target: float = 0.22,
        tau_easy: float = 0.16,
        tau_hard: float = 0.32,
        tau_variance_weight: float = 1e-2,
        tau_std_floor: float = 0.03,
        tau_rank_margin_difficulty: float = 0.10,
        tau_rank_margin_value: float = 0.01,
        concentration_margin: float = 0.05,
        lambda_aux: float = 0.2,
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
        self.fixed_cda_alpha = float(fixed_cda_alpha)
        self.lambda_aux = float(lambda_aux)
        self.use_focal = bool(use_focal)
        self.focal_gamma = float(focal_gamma)
        self.label_smoothing = float(label_smoothing)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.tau_base = float(tau_base)
        self.delta_scale = float(delta_scale)
        self.tau_target = float(tau_target)
        self.tau_easy = float(tau_easy)
        self.tau_hard = float(tau_hard)
        self.tau_variance_weight = float(tau_variance_weight)
        self.tau_std_floor = float(tau_std_floor)
        self.tau_rank_margin_difficulty = float(tau_rank_margin_difficulty)
        self.tau_rank_margin_value = float(tau_rank_margin_value)
        self.concentration_margin = float(concentration_margin)

        if self.loss_mode not in {
            "weighted_ce",
            "fixed_cda",
            "learnable_cda",
            "adaptive_ucl_cda",
            "adaptive_ucl_cda_v2",
            "adaptive_ucl_cda_v3",
            "corn",
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
            )
        elif self.loss_mode == "corn":
            self.ordinal_loss = CORNLoss(num_classes=self.num_classes)
        else:
            self.ordinal_loss = None

        self.register_buffer("canonical_positions", torch.linspace(0.0, 1.0, steps=self.num_classes))

    def _zero(self, reference: torch.Tensor) -> torch.Tensor:
        return reference.new_tensor(0.0)

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

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        aux_logits: torch.Tensor | None = None,
        sample_tau: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        positions = self.canonical_positions.to(device=logits.device, dtype=logits.dtype)
        loss_ce = self._zero(logits)
        loss_ord = self._zero(logits)
        loss_gap_reg = self._zero(logits)
        loss_aux = self._zero(logits)
        loss_unimodal = self._zero(logits)
        loss_concentration = self._zero(logits)
        loss_tau_reg = self._zero(logits)
        loss_tau_mean = self._zero(logits)
        loss_tau_var = self._zero(logits)
        loss_tau_diff = self._zero(logits)
        loss_tau_rank = self._zero(logits)
        soft_targets = None
        tau_tensor = None
        tau_stats = None
        soft_target_stats = None
        difficulty_stats = None
        corr_tau_difficulty = None
        difficulty = None
        difficulty_values = None
        tau_ref = None
        threshold_probabilities = None
        class_probabilities = None

        if self.loss_mode == "corn":
            assert isinstance(self.ordinal_loss, CORNLoss)
            ordinal_outputs = self.ordinal_loss(logits, targets)
            loss_ord = ordinal_outputs["loss_ord"]
            threshold_probabilities = ordinal_outputs["threshold_probabilities"]
            class_probabilities = ordinal_outputs["class_probabilities"]
            positions = ordinal_outputs["positions"]
            total_loss = loss_ord
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
                    ordinal_outputs = self.ordinal_loss(logits, targets, sample_tau=sample_tau)
                    loss_unimodal = ordinal_outputs["loss_unimodal"]
                    loss_concentration = ordinal_outputs["loss_concentration"]
                    loss_tau_reg = ordinal_outputs["loss_tau_reg"]
                    loss_tau_mean = ordinal_outputs.get("loss_tau_mean", self._zero(logits))
                    loss_tau_var = ordinal_outputs.get("loss_tau_var", self._zero(logits))
                    loss_tau_diff = ordinal_outputs.get("loss_tau_diff", self._zero(logits))
                    loss_tau_rank = ordinal_outputs.get("loss_tau_rank", self._zero(logits))
                    tau_stats = ordinal_outputs["tau_stats"]
                    soft_target_stats = ordinal_outputs.get("soft_target_stats")
                    difficulty_stats = ordinal_outputs.get("difficulty_stats")
                    corr_tau_difficulty = ordinal_outputs.get("corr_tau_difficulty")
                    difficulty = ordinal_outputs.get("difficulty")
                    difficulty_values = ordinal_outputs.get("difficulty_values")
                    tau_ref = ordinal_outputs.get("tau_ref")
                    total_loss = total_loss + self.lambda_uni * loss_unimodal + self.lambda_conc * loss_concentration
                    if self.loss_mode == "adaptive_ucl_cda_v2":
                        total_loss = total_loss + self.lambda_tau * loss_tau_reg
                    elif self.loss_mode == "adaptive_ucl_cda_v3":
                        total_loss = (
                            total_loss
                            + self.lambda_tau_mean * loss_tau_reg
                            + self.lambda_tau_diff * loss_tau_diff
                            + self.lambda_tau_rank * loss_tau_rank
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

        if aux_logits is not None:
            loss_aux = self.ce_loss(aux_logits, targets)
            total_loss = total_loss + self.lambda_aux * loss_aux

        return {
            "loss": total_loss,
            "loss_ce": loss_ce,
            "loss_ord": loss_ord,
            "loss_gap_reg": loss_gap_reg,
            "loss_aux": loss_aux,
            "loss_uni": loss_unimodal,
            "loss_conc": loss_concentration,
            "loss_unimodal": loss_unimodal,
            "loss_concentration": loss_concentration,
            "loss_tau_reg": loss_tau_reg,
            "loss_tau_mean": loss_tau_mean,
            "loss_tau_var": loss_tau_var,
            "loss_tau_diff": loss_tau_diff,
            "loss_tau_rank": loss_tau_rank,
            "soft_targets": soft_targets,
            "soft_target_stats": soft_target_stats,
            "positions": positions,
            "tau": tau_tensor,
            "tau_ref": tau_ref,
            "tau_stats": tau_stats,
            "difficulty": difficulty,
            "difficulty_values": difficulty_values,
            "difficulty_stats": difficulty_stats,
            "corr_tau_difficulty": corr_tau_difficulty,
            "threshold_probabilities": threshold_probabilities,
            "class_probabilities": class_probabilities,
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
            "lambda_aux": self.lambda_aux,
            "fixed_cda_alpha": self.fixed_cda_alpha if self.loss_mode == "fixed_cda" else None,
            "label_smoothing": self.label_smoothing,
            "use_focal": self.use_focal,
            "focal_gamma": self.focal_gamma if self.use_focal else None,
            "positions": positions,
            "soft_target_matrix": soft_target_matrix,
            "tau": tau_value,
            "tau_bounds": {
                "tau_min": self.tau_min,
                "tau_max": self.tau_max,
                "tau_base": self.tau_base,
                "delta_scale": self.delta_scale,
            },
            "tau_statistics": self.get_tau_statistics(),
            "difficulty_statistics": self.get_difficulty_statistics(),
            "corr_tau_difficulty": self.get_tau_difficulty_correlation(),
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
                } if self.loss_mode == "adaptive_ucl_cda_v3" else None
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
    fixed_cda_alpha: float = 0.3,
    tau_init: float = 0.35,
    tau_min: float = 0.05,
    tau_max: float = 1.0,
    tau_base: float = 0.22,
    delta_scale: float = 0.12,
    tau_target: float = 0.22,
    tau_easy: float = 0.16,
    tau_hard: float = 0.32,
    tau_variance_weight: float = 1e-2,
    tau_std_floor: float = 0.03,
    tau_rank_margin_difficulty: float = 0.10,
    tau_rank_margin_value: float = 0.01,
    concentration_margin: float = 0.05,
    lambda_aux: float = 0.2,
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
        fixed_cda_alpha=fixed_cda_alpha,
        tau_init=tau_init,
        tau_min=tau_min,
        tau_max=tau_max,
        tau_base=tau_base,
        delta_scale=delta_scale,
        tau_target=tau_target,
        tau_easy=tau_easy,
        tau_hard=tau_hard,
        tau_variance_weight=tau_variance_weight,
        tau_std_floor=tau_std_floor,
        tau_rank_margin_difficulty=tau_rank_margin_difficulty,
        tau_rank_margin_value=tau_rank_margin_value,
        concentration_margin=concentration_margin,
        lambda_aux=lambda_aux,
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
