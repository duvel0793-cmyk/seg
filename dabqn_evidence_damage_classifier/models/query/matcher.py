from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from losses.box_losses import box_cxcywh_to_xyxy, generalized_box_iou


def _sigmoid_dice_cost(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = logits.sigmoid().flatten(1)
    targets_flat = targets.flatten(1)
    intersection = torch.einsum("nc,mc->nm", probs, targets_flat)
    denominator = probs.sum(dim=1, keepdim=True) + targets_flat.sum(dim=1).unsqueeze(0)
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0))


def _sigmoid_bce_cost(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    num_queries = int(logits.shape[0])
    num_targets = int(targets.shape[0])
    cost = logits.new_zeros((num_queries, num_targets))
    for target_index in range(num_targets):
        target = targets[target_index].unsqueeze(0).expand_as(logits)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        cost[:, target_index] = bce.flatten(1).mean(dim=1)
    return cost


def _find_uncovered_zero(matrix: np.ndarray, row_cover: np.ndarray, col_cover: np.ndarray) -> tuple[int, int] | None:
    n = matrix.shape[0]
    for row in range(n):
        if row_cover[row]:
            continue
        for col in range(n):
            if not col_cover[col] and abs(matrix[row, col]) < 1e-10:
                return row, col
    return None


def _find_star_in_row(stars: np.ndarray, row: int) -> int | None:
    cols = np.where(stars[row])[0]
    return None if cols.size == 0 else int(cols[0])


def _find_star_in_col(stars: np.ndarray, col: int) -> int | None:
    rows = np.where(stars[:, col])[0]
    return None if rows.size == 0 else int(rows[0])


def _find_prime_in_row(primes: np.ndarray, row: int) -> int | None:
    cols = np.where(primes[row])[0]
    return None if cols.size == 0 else int(cols[0])


def _augment_path(stars: np.ndarray, primes: np.ndarray, start_row: int, start_col: int) -> None:
    path: list[tuple[int, int]] = [(start_row, start_col)]
    current_col = start_col
    while True:
        star_row = _find_star_in_col(stars, current_col)
        if star_row is None:
            break
        path.append((star_row, current_col))
        prime_col = _find_prime_in_row(primes, star_row)
        if prime_col is None:
            raise RuntimeError("Prime path is broken during Hungarian augmentation.")
        path.append((star_row, prime_col))
        current_col = prime_col
    for row, col in path:
        stars[row, col] = not stars[row, col]


def hungarian_assignment(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if cost_matrix.ndim != 2:
        raise ValueError(f"Cost matrix must be 2D, got shape={cost_matrix.shape}.")
    num_rows, num_cols = cost_matrix.shape
    if num_rows == 0 or num_cols == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    size = max(num_rows, num_cols)
    pad_value = float(cost_matrix.max()) + 1.0
    matrix = np.full((size, size), pad_value, dtype=np.float64)
    matrix[:num_rows, :num_cols] = cost_matrix.astype(np.float64)

    matrix -= matrix.min(axis=1, keepdims=True)
    matrix -= matrix.min(axis=0, keepdims=True)

    stars = np.zeros((size, size), dtype=bool)
    primes = np.zeros((size, size), dtype=bool)
    row_cover = np.zeros((size,), dtype=bool)
    col_cover = np.zeros((size,), dtype=bool)

    for row in range(size):
        for col in range(size):
            if abs(matrix[row, col]) < 1e-10 and (not row_cover[row]) and (not col_cover[col]):
                stars[row, col] = True
                row_cover[row] = True
                col_cover[col] = True
    row_cover[:] = False
    col_cover[:] = np.any(stars, axis=0)

    while int(col_cover.sum()) < size:
        zero_position = _find_uncovered_zero(matrix, row_cover, col_cover)
        while zero_position is None:
            uncovered = matrix[~row_cover][:, ~col_cover]
            if uncovered.size == 0:
                break
            min_value = float(uncovered.min())
            matrix[row_cover] += min_value
            matrix[:, ~col_cover] -= min_value
            zero_position = _find_uncovered_zero(matrix, row_cover, col_cover)
        if zero_position is None:
            break
        row, col = zero_position
        primes[row, col] = True
        star_col = _find_star_in_row(stars, row)
        if star_col is None:
            _augment_path(stars, primes, row, col)
            primes[:] = False
            row_cover[:] = False
            col_cover[:] = np.any(stars, axis=0)
        else:
            row_cover[row] = True
            col_cover[star_col] = False

    row_indices: list[int] = []
    col_indices: list[int] = []
    for row in range(num_rows):
        cols = np.where(stars[row, :num_cols])[0]
        if cols.size == 0:
            continue
        row_indices.append(row)
        col_indices.append(int(cols[0]))
    return np.asarray(row_indices, dtype=np.int64), np.asarray(col_indices, dtype=np.int64)


@dataclass
class MatcherCosts:
    objectness: float
    box_l1: float
    giou: float
    mask: float
    dice: float


class HungarianMatcher:
    def __init__(
        self,
        *,
        cost_objectness: float = 1.0,
        cost_box_l1: float = 5.0,
        cost_giou: float = 2.0,
        cost_mask: float = 2.0,
        cost_dice: float = 5.0,
        add_damage_cost: bool = False,
        cost_damage: float = 0.0,
    ) -> None:
        self.costs = MatcherCosts(cost_objectness, cost_box_l1, cost_giou, cost_mask, cost_dice)
        self.add_damage_cost = bool(add_damage_cost)
        self.cost_damage = float(cost_damage)

    def _build_cost_matrix(self, outputs: dict[str, Any], target: dict[str, torch.Tensor], batch_index: int) -> torch.Tensor:
        pred_logits = outputs["pred_logits"][batch_index]
        pred_boxes = outputs["pred_boxes"][batch_index]
        pred_masks = outputs["pred_masks_lowres"][batch_index]
        target_boxes = target["boxes_norm"]
        target_masks = target["masks"]

        if target_boxes.numel() == 0:
            return pred_boxes.new_zeros((pred_boxes.shape[0], 0))

        if pred_masks.shape[-2:] != target_masks.shape[-2:]:
            target_masks = F.interpolate(
                target_masks.unsqueeze(1).float(),
                size=pred_masks.shape[-2:],
                mode="nearest",
            ).squeeze(1)

        object_prob = pred_logits.softmax(dim=-1)[:, 1]
        objectness_cost = -object_prob[:, None].expand(-1, target_boxes.shape[0])
        box_l1_cost = torch.cdist(pred_boxes, target_boxes, p=1)
        giou_cost = 1.0 - generalized_box_iou(box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(target_boxes))
        mask_cost = _sigmoid_bce_cost(pred_masks, target_masks)
        dice_cost = _sigmoid_dice_cost(pred_masks, target_masks)

        total_cost = (
            self.costs.objectness * objectness_cost
            + self.costs.box_l1 * box_l1_cost
            + self.costs.giou * giou_cost
            + self.costs.mask * mask_cost
            + self.costs.dice * dice_cost
        )
        if self.add_damage_cost and outputs.get("damage_probabilities") is not None:
            damage_probs = outputs["damage_probabilities"][batch_index]
            total_cost = total_cost - (self.cost_damage * damage_probs[:, target["labels"]])
        return total_cost

    @torch.no_grad()
    def __call__(self, outputs: dict[str, Any], targets: list[dict[str, torch.Tensor]]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        matches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for batch_index, target in enumerate(targets):
            num_targets = int(target["labels"].numel())
            if num_targets == 0:
                matches.append(
                    (
                        torch.zeros((0,), dtype=torch.long, device=outputs["pred_logits"].device),
                        torch.zeros((0,), dtype=torch.long, device=outputs["pred_logits"].device),
                    )
                )
                continue
            cost = self._build_cost_matrix(outputs, target, batch_index).detach().cpu().numpy()
            pred_indices, target_indices = hungarian_assignment(cost)
            matches.append(
                (
                    torch.as_tensor(pred_indices, dtype=torch.long, device=outputs["pred_logits"].device),
                    torch.as_tensor(target_indices, dtype=torch.long, device=outputs["pred_logits"].device),
                )
            )
        return matches
