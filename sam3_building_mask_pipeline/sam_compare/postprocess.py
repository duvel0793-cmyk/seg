"""Post-processing utilities for logits, masks, and export-friendly arrays."""

from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

try:  # pragma: no cover - optional dependency path
    from scipy import ndimage as scipy_ndimage
except ModuleNotFoundError:  # pragma: no cover - optional dependency path
    scipy_ndimage = None


def resize_logits(logits: torch.Tensor, output_shape: Sequence[int]) -> torch.Tensor:
    """Resize logits to a target spatial shape."""
    return F.interpolate(
        logits,
        size=tuple(output_shape),
        mode="bilinear",
        align_corners=False,
    )


def resize_probabilities(probabilities: torch.Tensor, output_shape: Sequence[int]) -> torch.Tensor:
    """Resize a probability map to a target spatial shape."""
    return F.interpolate(
        probabilities,
        size=tuple(output_shape),
        mode="bilinear",
        align_corners=False,
    )


def logits_to_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convert logits to probabilities while keeping full-resolution float maps."""
    return torch.sigmoid(logits)


def probabilities_to_binary_mask(
    probabilities: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Convert probabilities to a binary mask tensor in {0, 1}."""
    return (probabilities >= threshold).to(dtype=torch.uint8)


def logits_to_binary_mask(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Backwards-compatible logits-to-mask helper."""
    return probabilities_to_binary_mask(logits_to_probabilities(logits), threshold=threshold)


def _component_areas_numpy(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.zeros(mask.shape, dtype=np.int32)
    areas = [0]
    component_id = 0
    height, width = mask.shape
    neighbors = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or labels[row, col] != 0:
                continue
            component_id += 1
            queue: deque[tuple[int, int]] = deque([(row, col)])
            labels[row, col] = component_id
            area = 0

            while queue:
                current_row, current_col = queue.popleft()
                area += 1
                for offset_row, offset_col in neighbors:
                    next_row = current_row + offset_row
                    next_col = current_col + offset_col
                    if not (0 <= next_row < height and 0 <= next_col < width):
                        continue
                    if not mask[next_row, next_col] or labels[next_row, next_col] != 0:
                        continue
                    labels[next_row, next_col] = component_id
                    queue.append((next_row, next_col))
            areas.append(area)

    return labels, np.asarray(areas, dtype=np.int32)


def _remove_small_components(mask: np.ndarray, min_component_area: int) -> np.ndarray:
    if min_component_area <= 1 or not mask.any():
        return mask

    if scipy_ndimage is not None:
        structure = np.ones((3, 3), dtype=np.uint8)
        labels, num_labels = scipy_ndimage.label(mask, structure=structure)
        if num_labels == 0:
            return mask
        areas = np.bincount(labels.ravel())
        remove_labels = np.where((areas < min_component_area) & (np.arange(areas.size) != 0))[0]
        if remove_labels.size:
            mask = mask.copy()
            mask[np.isin(labels, remove_labels)] = False
        return mask

    labels, areas = _component_areas_numpy(mask)
    remove_labels = np.where((areas < min_component_area) & (np.arange(areas.size) != 0))[0]
    if remove_labels.size:
        mask = mask.copy()
        mask[np.isin(labels, remove_labels)] = False
    return mask


def _fill_small_holes(mask: np.ndarray, max_hole_area: int) -> np.ndarray:
    if max_hole_area <= 0 or not (~mask).any():
        return mask

    background = ~mask
    if scipy_ndimage is not None:
        structure = np.ones((3, 3), dtype=np.uint8)
        labels, num_labels = scipy_ndimage.label(background, structure=structure)
        if num_labels == 0:
            return mask
        areas = np.bincount(labels.ravel())
    else:
        labels, areas = _component_areas_numpy(background)
        if areas.size <= 1:
            return mask

    border_labels = np.unique(
        np.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]))
    )
    fill_labels = [
        label
        for label in range(1, areas.size)
        if areas[label] <= max_hole_area and label not in border_labels
    ]
    if fill_labels:
        mask = mask.copy()
        mask[np.isin(labels, fill_labels)] = True
    return mask


def postprocess_binary_mask(
    mask: torch.Tensor | np.ndarray,
    *,
    enable_postprocess: bool,
    min_component_area: int,
    max_hole_area: int,
) -> torch.Tensor | np.ndarray:
    """Apply conservative component removal and hole filling to a binary mask."""
    if not enable_postprocess:
        return mask

    input_tensor = isinstance(mask, torch.Tensor)
    if input_tensor:
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = np.asarray(mask)

    squeeze_channel = mask_np.ndim == 3 and mask_np.shape[0] == 1
    if squeeze_channel:
        mask_2d = mask_np[0] > 0
    else:
        mask_2d = mask_np > 0

    mask_2d = _remove_small_components(mask_2d, min_component_area=min_component_area)
    mask_2d = _fill_small_holes(mask_2d, max_hole_area=max_hole_area)
    processed_np = mask_2d.astype(np.uint8)

    if squeeze_channel:
        processed_np = processed_np[None, ...]
    if input_tensor:
        return torch.from_numpy(processed_np).to(device=mask.device, dtype=torch.uint8)
    return processed_np


def binary_mask_to_uint8(mask: torch.Tensor | np.ndarray, foreground_value: int = 255) -> np.ndarray:
    """Convert a binary mask to uint8 for PNG export."""
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = np.asarray(mask)
    mask_np = (mask_np > 0).astype(np.uint8)
    return mask_np * foreground_value
