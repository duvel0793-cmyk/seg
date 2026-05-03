from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mean_pool(feature: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    if feature.ndim != 4 or masks.ndim != 4:
        raise ValueError(f"Expected feature [B,C,H,W] and masks [B,Q,H,W], got {feature.shape} and {masks.shape}.")
    feature_flat = feature.flatten(2)
    masks_flat = masks.flatten(2)
    weights = masks_flat / masks_flat.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    pooled = torch.einsum("bqh,bch->bqc", weights, feature_flat)
    return pooled


def dilate_masks(masks: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return masks
    padding = kernel_size // 2
    return F.max_pool2d(masks, kernel_size=kernel_size, stride=1, padding=padding)


def expand_boxes_cxcywh(boxes: torch.Tensor, scale_factor: float) -> torch.Tensor:
    cx, cy, width, height = boxes.unbind(dim=-1)
    return torch.stack([cx, cy, width * scale_factor, height * scale_factor], dim=-1)


def box_cxcywh_to_grid_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, width, height = boxes.unbind(dim=-1)
    x1 = (cx - (width * 0.5)).clamp(0.0, 1.0)
    y1 = (cy - (height * 0.5)).clamp(0.0, 1.0)
    x2 = (cx + (width * 0.5)).clamp(0.0, 1.0)
    y2 = (cy + (height * 0.5)).clamp(0.0, 1.0)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def crop_feature_by_boxes(feature: torch.Tensor, boxes_cxcywh: torch.Tensor, output_size: int) -> torch.Tensor:
    if feature.ndim != 4 or boxes_cxcywh.ndim != 3:
        raise ValueError(f"Expected feature [B,C,H,W] and boxes [B,Q,4], got {feature.shape} and {boxes_cxcywh.shape}.")
    batch_size, channels, _, _ = feature.shape
    num_queries = int(boxes_cxcywh.shape[1])
    boxes_xyxy = box_cxcywh_to_grid_xyxy(boxes_cxcywh)
    grid_y = torch.linspace(0.0, 1.0, output_size, device=feature.device, dtype=feature.dtype)
    grid_x = torch.linspace(0.0, 1.0, output_size, device=feature.device, dtype=feature.dtype)
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    base_grid = torch.stack([xx, yy], dim=-1).view(1, 1, output_size, output_size, 2)
    boxes = boxes_xyxy.view(batch_size, num_queries, 1, 1, 4)
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    grid_x_norm = x1 + ((x2 - x1).clamp_min(1e-6) * base_grid[..., 0])
    grid_y_norm = y1 + ((y2 - y1).clamp_min(1e-6) * base_grid[..., 1])
    grid = torch.stack([(grid_x_norm * 2.0) - 1.0, (grid_y_norm * 2.0) - 1.0], dim=-1)
    feature_expanded = feature.unsqueeze(1).expand(-1, num_queries, -1, -1, -1).reshape(batch_size * num_queries, channels, feature.shape[-2], feature.shape[-1])
    grid = grid.reshape(batch_size * num_queries, output_size, output_size, 2)
    crops = F.grid_sample(feature_expanded, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return crops.view(batch_size, num_queries, channels, output_size, output_size)
