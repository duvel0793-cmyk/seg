from __future__ import annotations

import torch


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, width, height = boxes.unbind(dim=-1)
    x1 = cx - (width * 0.5)
    y1 = cy - (height * 0.5)
    x2 = cx + (width * 0.5)
    y2 = cy + (height * 0.5)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    width = x2 - x1
    height = y2 - y1
    return torch.stack([cx, cy, width, height], dim=-1)


def _box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[..., 2] - boxes[..., 0]).clamp(min=0.0) * (boxes[..., 3] - boxes[..., 1]).clamp(min=0.0)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    area1 = _box_area(boxes1)
    area2 = _box_area(boxes2)

    top_left = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    width_height = (bottom_right - top_left).clamp(min=0.0)
    intersection = width_height[..., 0] * width_height[..., 1]
    union = area1[:, None] + area2 - intersection
    return intersection / union.clamp_min(1e-6), union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    iou, union = box_iou(boxes1, boxes2)
    top_left = torch.minimum(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.maximum(boxes1[:, None, 2:], boxes2[:, 2:])
    width_height = (bottom_right - top_left).clamp(min=0.0)
    area = width_height[..., 0] * width_height[..., 1]
    return iou - ((area - union) / area.clamp_min(1e-6))
