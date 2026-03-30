"""Shared dataclasses used across inference and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple


@dataclass
class ImageRecord:
    path: Path
    stem: str


@dataclass
class InstancePrediction:
    instance_id: int
    bbox_xyxy: Tuple[int, int, int, int]
    area: int
    score: Optional[float]
    prompt_type: str
    mask: Any
    prompt_id: Optional[int] = None
    prompt_bbox_xyxy: Optional[Tuple[int, int, int, int]] = None
    prompt_point_xy: Optional[Tuple[int, int]] = None
    coarse_area: Optional[int] = None


@dataclass
class PromptRegion:
    prompt_id: int
    bbox_xyxy: Tuple[int, int, int, int]
    center_point_xy: Tuple[int, int]
    area: int
