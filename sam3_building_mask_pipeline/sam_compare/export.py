"""Export helpers for predictions and evaluation reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .utils import ensure_dir, to_jsonable


COMPARISON_PANEL_TITLES = (
    "Original",
    "Label",
    "Prediction",
    "Diff (TP/FP/FN)",
)
_GRID_GAP = 8
_TITLE_BAND_HEIGHT = 26
_GRID_BACKGROUND = (20, 20, 20)
_TITLE_COLOR = (255, 255, 255)
_GT_COLOR = (46, 204, 113)
_PRED_COLOR = (255, 153, 51)
_FP_COLOR = (231, 76, 60)
_FN_COLOR = (52, 152, 219)
_BOUNDARY_COLOR = (255, 255, 0)
_OVERLAY_ALPHA = 110


def save_mask_png(mask_array: np.ndarray, output_path: str | Path) -> Path:
    """Save a uint8 mask as PNG."""
    output = Path(output_path).expanduser().resolve()
    ensure_dir(output.parent)
    Image.fromarray(mask_array.astype(np.uint8), mode="L").save(output)
    return output


def write_metrics_json(metrics: dict, output_path: str | Path) -> Path:
    """Write a metrics dictionary to JSON."""
    output = Path(output_path).expanduser().resolve()
    ensure_dir(output.parent)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(metrics), handle, indent=2, ensure_ascii=False)
    return output


def write_per_image_metrics_csv(
    records: Iterable[dict],
    output_path: str | Path,
) -> Path:
    """Write per-image metric rows to CSV."""
    records = list(records)
    output = Path(output_path).expanduser().resolve()
    ensure_dir(output.parent)
    if not records:
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "image_id",
                    "iou",
                    "f1",
                    "precision",
                    "recall",
                    "pixel_accuracy",
                    "tp",
                    "fp",
                    "fn",
                    "tn",
                    "total_pixels",
                    "gt_foreground_ratio",
                    "pred_foreground_ratio",
                    "gt_empty",
                    "pred_empty",
                    "size_bucket",
                    "image_path",
                    "target_path",
                    "pred_mask_path",
                    "comparison_vis_path",
                    "overlay_vis_path",
                ]
            )
        return output

    fieldnames = list(records[0].keys())
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    return output


def _to_binary_mask(mask_array: np.ndarray) -> np.ndarray:
    mask_np = np.asarray(mask_array)
    if mask_np.ndim > 2:
        mask_np = np.squeeze(mask_np)
    return (mask_np > 0).astype(np.uint8)


def _load_rgb_image(image_path: str | Path, expected_shape: tuple[int, int]) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    expected_width, expected_height = expected_shape
    if image.size != (expected_width, expected_height):
        image = image.resize((expected_width, expected_height), resample=Image.Resampling.BILINEAR)
    return image


def _colorize_binary_mask(
    mask_array: np.ndarray,
    *,
    foreground_color: tuple[int, int, int],
    background_color: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    mask = _to_binary_mask(mask_array).astype(bool)
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    colored[...] = background_color
    colored[mask] = foreground_color
    return Image.fromarray(colored, mode="RGB")


def _build_diff_panel(pred_mask_array: np.ndarray, gt_mask_array: np.ndarray) -> Image.Image:
    pred_mask = _to_binary_mask(pred_mask_array).astype(bool)
    gt_mask = _to_binary_mask(gt_mask_array).astype(bool)
    diff = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    diff[np.logical_and(pred_mask, gt_mask)] = _GT_COLOR
    diff[np.logical_and(pred_mask, np.logical_not(gt_mask))] = _FP_COLOR
    diff[np.logical_and(np.logical_not(pred_mask), gt_mask)] = _FN_COLOR
    return Image.fromarray(diff, mode="RGB")


def _draw_titled_grid(
    panels: list[Image.Image],
    titles: tuple[str, ...],
    output_path: str | Path,
) -> Path:
    if not panels:
        raise ValueError("At least one panel is required to build a visualization grid.")

    panel_width, panel_height = panels[0].size
    columns = 2
    rows = (len(panels) + columns - 1) // columns
    canvas_width = columns * panel_width + (columns + 1) * _GRID_GAP
    canvas_height = rows * (panel_height + _TITLE_BAND_HEIGHT) + (rows + 1) * _GRID_GAP
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=_GRID_BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for index, panel in enumerate(panels):
        row = index // columns
        col = index % columns
        x_offset = _GRID_GAP + col * (panel_width + _GRID_GAP)
        y_offset = _GRID_GAP + row * (panel_height + _TITLE_BAND_HEIGHT + _GRID_GAP)
        draw.text(
            (x_offset + 8, y_offset + 6),
            titles[index],
            fill=_TITLE_COLOR,
            font=font,
        )
        canvas.paste(panel, (x_offset, y_offset + _TITLE_BAND_HEIGHT))

    output = Path(output_path).expanduser().resolve()
    ensure_dir(output.parent)
    canvas.save(output)
    return output


def _compute_mask_boundary(mask_array: np.ndarray) -> np.ndarray:
    mask = _to_binary_mask(mask_array).astype(bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)

    padded = np.pad(mask, pad_width=1, mode="constant", constant_values=False)
    eroded = (
        padded[:-2, :-2]
        & padded[:-2, 1:-1]
        & padded[:-2, 2:]
        & padded[1:-1, :-2]
        & padded[1:-1, 1:-1]
        & padded[1:-1, 2:]
        & padded[2:, :-2]
        & padded[2:, 1:-1]
        & padded[2:, 2:]
    )
    return np.logical_and(mask, np.logical_not(eroded))


def _compute_bbox(mask_array: np.ndarray) -> tuple[int, int, int, int] | None:
    mask = _to_binary_mask(mask_array).astype(bool)
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def save_prediction_label_comparison(
    image_path: str | Path,
    pred_mask_array: np.ndarray,
    gt_mask_array: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Save a 2x2 visualization grid for original image, GT, prediction, and diff."""
    gt_mask = _to_binary_mask(gt_mask_array)
    pred_mask = _to_binary_mask(pred_mask_array)
    expected_shape = (gt_mask.shape[1], gt_mask.shape[0])
    original_image = _load_rgb_image(image_path, expected_shape)
    panels = [
        original_image,
        _colorize_binary_mask(gt_mask, foreground_color=_GT_COLOR),
        _colorize_binary_mask(pred_mask, foreground_color=_PRED_COLOR),
        _build_diff_panel(pred_mask, gt_mask),
    ]
    return _draw_titled_grid(panels, COMPARISON_PANEL_TITLES, output_path)


def save_prediction_overlay(
    image_path: str | Path,
    pred_mask_array: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Overlay the predicted region on the original RGB image."""
    pred_mask = _to_binary_mask(pred_mask_array)
    expected_shape = (pred_mask.shape[1], pred_mask.shape[0])
    image = _load_rgb_image(image_path, expected_shape).convert("RGBA")
    pred_foreground = pred_mask.astype(bool)

    overlay = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 4), dtype=np.uint8)
    overlay[pred_foreground] = (*_PRED_COLOR, _OVERLAY_ALPHA)
    boundary = _compute_mask_boundary(pred_mask)
    overlay[boundary] = (*_BOUNDARY_COLOR, 255)

    blended = Image.alpha_composite(image, Image.fromarray(overlay, mode="RGBA"))
    bbox = _compute_bbox(pred_mask)
    if bbox is not None:
        draw = ImageDraw.Draw(blended)
        draw.rectangle(bbox, outline=_BOUNDARY_COLOR, width=3)

    output = Path(output_path).expanduser().resolve()
    ensure_dir(output.parent)
    blended.convert("RGB").save(output)
    return output
