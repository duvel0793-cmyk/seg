from __future__ import annotations

from pathlib import Path
from typing import Any

from bridge.pixel_bridge import (
    aggregate_xview2_style_metrics,
    build_tile_bridge_result,
    build_tile_instance_groups,
    save_colorized_label_png,
    save_damage_confusion_matrix_plot,
    save_label_png,
)
from utils.io import ensure_dir, write_json


def save_bridge_tile_outputs(
    save_dir: Path,
    tile_id: str,
    result: dict[str, Any],
    *,
    save_pixel_predictions: bool,
    save_visuals: bool,
    save_official_style_pngs: bool,
) -> None:
    if save_pixel_predictions:
        pixel_dir = ensure_dir(save_dir / "pixel_predictions")
        save_label_png(pixel_dir / f"{tile_id}_damage_prediction.png", result["prediction_map"])
    if save_official_style_pngs:
        official_dir = ensure_dir(save_dir / "official_prediction_pngs")
        save_label_png(official_dir / f"{tile_id}_damage_prediction.png", result["prediction_map"])
        save_label_png(
            official_dir / f"{tile_id}_localization_prediction.png",
            (result["prediction_map"] > 0).astype("uint8"),
        )
    if save_visuals:
        visuals_dir = ensure_dir(save_dir / "visuals")
        save_colorized_label_png(visuals_dir / f"{tile_id}_pred_color.png", result["prediction_map"])
        save_colorized_label_png(visuals_dir / f"{tile_id}_gt_color.png", result["gt_damage_map"])


def export_bridge_outputs(
    dataset,
    prediction_records: list[dict[str, Any]],
    *,
    save_dir: str | Path,
    target_source: str,
    use_official_target_png: bool,
    overlap_policy: str,
    save_pixel_predictions: bool,
    save_visuals: bool,
    save_official_style_pngs: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    save_dir = ensure_dir(save_dir)
    tile_groups = build_tile_instance_groups(dataset.samples, prediction_records)
    per_tile_scores: list[dict[str, Any]] = []
    for tile_id, tile_group in sorted(tile_groups.items()):
        tile_result = build_tile_bridge_result(
            tile_group,
            target_source=target_source,
            use_official_target_png=use_official_target_png,
            overlap_policy=overlap_policy,
        )
        save_bridge_tile_outputs(
            save_dir,
            tile_id,
            tile_result,
            save_pixel_predictions=save_pixel_predictions,
            save_visuals=save_visuals,
            save_official_style_pngs=save_official_style_pngs,
        )
        per_tile_scores.append(tile_result["tile_scores"])

    aggregate_metrics = aggregate_xview2_style_metrics(per_tile_scores)
    write_json(Path(save_dir) / "bridge_metrics.json", aggregate_metrics)
    write_json(Path(save_dir) / "per_tile_scores.json", per_tile_scores)
    save_damage_confusion_matrix_plot(
        aggregate_metrics["official_style_confusion_matrix_on_gt_building_pixels"],
        Path(save_dir) / "official_style_confusion_matrix_on_gt_building_pixels.png",
    )
    return aggregate_metrics, per_tile_scores


def run_bridge_evaluation(dataset, prediction_records: list[dict[str, Any]], *, config: dict[str, Any], save_dir: str | Path):
    bridge_cfg = config["bridge"]
    return export_bridge_outputs(
        dataset,
        prediction_records,
        save_dir=save_dir,
        target_source=str(bridge_cfg["target_source"]),
        use_official_target_png=bool(bridge_cfg["use_official_target_png"]),
        overlap_policy=str(bridge_cfg["overlap_policy"]),
        save_pixel_predictions=bool(bridge_cfg["save_pixel_predictions"]),
        save_visuals=bool(bridge_cfg["save_visuals"]),
        save_official_style_pngs=bool(bridge_cfg["save_official_style_pngs"]),
    )
