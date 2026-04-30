"""Visualize common misclassification patterns from predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.crop_policy import build_tight_context_crops, polygon_to_bbox
from data.manifest import load_manifest_dataframe, parse_json_field, resolve_path
from data.mask_ops import bbox_to_mask, polygon_to_mask
from utils.common import CLASS_NAMES, ensure_dir
from utils.visualization import overlay_mask, save_image_grid


ERROR_PAIRS = [(0, 3), (3, 0), (1, 2), (2, 1)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize typical Clean-MODG errors.")
    parser.add_argument("--predictions", type=str, required=True, help="Predictions CSV path.")
    parser.add_argument("--manifest", type=str, required=True, help="Manifest CSV path.")
    parser.add_argument("--image-root", type=str, default="", help="Optional image root for relative paths.")
    parser.add_argument("--output-dir", type=str, default="outputs/visualize_errors", help="Output directory.")
    parser.add_argument("--limit-per-type", type=int, default=8, help="Max cases per error type.")
    return parser.parse_args()


def _crop_case(row: pd.Series, image_root: str):
    pre_image = np.asarray(Image.open(resolve_path(row["pre_image"], image_root)).convert("RGB"))
    post_image = np.asarray(Image.open(resolve_path(row["post_image"], image_root)).convert("RGB"))
    polygon = parse_json_field(row.get("polygon"))
    bbox = parse_json_field(row.get("bbox"))
    if polygon is not None:
        bbox = polygon_to_bbox(polygon)
        mask = polygon_to_mask(pre_image.shape[:2], polygon)
    elif bbox is not None:
        mask = bbox_to_mask(pre_image.shape[:2], bbox)
    else:
        return None
    return build_tight_context_crops(
        pre_image=pre_image,
        post_image=post_image,
        mask=mask,
        bbox=bbox,
        tight_size=256,
        context_size=256,
        tight_padding_ratio=0.15,
        context_padding_ratio=1.0,
    )


def main() -> None:
    args = parse_args()
    pred_df = pd.read_csv(args.predictions)
    manifest_df = load_manifest_dataframe(args.manifest)
    merged = pred_df.merge(manifest_df, how="left", on=["building_id", "disaster_id"])
    output_dir = ensure_dir(args.output_dir)
    for target_label, pred_label in ERROR_PAIRS:
        subset = merged[(merged["target"] == target_label) & (merged["pred"] == pred_label)].head(args.limit_per_type)
        if subset.empty:
            continue
        images = []
        titles = []
        for _, row in subset.iterrows():
            crops = _crop_case(row, args.image_root)
            if crops is None:
                continue
            images.extend(
                [
                    overlay_mask(crops["pre_tight"], crops["mask_tight"]),
                    overlay_mask(crops["post_tight"], crops["mask_tight"]),
                ]
            )
            titles.extend(
                [
                    f"pre | {CLASS_NAMES[target_label]}->{CLASS_NAMES[pred_label]}",
                    f"post | {row['building_id']}",
                ]
            )
        if images:
            filename = f"{CLASS_NAMES[target_label]}_to_{CLASS_NAMES[pred_label]}.png"
            save_image_grid(images, titles, Path(output_dir) / filename, cols=2, figsize=(10, 4 * len(images)))
    print(f"Saved error visualizations to {output_dir}")


if __name__ == "__main__":
    main()
