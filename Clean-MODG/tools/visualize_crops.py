"""Visualize synchronized tight/context crops from the manifest."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.crop_policy import build_tight_context_crops, polygon_to_bbox
from data.manifest import load_manifest_dataframe, parse_json_field, resolve_path
from data.mask_ops import bbox_to_mask, polygon_to_mask
from utils.common import ensure_dir
from utils.visualization import overlay_mask, save_image_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Clean-MODG tight/context crops.")
    parser.add_argument("--manifest", type=str, required=True, help="Manifest CSV path.")
    parser.add_argument("--image-root", type=str, default="", help="Optional root for relative image paths.")
    parser.add_argument("--output-dir", type=str, default="outputs/visualize_crops", help="Output directory.")
    parser.add_argument("--num-samples", type=int, default=4, help="Number of random samples.")
    parser.add_argument("--tight-size", type=int, default=256, help="Tight crop size.")
    parser.add_argument("--context-size", type=int, default=256, help="Context crop size.")
    parser.add_argument("--tight-padding", type=float, default=0.15, help="Tight padding ratio.")
    parser.add_argument("--context-padding", type=float, default=1.0, help="Context padding ratio.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_manifest_dataframe(args.manifest)
    if len(df) == 0:
        raise SystemExit("Manifest is empty.")
    indices = random.sample(range(len(df)), k=min(args.num_samples, len(df)))
    output_dir = ensure_dir(args.output_dir)
    for out_idx, row_idx in enumerate(indices):
        row = df.iloc[row_idx].to_dict()
        pre_image = np.asarray(Image.open(resolve_path(row["pre_image"], args.image_root)).convert("RGB"))
        post_image = np.asarray(Image.open(resolve_path(row["post_image"], args.image_root)).convert("RGB"))
        polygon = parse_json_field(row.get("polygon"))
        bbox = parse_json_field(row.get("bbox"))
        if polygon is not None:
            bbox = polygon_to_bbox(polygon)
            mask = polygon_to_mask(pre_image.shape[:2], polygon)
        elif bbox is not None:
            mask = bbox_to_mask(pre_image.shape[:2], bbox)
        else:
            continue
        crops = build_tight_context_crops(
            pre_image=pre_image,
            post_image=post_image,
            mask=mask,
            bbox=bbox,
            tight_size=args.tight_size,
            context_size=args.context_size,
            tight_padding_ratio=args.tight_padding,
            context_padding_ratio=args.context_padding,
        )
        images = [
            overlay_mask(crops["pre_tight"], crops["mask_tight"]),
            overlay_mask(crops["post_tight"], crops["mask_tight"]),
            overlay_mask(crops["pre_context"], crops["mask_context"]),
            overlay_mask(crops["post_context"], crops["mask_context"]),
        ]
        titles = [
            f"pre_tight | y={row['label']}",
            "post_tight",
            "pre_context",
            "post_context",
        ]
        save_image_grid(images, titles, Path(output_dir) / f"sample_{out_idx:03d}.png", cols=2, figsize=(8, 8))
    print(f"Saved crop visualizations to {output_dir}")


if __name__ == "__main__":
    main()
