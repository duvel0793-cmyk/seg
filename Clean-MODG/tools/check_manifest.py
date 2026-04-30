"""Validate a manifest CSV before training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.manifest import REQUIRED_COLUMNS, count_missing_geometry, load_manifest_dataframe, preview_rows, resolve_path, validate_manifest_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Clean-MODG manifest integrity.")
    parser.add_argument("--manifest", type=str, required=True, help="Manifest CSV path.")
    parser.add_argument("--image-root", type=str, default="", help="Optional root for relative image paths.")
    parser.add_argument("--samples", type=int, default=3, help="Number of preview rows to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_manifest_dataframe(args.manifest)
    validate_manifest_dataframe(df)
    print(f"Manifest rows: {len(df)}")
    print(f"Required columns: {REQUIRED_COLUMNS}")
    print(f"Missing geometry rows: {count_missing_geometry(df)}")
    invalid_labels = df[~df["label"].isin([0, 1, 2, 3])]
    print(f"Invalid label rows: {len(invalid_labels)}")
    missing_paths = 0
    for column in ["pre_image", "post_image"]:
        for path_str in df[column].head(100):
            if not resolve_path(path_str, args.image_root).exists():
                missing_paths += 1
    print(f"Missing image paths among first 100 rows per column: {missing_paths}")
    for row in preview_rows(df, n=args.samples):
        print(row)


if __name__ == "__main__":
    main()
