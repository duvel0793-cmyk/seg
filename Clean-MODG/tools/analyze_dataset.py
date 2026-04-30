"""Quick dataset analysis utility for manifest-driven xBD instances."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.manifest import load_manifest_dataframe, parse_json_field, validate_manifest_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze label and split distribution of a Clean-MODG manifest.")
    parser.add_argument("--manifest", type=str, required=True, help="Manifest CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_manifest_dataframe(args.manifest)
    validate_manifest_dataframe(df)
    print("Rows:", len(df))
    print("Split distribution:", dict(Counter(df["split"].astype(str).tolist())))
    print("Label distribution:", dict(Counter(df["label"].astype(int).tolist())))
    polygon_rows = sum(parse_json_field(value) is not None for value in df["polygon"].tolist())
    bbox_rows = sum(parse_json_field(value) is not None for value in df["bbox"].tolist())
    print("Rows with polygon:", polygon_rows)
    print("Rows with bbox:", bbox_rows)


if __name__ == "__main__":
    main()
