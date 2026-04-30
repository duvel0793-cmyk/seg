"""Export confusion matrix PNG from a predictions CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics.confusion import confusion_matrix, save_confusion_matrix_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export confusion matrix from predictions CSV.")
    parser.add_argument("--predictions", type=str, required=True, help="Predictions CSV path.")
    parser.add_argument("--output", type=str, required=True, help="Output PNG path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.predictions)
    matrix = confusion_matrix(df["pred"].tolist(), df["target"].tolist())
    save_confusion_matrix_png(matrix, args.output)
    print(f"Saved confusion matrix to {args.output}")


if __name__ == "__main__":
    main()
