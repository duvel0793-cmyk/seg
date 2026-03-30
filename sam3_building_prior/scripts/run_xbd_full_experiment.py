#!/usr/bin/env python3
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from sam3_building_prior.experiment import main as experiment_main


def main():
    return experiment_main()


if __name__ == "__main__":
    raise SystemExit(main())
