#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python train.py --config configs/clean_dual_scale.yaml
