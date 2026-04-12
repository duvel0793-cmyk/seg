#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${1:-$PROJECT_ROOT/configs/exp_step1.yaml}"
CONDA_ENV="${CONDA_ENV:-}"

if [[ -n "$CONDA_ENV" ]]; then
  conda run -n "$CONDA_ENV" python "$PROJECT_ROOT/tools/train.py" --config "$CONFIG"
else
  python "$PROJECT_ROOT/tools/train.py" --config "$CONFIG"
fi
