#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${2:-$PROJECT_ROOT/configs/exp_step1.yaml}"
CHECKPOINT="${1:-$PROJECT_ROOT/outputs/exp_step1_vmamba/checkpoints/latest.pth}"
CONDA_ENV="${CONDA_ENV:-}"

if [[ -n "$CONDA_ENV" ]]; then
  conda run -n "$CONDA_ENV" python "$PROJECT_ROOT/tools/validate.py" --config "$CONFIG" --checkpoint "$CHECKPOINT"
else
  python "$PROJECT_ROOT/tools/validate.py" --config "$CONFIG" --checkpoint "$CHECKPOINT"
fi
