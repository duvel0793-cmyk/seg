#!/usr/bin/env bash
set -euo pipefail

cd /home/lky/code/dabqn_evidence_damage_classifier
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" validate_dabqn.py \
  --config configs/stage3_joint.yaml \
  --checkpoint outputs/stage3_joint/checkpoints/best_bridge_score.pth \
  --split val
