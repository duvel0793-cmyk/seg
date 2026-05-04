#!/usr/bin/env bash
set -euo pipefail

cd /home/lky/code/dabqn_evidence_damage_classifier
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" train_dabqn.py --config configs/stage1_localization.yaml --stage localization
