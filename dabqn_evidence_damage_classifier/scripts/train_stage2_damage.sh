#!/usr/bin/env bash
set -euo pipefail

cd /home/lky/code/dabqn_evidence_damage_classifier
python3 train_dabqn.py --config configs/stage2_damage.yaml --stage damage
