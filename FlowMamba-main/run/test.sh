#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CFG="${CFG:-$ROOT_DIR/changedetection/configs/vssm1/vssm_small_224.yaml}"
DATA_ROOT="${DATA_ROOT:-/home/lky/data/xBD}"
PRETRAINED="${PRETRAINED:-$ROOT_DIR/pretrained_weight/vssm_small_0229_ckpt_epoch_222.pth}"
CKPT="${CKPT:-$ROOT_DIR/saved_models/flowmamba/best_model.pth}"
SAVE_DIR="${SAVE_DIR:-$ROOT_DIR/saved_models/flowmamba/test_results}"
NUM_WORKERS="${NUM_WORKERS:-8}"

mkdir -p "$SAVE_DIR"

python "$ROOT_DIR/changedetection/script/test.py" \
  --cfg "$CFG" \
  --pretrained_weight_path "$PRETRAINED" \
  --test_dataset_path "$DATA_ROOT/test" \
  --test_data_list_path "$DATA_ROOT/xBD_list/val_all.txt" \
  --resume "$CKPT" \
  --result_saved_path "$SAVE_DIR" \
  --log_file "$SAVE_DIR/test.log" \
  --num_workers "$NUM_WORKERS"
