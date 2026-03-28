#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CFG="${CFG:-$ROOT_DIR/changedetection/configs/vssm1/vssm_small_224.yaml}"
DATA_ROOT="${DATA_ROOT:-/home/lky/data/xBD}"
SAVE_DIR="${SAVE_DIR:-$ROOT_DIR/saved_models/flowmamba}"
PRETRAINED="${PRETRAINED:-$ROOT_DIR/pretrained_weight/vssm_small_0229_ckpt_epoch_222.pth}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CROP_SIZE="${CROP_SIZE:-256}"
MAX_ITERS="${MAX_ITERS:-80000}"
VAL_INTERVAL="${VAL_INTERVAL:-1000}"
NUM_WORKERS="${NUM_WORKERS:-8}"
ALIGN_LOSS_WEIGHT="${ALIGN_LOSS_WEIGHT:-1.0}"

mkdir -p "$SAVE_DIR"

python "$ROOT_DIR/changedetection/script/train.py" \
  --cfg "$CFG" \
  --pretrained_weight_path "$PRETRAINED" \
  --train_dataset_path "$DATA_ROOT/train" \
  --train_data_list_path "$DATA_ROOT/xBD_list/train_all.txt" \
  --test_dataset_path "$DATA_ROOT/test" \
  --test_data_list_path "$DATA_ROOT/xBD_list/val_all.txt" \
  --batch_size "$BATCH_SIZE" \
  --crop_size "$CROP_SIZE" \
  --max_iters "$MAX_ITERS" \
  --val_interval "$VAL_INTERVAL" \
  --num_workers "$NUM_WORKERS" \
  --align_loss_weight "$ALIGN_LOSS_WEIGHT" \
  --model_param_path "$SAVE_DIR" \
  --log_file "$SAVE_DIR/train.log"
