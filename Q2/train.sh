#!/bin/bash
# train.sh — orchestrates baseline / pre_adapt / post_adapt pipelines for Q2.
#
# Usage:
#   ./train.sh [--mode baseline|pre_adapt|post_adapt]
#              [--output_dir <dir>]
#              [--languages hi,kn]
#              [--max_samples 5000]
#              [--mlm_epochs 1]
#              [--mlm_lr 1e-5]
#              [--mlm_batch_size 4]
#              [--mlm_max_len 256]
#              [--mlm_accumulation_steps 4]
#              [--retune_epochs 1]
#              [--config_path ./config.json]
#
# Modes:
#   baseline    — English RE fine-tuning only (default, identical to old behaviour)
#   pre_adapt   — MLM on hi/kn unlabeled corpus → RE fine-tuning
#   post_adapt  — RE fine-tuning → MLM on hi/kn corpus → short RE re-tune

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
MODE="baseline"
OUTPUT_DIR="./output"
LANGUAGES="hi,kn"
MAX_SAMPLES=5000
MLM_EPOCHS=1
MLM_LR=1e-5
MLM_BATCH=4
MLM_MAX_LEN=256
MLM_ACCUM=4
RETUNE_EPOCHS=1
CONFIG_PATH="./config.json"

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)               MODE="$2";            shift 2 ;;
    --output_dir)         OUTPUT_DIR="$2";      shift 2 ;;
    --languages)          LANGUAGES="$2";       shift 2 ;;
    --max_samples)        MAX_SAMPLES="$2";     shift 2 ;;
    --mlm_epochs)         MLM_EPOCHS="$2";      shift 2 ;;
    --mlm_lr)             MLM_LR="$2";          shift 2 ;;
    --mlm_batch_size)     MLM_BATCH="$2";       shift 2 ;;
    --mlm_max_len)        MLM_MAX_LEN="$2";     shift 2 ;;
    --mlm_accumulation_steps) MLM_ACCUM="$2";  shift 2 ;;
    --retune_epochs)      RETUNE_EPOCHS="$2";   shift 2 ;;
    --config_path)        CONFIG_PATH="$2";     shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

mkdir -p "$OUTPUT_DIR"

# ── Helper: run MLM pretraining ──────────────────────────────────────────────
run_mlm() {
  local start_from="$1"
  local out_dir="$2"
  local extra_args=()
  if [[ -n "$start_from" ]]; then
    extra_args=(--start_from "$start_from")
  fi
  python -u pretrain.py \
    --output_dir "$out_dir" \
    "${extra_args[@]}" \
    --languages "$LANGUAGES" \
    --max_samples "$MAX_SAMPLES" \
    --epochs "$MLM_EPOCHS" \
    --lr "$MLM_LR" \
    --batch_size "$MLM_BATCH" \
    --max_len "$MLM_MAX_LEN" \
    --accumulation_steps "$MLM_ACCUM"
}

# ── Pipeline dispatch ────────────────────────────────────────────────────────
case "$MODE" in

  baseline)
    echo "[Stage 1/1] Baseline RE fine-tuning …"
    python -u train.py \
      --output_dir "${OUTPUT_DIR}/re_finetuned" \
      --config_path "$CONFIG_PATH"
    ;;

  pre_adapt)
    echo "[Stage 1/2] MLM adaptation on unlabeled '${LANGUAGES}' corpus …"
    run_mlm "" "${OUTPUT_DIR}/mlm_adapted"

    echo "[Stage 2/2] RE fine-tuning from MLM-adapted checkpoint …"
    python -u train.py \
      --output_dir "${OUTPUT_DIR}/re_finetuned" \
      --config_path "$CONFIG_PATH" \
      --pretrained_dir "${OUTPUT_DIR}/mlm_adapted"
    ;;

  post_adapt)
    echo "[Stage 1/3] RE fine-tuning on English labeled data …"
    python -u train.py \
      --output_dir "${OUTPUT_DIR}/re_finetuned" \
      --config_path "$CONFIG_PATH"

    echo "[Stage 2/3] MLM adaptation on unlabeled '${LANGUAGES}' corpus …"
    run_mlm "${OUTPUT_DIR}/re_finetuned" "${OUTPUT_DIR}/mlm_adapted"

    echo "[Stage 3/3] Short RE re-tune from MLM-adapted checkpoint …"
    python -u train.py \
      --output_dir "${OUTPUT_DIR}/re_retuned" \
      --config_path "$CONFIG_PATH" \
      --pretrained_dir "${OUTPUT_DIR}/mlm_adapted" \
      --epochs "$RETUNE_EPOCHS"
    ;;

  *)
    echo "Unknown mode '${MODE}'. Choose one of: baseline, pre_adapt, post_adapt"
    exit 1
    ;;

esac

echo "Done. Artifacts in: ${OUTPUT_DIR}/"
