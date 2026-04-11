#!/bin/bash
# Usage: ./train.sh <output_dir> [baseline|pre_adapt|post_adapt]

set -euo pipefail

OUTPUT_DIR="${1:-./output}"
MODE="${2:-baseline}"

mkdir -p "$OUTPUT_DIR"

case "$MODE" in

  baseline)
    echo "[1/1] RE fine-tuning only ..."
    python -u train.py --output_dir "$OUTPUT_DIR/re_finetuned_b" --config_path ./config.json
    ;;

  pre_adapt)
    echo "[1/2] CPT on Indic Wikipedia ..."
    python -u pretrain.py --output_dir "$OUTPUT_DIR/cpt_adapted_pre" --config_path ./pretrain_config.json

    echo "[2/2] RE fine-tuning from adapted checkpoint ..."
    python -u train.py --output_dir "$OUTPUT_DIR/re_finetuned_pre" --config_path ./config.json --pretrained_dir "$OUTPUT_DIR/cpt_adapted_pre"
    ;;

  post_adapt)
    echo "[1/3] RE fine-tuning on English ..."
    python -u train.py --output_dir "$OUTPUT_DIR/re_finetuned_post" --config_path ./config.json

    echo "[2/3] CPT on Indic Wikipedia ..."
    python -u pretrain.py --output_dir "$OUTPUT_DIR/cpt_adapted_post" --config_path ./pretrain_config.json --pretrained_dir "$OUTPUT_DIR/re_finetuned_post"

    echo "[3/3] Short RE re-tune ..."
    python -u train.py --output_dir "$OUTPUT_DIR/re_retuned_post" --config_path ./config.json --pretrained_dir "$OUTPUT_DIR/cpt_adapted_post" --epochs 1
    ;;

  *)
    echo "Unknown mode: $MODE. Use baseline|pre_adapt|post_adapt"
    exit 1
    ;;
esac

echo "Done. Output in $OUTPUT_DIR/"