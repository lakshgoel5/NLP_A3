#!/bin/bash
# Usage: ./train.sh <output_dir>

OUTPUT_DIR="${1:-./output}"
mkdir -p "$OUTPUT_DIR"
python -u train.py --output_dir "$OUTPUT_DIR" --config_path ./config.json