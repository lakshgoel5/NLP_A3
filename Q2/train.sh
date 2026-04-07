#!/bin/bash
# Usage: ./train.sh <output_dir>
output_dir=${1:-./output}
mkdir -p "$output_dir"
python -u train.py --output_dir "$output_dir" --config_path ./config.json
