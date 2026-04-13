#!/bin/bash
ZIP_NAME="2023CS10848_A3_NLP.zip"
TEMP_DIR="temp_sub"
INNER_DIR="$TEMP_DIR/starter_code"

rm -rf "$TEMP_DIR"
mkdir -p "$INNER_DIR"

# Copy required files
cp -r Q1 Q2 Q3 "$INNER_DIR/"
cp eval.py icl_starter.py requirements.txt README.md "$INNER_DIR/"

# Remove data files, caches, model outputs
find "$INNER_DIR" -name "*.jsonl" -type f -delete
find "$INNER_DIR" -name "*.pt"    -type f -delete
find "$INNER_DIR" -name "*.bin"   -type f -delete
find "$INNER_DIR" -name "*.safetensors" -type f -delete
find "$INNER_DIR" -name "__pycache__" -type d -exec rm -rf {} +
find "$INNER_DIR" -name "output"  -type d -exec rm -rf {} +
find "$INNER_DIR" -name "val_splits" -type d -exec rm -rf {} +

cd "$TEMP_DIR"
zip -r "../$ZIP_NAME" .
cd ..

rm -rf "$TEMP_DIR"
echo "Submission zip created: $ZIP_NAME"
echo "Contents:"
unzip -l "$ZIP_NAME" | head -40
