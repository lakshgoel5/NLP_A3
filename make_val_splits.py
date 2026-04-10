"""
Reproduce the exact val splits used during Q1/Q2 training (seed=42, val_ratio=0.2).
Writes val split files to --out_dir and prints the output paths.
Usage:
    python make_val_splits.py --files a.jsonl b.jsonl --out_dir ./val_splits
"""
import os
import random
import argparse


def split_indic_file(file_path, out_dir, val_ratio=0.2, seed=42):
    """Same logic as train.py — returns the val split path."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    random.Random(seed).shuffle(lines)
    n_val = max(1, int(len(lines) * val_ratio))
    stem = os.path.splitext(os.path.basename(file_path))[0]
    val_path = os.path.join(out_dir, f"{stem}_val.jsonl")
    with open(val_path, "w", encoding="utf-8") as f:
        f.writelines(lines[:n_val])
    return val_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--files", nargs="+", required=True, help="Input jsonl files to split")
    p.add_argument("--out_dir", required=True, help="Directory to write val split files")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for fpath in args.files:
        if not os.path.isfile(fpath):
            print(f"SKIP (not found): {fpath}")
            continue
        val_path = split_indic_file(fpath, args.out_dir)
        print(val_path)


if __name__ == "__main__":
    main()
