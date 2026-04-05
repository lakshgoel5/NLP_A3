import os
import json
import argparse
import torch

from dataset import build_label_map, NUM_CLASSES
from model import load_base_model, lora

# Evaluated on English, Hindi, Kannada NYT-10 test set
def main():
    p = argparser.ArgumentParser()
    p.add_argument("--output_dir", default="./output")
    p.add_argument("--config_path", default="./config.json")
    args = p.parse_args()

    with open(args.config_path, "r") as f:
        config = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}\n")

    file_paths = [
        "../en_sft_dataset/train.jsonl",
        "../sft_dataset/hi_train.jsonl",
        "../sft_dataset/kn_train.jsonl",
    ]

    map_paths = [
        "../sft_dataset/hi_map.json",
        "../sft_dataset/kn_map.json",
    ]

    train_files = [f for f in file_paths if os.path.isfile(f)]
    train_maps = [f for f in map_paths if os.path.isfile(f)]
    print(f"Training files: {train_files}\n")
    print(f"Training maps: {train_maps}\n")

    label2id, id2label = build_label_map()
    num_classes = NUM_CLASSES

    tokenizer, base_model = load_base_model()

    base_model = lora(base_model, config["lora_rank"], config["lora_alpha"], config["lora_dropout"])
    



if __name__ == "__main__":
    main()