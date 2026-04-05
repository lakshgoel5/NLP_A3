import os
import json
import copy
import argparse

import torch
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

from dataset import ALL_RELATION_LABELS, build_label_map, build_inverse_map,mark_entities, NUM_CLASSES
from model import RelationClassifier, SPECIAL_TOKENS

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", required=True, choices=["en", "hi", "kn"])
    p.add_argument("--test_file", required=True)
    p.add_argument("--output_dir", default="./output")
    p.add_argument("--config_path", default="./config.json")
    args = p.parse_args()

    with open(args.config_path) as f:
        cfg = json.load(f)

    max_len = cfg["max_len"]
    entity_repr = cfg["entity_repr"]

    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}\n")

if __name__ == "__main__":
    main()