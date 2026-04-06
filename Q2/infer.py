import os
import json
import difflib
import argparse

import torch
from tqdm import tqdm

from dataset import ALL_RELATION_LABELS
from model import load_trained_model


def make_prompt(sent_text, em1, em2):
    return (
        f'Entity 1: {em1}\n'
        f'Entity 2: {em2}\n'
        f'Sentence: {sent_text}\n'
        f'Relation: '
    )


def closest_label(pred, valid_labels):
    pred = pred.strip()
    if pred in valid_labels:
        return pred
    # Try prefix match (model may output extra text after the label)
    for label in valid_labels:
        if pred.startswith(label):
            return label
    # Fuzzy match as fallback
    matches = difflib.get_close_matches(pred, valid_labels, n=1, cutoff=0.0)
    return matches[0] if matches else "NA"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", required=True, choices=["en", "hi", "kn", "or", "tcy"])
    p.add_argument("--test_file", required=True)
    p.add_argument("--output_dir", default="./output")
    p.add_argument("--config_path", default="./config.json")
    args = p.parse_args()

    with open(args.config_path) as f:
        cfg = json.load(f)

    max_new_tokens = cfg.get("max_new_tokens", 48)
    max_len = cfg.get("max_len", 256)

    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    tokenizer, model = load_trained_model(args.output_dir, device)

    # Load forward map for non-English (English label -> indic label)
    forward_map = None
    if args.lang != "en":
        map_file = f"../sft_dataset/{args.lang}_map.json"
        if os.path.isfile(map_file):
            with open(map_file, "r", encoding="utf-8") as f:
                forward_map = json.load(f)

    results = []
    with open(args.test_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in tqdm(lines, desc=f"Inferring [{args.lang}]"):
        data = json.loads(line)
        sent = data["sentText"]
        out_entry = {
            "articleId": data.get("articleId", ""),
            "sentId": data.get("sentId", ""),
            "sentText": sent,
            "relationMentions": [],
        }

        for rm in data.get("relationMentions", []):
            em1 = rm["em1Text"]
            em2 = rm["em2Text"]

            prompt = make_prompt(sent, em1, em2)
            enc = tokenizer(
                prompt,
                max_length=max_len,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            with torch.no_grad():
                out_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode only the newly generated tokens
            new_tokens = out_ids[0][input_ids.shape[1]:]
            pred_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            # Snap to nearest valid English label
            en_label = closest_label(pred_text, ALL_RELATION_LABELS)

            # Convert to indic label for non-English outputs
            if forward_map is not None:
                final_label = forward_map.get(en_label, en_label)
            else:
                final_label = en_label

            out_entry["relationMentions"].append({
                "em1Text": em1,
                "em2Text": em2,
                "label": final_label,
            })

        results.append(out_entry)

    out_file = os.path.join(args.output_dir, f"output_{args.lang}.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} predictions to {out_file}")


if __name__ == "__main__":
    main()
