import os
import json
import difflib
import argparse
from functools import partial
from torch.utils.data import Dataset, DataLoader

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


class InferenceDataset(Dataset):
    def __init__(self, test_entries, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.flat_data = []

        for rec_idx, entry in enumerate(test_entries):
            sentence = entry["sentText"]
            for men_idx, rm in enumerate(entry.get("relationMentions", [])):
                prompt = make_prompt(sentence, rm["em1Text"], rm["em2Text"])
                self.flat_data.append((rec_idx, men_idx, prompt))

    def __len__(self):
        return len(self.flat_data)

    def __getitem__(self, idx):
        rec_idx, men_idx, prompt = self.flat_data[idx]

        tokens = self.tokenizer(
            prompt,
            max_length=self.max_len,
            truncation=True,
            padding=False,  # Because prompts in a batch have different lengths. collate handles it.
            return_tensors=None, # Return plain Python lists, not torch tensors.
        )

        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "rec_idx": rec_idx,
            "men_idx": men_idx,
        }


def infer_collate_fn(batch, pad_id):
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids, attention_masks = [], []
    rec_idxs, men_idxs = [], []

    for x in batch:
        pad_len = max_len - len(x["input_ids"])
        # Left-pad: generation requires real tokens at the RIGHT end so the model
        # continues from the last meaningful token, not from padding.
        input_ids.append([pad_id] * pad_len + x["input_ids"])
        attention_masks.append([0] * pad_len + x["attention_mask"]) # Padding attention mask here as we didn't set padding=True in InferenceDataset
        rec_idxs.append(x["rec_idx"])
        men_idxs.append(x["men_idx"])

    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_masks),
        "rec_idx": rec_idxs,
        "men_idx": men_idxs,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", required=True, choices=["en", "hi", "kn", "or", "tcy"])
    p.add_argument("--test_file", required=True)
    p.add_argument("--output_dir", default="./output")
    p.add_argument("--config_path", default="./config.json")
    args = p.parse_args()

    with open(args.config_path) as f:
        cfg = json.load(f)

    max_len = cfg["max_len"]
    max_new_tokens = cfg.get("max_new_tokens", 48)
    # No use of cfg after this

    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}\n")

    # For non-English, load a mapping: English label -> target-language label
    fwd_map = None
    if args.lang != "en":
        map_path = f"../sft_dataset/{args.lang}_map.json"
        if os.path.isfile(map_path):
            with open(map_path, "r", encoding="utf-8") as f:
                fwd_map = json.load(f)
            print(f"Loaded label map for '{args.lang}' from {map_path}")
        else:
            print(f"Warning: no label map found at {map_path}; outputting English labels.")

    tokenizer, model = load_trained_model(args.output_dir, device)
    # load_trained_model already sets tokenizer.padding_side = "left" for generation

    # ----- Read test file -----
    test_entries = []
    with open(args.test_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_entries.append(json.loads(line))

    # ---------- DataLoader ----------
    dataset = InferenceDataset(test_entries, tokenizer, max_len)
    collate = partial(infer_collate_fn, pad_id=tokenizer.pad_token_id)

    inf_loader = DataLoader(
        dataset,
        batch_size=16,  # larger than train batch_size is fine: no gradients stored
        shuffle=False,
        collate_fn=collate,
        num_workers=4,
        pin_memory=True,
    )

    # --------- Initialise results dict ---------
    # Pre-fill every mention with "NA" so that even if something goes wrong
    # in generation, every mention still has a valid label.
    results = []
    for record in test_entries:
        results.append({
            "articleId": record.get("articleId", ""),
            "sentId":    record.get("sentId", ""),
            "sentText":  record["sentText"],
            "relationMentions": [
                {"em1Text": rm["em1Text"], "em2Text": rm["em2Text"], "label": "NA"}
                for rm in record.get("relationMentions", [])
            ],
        })

    # ----------- batch inference -----------
    for batch in tqdm(inf_loader, desc="Inference"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        prompt_len = input_ids.shape[1]  # padded prompt length; generated tokens come after this

        with torch.no_grad():
            out_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (strip the left-padded prompt prefix)
        new_tokens = out_ids[:, prompt_len:]  # [batch, gen_len]
        pred_texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        # -------- fill results dict --------
        for i, (rec_idx, men_idx) in enumerate(zip(batch["rec_idx"], batch["men_idx"])):
            # Snap free-form generated text to the nearest valid English label
            en_label = closest_label(pred_texts[i], ALL_RELATION_LABELS)
            # Convert to indic label for non-English outputs
            label = fwd_map.get(en_label, en_label) if fwd_map else en_label
            results[rec_idx]["relationMentions"][men_idx]["label"] = label

    # --------- write output ---------
    output_path = os.path.join(args.output_dir, f"output_{args.lang}.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} predictions to {output_path}")


if __name__ == "__main__":
    main()
