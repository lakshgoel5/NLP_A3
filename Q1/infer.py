import os
import json
import copy
import argparse
from functools import partial
from torch.utils.data import Dataset, DataLoader

try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast

import torch
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

from dataset import ALL_RELATION_LABELS, build_label_map, NUM_CLASSES, mark_entities, find_token_position
from model import RelationClassifier, SPECIAL_TOKENS

class InferenceDataset(Dataset):
    def __init__(self, test_entries, tokeniser, max_len):
        self.tokeniser = tokeniser
        self.max_len = max_len

        self.flat_data = []

        for rec_idx, entry in enumerate(test_entries):
            sentence = entry["sentText"]
            for men_idx, rm in enumerate(entry.get("relationMentions", [])):
                marked = mark_entities(sentence, rm["em1Text"], rm["em2Text"])
                self.flat_data.append((rec_idx, men_idx, marked))

    def __len__(self):
        return len(self.flat_data)

    def __getitem__(self, idx):
        rec_idx, men_idx, marked = self.flat_data[idx]

        if marked is None:
            marked = ""

        tokens = self.tokeniser(
            marked,
            max_length=self.max_len,
            truncation=True,
            padding=False, # Because sentences in a batch have different lengths. collate later efficiently
            return_tensors=None, # return plain Python lists, not torch tensors # Because sentences in a batch have different lengths.
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
        input_ids.append(x["input_ids"] + [pad_id] * pad_len)
        attention_masks.append(x["attention_mask"] + [0] * pad_len) #Padding attention mask here as we didnt set padding=true in self.tokeniser in dataset class
        rec_idxs.append(x["rec_idx"])
        men_idxs.append(x["men_idx"])

    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_masks),
        "rec_idx": rec_idxs,
        "men_idx": men_idxs,
    }

def load_model(output_dir, entity_repr, num_classes, device):
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModel.from_pretrained("Qwen/Qwen2.5-1.5B", torch_dtype=torch.float16) #pulls the full frozen base weights from the Hub (or cache), reconstructing the original Qwen model.
    base.resize_token_embeddings(len(tokenizer)) # The order matters: resize before loading LoRA, because the saved adapter weights were shaped against the already-resized model.
    base = PeftModel.from_pretrained(base, output_dir) #injects the trained LoRA matrices (A and B) back into the frozen base

    model = RelationClassifier(base, base.config.hidden_size, num_classes, entity_repr)
    model.classifier.load_state_dict(torch.load(os.path.join(output_dir, "classifier_head.pt"), map_location=device))

    model = model.to(device)
    model.eval()
    return tokenizer, model

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
    # No use of cfg after this

    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}\n")

    label2id, id2label = build_label_map() # Just to get possible outputs

    # For non-English, load a mapping: English label → target-language label
    fwd_map = None
    if args.lang != "en":
        map_path = f"../sft_dataset/{args.lang}_map.json"
        if os.path.isfile(map_path):
            with open(map_path, "r", encoding="utf-8") as f:
                fwd_map = json.load(f)

            print(f"Loaded label map for '{args.lang}' from {map_path}")
        else:
            print(f"Warning: no label map found at {map_path}; outputting English labels.")

    tokenizer, model = load_model(args.output_dir, entity_repr, NUM_CLASSES, device)

    #-----Read test file-----
    test_entries = []
    with open(args.test_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_entries.append(json.loads(line))

    #----------Dataloader------------
    # no label, therefore write a new data loader
    dataset = InferenceDataset(test_entries, tokenizer, max_len)
    collate = partial(infer_collate_fn, pad_id=tokenizer.pad_token_id)

    inf_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate,
        num_workers=4,
        pin_memory=True,
    )

    #-------------special token IDs-------------
    e1_id = tokenizer.convert_tokens_to_ids("[E1]")
    e1_end_id = tokenizer.convert_tokens_to_ids("[/E1]")
    e2_id = tokenizer.convert_tokens_to_ids("[E2]")
    e2_end_id = tokenizer.convert_tokens_to_ids("[/E2]")

    #---------Initialise result dict------------
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

    #------------batch inference------------
    for batch in tqdm(inf_loader, desc="Inference"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        ids = input_ids.tolist() # Because input_ids was already batched, input_ids.tolist() returned a list of lists.

        e1_pos = torch.tensor([find_token_position(row, e1_id) or 0 for row in ids], dtype=torch.long, device=device)
        e1_end_pos = torch.tensor([find_token_position(row, e1_end_id) or 0 for row in ids], dtype=torch.long, device=device)
        e2_pos = torch.tensor([find_token_position(row, e2_id) or 0 for row in ids], dtype=torch.long, device=device)
        e2_end_pos = torch.tensor([find_token_position(row, e2_end_id) or 0 for row in ids], dtype=torch.long, device=device)

        use_amp = device.type in ("cuda", "mps")

        with torch.no_grad():
            try:
                # Modern PyTorch (2.x) expects the device_type argument
                ctx = autocast(device_type=device.type, enabled=use_amp)
            except TypeError:
                # Legacy PyTorch (1.x) fallback
                ctx = autocast(enabled=use_amp)

            with ctx:
                logits = model(input_ids, attention_mask, e1_pos, e1_end_pos, e2_pos, e2_end_pos)
        
        preds = logits.argmax(dim=-1).cpu().tolist()

        #--------create results dict--------
        for i, (rec_idx, men_idx) in enumerate(zip(batch["rec_idx"], batch["men_idx"])):
            en_label = id2label[preds[i]] #What label is in english
            label = fwd_map.get(en_label, en_label) if fwd_map else en_label
            results[rec_idx]["relationMentions"][men_idx]["label"] = label

    #---------write output-----------
    output_path = os.path.join(args.output_dir, f"Q1_{args.lang}.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} predictions to {output_path}")

if __name__ == "__main__":
    main()