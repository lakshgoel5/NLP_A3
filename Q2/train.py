import os
import json
import random
import shutil
import tempfile
import difflib
import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader
try:
    from torch.amp import GradScaler, autocast  # PyTorch >= 1.10
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from dataset import SFTDataset, ALL_RELATION_LABELS, build_label_map, collate_fn
from model import load_base_model, lora


def split_indic_file(file_path, tmp_dir, val_ratio=0.2, seed=42):
    """Shuffle and split a jsonl file. Returns (train_path, val_path) in tmp_dir."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    random.Random(seed).shuffle(lines)
    n_val = max(1, int(len(lines) * val_ratio))
    stem = os.path.splitext(os.path.basename(file_path))[0]
    train_path = os.path.join(tmp_dir, f"{stem}_80pct.jsonl")
    val_path   = os.path.join(tmp_dir, f"{stem}_20pct.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        f.writelines(lines[n_val:])
    with open(val_path, "w", encoding="utf-8") as f:
        f.writelines(lines[:n_val])
    return train_path, val_path


def closest_label(pred, valid_labels):
    pred = pred.strip()
    if pred in valid_labels:
        return pred
    for label in valid_labels:
        if pred.startswith(label):
            return label
    matches = difflib.get_close_matches(pred, valid_labels, n=1, cutoff=0.0)
    return matches[0] if matches else "NA"


def make_prompt(sent_text, em1, em2):
    return (
        f'Entity 1: {em1}\n'
        f'Entity 2: {em2}\n'
        f'Sentence: {sent_text}\n'
        f'Relation: '
    )


def evaluate_f1(model, tokenizer, val_files, device, max_len, max_new_tokens, max_samples=500, file_to_inv_map=None):
    label2id, id2label = build_label_map()
    na_id = label2id["NA"]
    non_na_ids = [i for i in id2label if i != na_id]
    file_to_inv_map = file_to_inv_map or {}

    all_true, all_pred = [], []

    model.eval()
    with torch.no_grad():
        for val_file in val_files:
            if not os.path.isfile(val_file):
                continue
            inv_map = file_to_inv_map.get(val_file)
            with open(val_file, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]

            # Subsample to keep eval fast
            if max_samples and len(lines) > max_samples:
                step = len(lines) // max_samples
                lines = lines[::step][:max_samples]

            for line in tqdm(lines, desc="Val F1", leave=False):
                data = json.loads(line)
                sent = data["sentText"]
                for rm in data.get("relationMentions", []):
                    true_label = rm.get("label")
                    if inv_map is not None:
                        true_label = inv_map.get(true_label, true_label)
                    if true_label is None or true_label not in label2id:
                        continue
                    em1, em2 = rm["em1Text"], rm["em2Text"]

                    prompt = make_prompt(sent, em1, em2)
                    enc = tokenizer(
                        prompt,
                        max_length=max_len,
                        truncation=True,
                        return_tensors="pt",
                    )
                    input_ids = enc["input_ids"].to(device)
                    attn_mask = enc["attention_mask"].to(device)

                    out_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    new_tokens = out_ids[0][input_ids.shape[1]:]
                    pred_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    pred_label = closest_label(pred_text, ALL_RELATION_LABELS)

                    all_true.append(label2id[true_label])
                    all_pred.append(label2id[pred_label])

    if not all_true:
        return 0.0, 0.0

    macro = f1_score(all_true, all_pred, labels=non_na_ids, average="macro", zero_division=0)
    micro = f1_score(all_true, all_pred, labels=non_na_ids, average="micro", zero_division=0)
    return macro, micro


# Evaluated on English, Hindi, Kannada, Oriya, Telugu NYT-10 test set
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="./output")
    p.add_argument("--config_path", default="./config.json")
    p.add_argument("--pretrained_dir", default=None, help="Path to CPT-adapted model from pretrain.py")
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

    # ---------- Hyperparameters ----------
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    lr = config["lr"]
    max_len = config["max_len"]
    max_new_tokens = config.get("max_new_tokens", 48)
    weight_decay = config.get("weight_decay", 0.01)
    accumulation = config.get("accumulation_steps", 1)
    warmup_ratio = config["warmup_ratio"]
    indic_repeat = config.get("indic_repeat", 1)

    # ---------- Data Paths ----------
    INDIC_FILES = [
        ("../sft_dataset/hi_train.jsonl",  "../sft_dataset/hi_map.json"),
        ("../sft_dataset/kn_train.jsonl",  "../sft_dataset/kn_map.json"),
        ("../sft_dataset/or_train.jsonl",  "../sft_dataset/or_map.json"),
        ("../sft_dataset/tcy_val.jsonl",   "../sft_dataset/tcy_map.json"),
    ]

    tmp_dir = tempfile.mkdtemp(prefix="q2_splits_")
    train_files = ["../en_sft_dataset/train.jsonl"]
    val_files   = ["../en_sft_dataset/valid.jsonl"]
    train_maps  = []
    val_inv_maps = {}  # val_file_path -> inv_map dict, for evaluate_f1

    for data_file, map_file in INDIC_FILES:
        if not os.path.isfile(data_file):
            continue
        tr, vl = split_indic_file(data_file, tmp_dir)
        train_files.append(tr)
        val_files.append(vl)
        if os.path.isfile(map_file):
            train_maps.append(map_file)
            with open(map_file, "r", encoding="utf-8") as mf:
                fwd = json.load(mf)
            val_inv_maps[vl] = {v: k for k, v in fwd.items()}

    train_files = [f for f in train_files if os.path.isfile(f)]
    val_files   = [f for f in val_files   if os.path.isfile(f)]
    print(f"Training files: {train_files}")
    print(f"Training maps: {train_maps}\n")

    # ---------- Model ----------
    tokenizer, base_model = load_base_model(args.pretrained_dir)
    # this is only model now
    model = lora(base_model, config["lora_rank"], config["lora_alpha"], config["lora_dropout"])

    # Removed relational classifier class
    model = model.to(device)

    # ---------- Dataset & Loader ----------
    collate_function = partial(collate_fn, pad_id=tokenizer.pad_token_id)

    train_dataset=SFTDataset(
        file_paths=train_files,
        tokenizer=tokenizer,
        max_length=max_len,
        map_paths=train_maps,
        indic_repeat=indic_repeat,
    )

    train_loader=DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_function,
        num_workers=4,
        pin_memory=True,
    )

    # ------loss------
    # Internally, the model calculates the loss for every token (including padding tokens) and then masks out the padding tokens before averaging. 

    # ---------Optimiser---------
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)

    #-----scaler for float16 error------
    scaler = GradScaler("cpu", enabled=False) if device.type not in ("cuda",) else GradScaler("cuda")

    # ---------Scheduler----------
    total_steps = (len(train_loader) // accumulation) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    #DESIGN alternatives: linear, polynomial, constant-with-warmup
    # num_cycles=0.5 = one half-cosine; try CosineAnnealingWarmRestarts for multi-cycle
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ---------train------------
    best_macro_f1 = 0
    
    print(f"Starting training: {epochs} epochs, {total_steps} total steps, {warmup_steps} warmup steps\n")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=True)
        for step, batch in enumerate(pbar, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss_val = output.loss / accumulation

            scaler.scale(loss_val).backward()

            if step % accumulation == 0:
                # We must bring the gradients back to their real size before we look at them.
                scaler.unscale_(optimizer)

                # If a gradient is too large (above 1.0), it forcibly shrinks it. This keeps the training stable and prevents the "NaN" (Not a Number) error.
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,  #DESIGN: 0.5–2.0
                )

                # This is the "step" where the model actually updates its weights.
                scaler.step(optimizer)

                # This adjusts the scale factor for the next batch. If training is going smoothly, it might increase the multiplier; if it sees overflows, it decreases it.
                scaler.update()

                # This moves the learning rate to the next step (e.g., from 0.0001 to 0.00011).
                scheduler.step()

                # This resets the gradients to zero so they don't accumulate from the previous batch.
                optimizer.zero_grad()

            total_loss += loss_val.item() * accumulation
            pbar.set_postfix(loss=f"{total_loss / step:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        # Flush any remaining gradients from a partial accumulation window
        if step % accumulation != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)

        # Switch to left-padding for generation
        tokenizer.padding_side = "left"
        macro_f1, micro_f1 = evaluate_f1(model, tokenizer, val_files, device, max_len, max_new_tokens, file_to_inv_map=val_inv_maps)
        tokenizer.padding_side = "right"

        print(f"\nEpoch {epoch}/{epochs} | Train Loss: {avg_loss:.4f} | Val Macro-F1: {macro_f1:.4f} | Val Micro-F1: {micro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"\nSaved best model (macro_f1={macro_f1:.4f})")

    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()