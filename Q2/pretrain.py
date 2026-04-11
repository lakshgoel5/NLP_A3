"""
pretrain.py — Continued CLM pretraining on unlabeled Wikipedia text.

Used for both "pre-adapt" and "post-adapt" pipelines:
  - pre_adapt:  run before RE fine-tuning on the base Qwen model
  - post_adapt: run after RE fine-tuning, starting from the LoRA-merged checkpoint

Saves a plain AutoModelForCausalLM checkpoint that train.py can load via
--pretrained_dir.
"""

import os
import json
import random
import argparse
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def fetch_wikipedia_texts(languages, max_samples, seed):
    """Stream Wikipedia articles for the given language codes and return a list of strings."""
    texts = []
    for lang in languages:
        print(f"  [MLM] Fetching Wikipedia '{lang}' (up to {max_samples} articles) …")
        ds = load_dataset(
            "wikimedia/wikipedia",
            f"20231101.{lang}",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        count = 0
        for item in ds:
            if item.get("text", "").strip():
                texts.append(item["text"])
                count += 1
            if count >= max_samples:
                break
        print(f"  [MLM] Collected {count} articles for '{lang}'")
    random.Random(seed).shuffle(texts)
    return texts


class TextDataset(Dataset):
    """Tokenize raw texts and chunk into fixed-length windows for CLM."""

    def __init__(self, texts, tokenizer, max_length):
        self.examples = []
        for text in tqdm(texts, desc="Tokenizing corpus", leave=False):
            if not text.strip():
                continue
            token_ids = tokenizer(
                text,
                truncation=False,
                add_special_tokens=False,
                return_tensors=None,
            )["input_ids"]
            # Slide a fixed window over the token sequence
            for i in range(0, len(token_ids), max_length):
                chunk = token_ids[i : i + max_length]
                if len(chunk) < 16:  # discard very short tails
                    continue
                self.examples.append(torch.tensor(chunk, dtype=torch.long))
        print(f"  [MLM] {len(self.examples)} chunks from {len(texts)} articles")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch, pad_id):
    max_len = max(x.size(0) for x in batch)
    input_ids_padded, attn_mask_padded = [], []
    for x in batch:
        L = x.size(0)
        pad_len = max_len - L
        input_ids_padded.append(
            torch.cat([x, torch.full((pad_len,), pad_id, dtype=torch.long)])
        )
        attn_mask_padded.append(
            torch.cat([torch.ones(L, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
        )
    return {
        "input_ids": torch.stack(input_ids_padded),
        "attention_mask": torch.stack(attn_mask_padded),
    }


# ---------------------------------------------------------------------------
# Model loading (handles both plain and PEFT checkpoints)
# ---------------------------------------------------------------------------

def load_model_for_pretraining(model_path):
    """
    Load a causal-LM model for continued pretraining.

    If the checkpoint is a PEFT / LoRA adapter (contains adapter_config.json),
    the adapters are merged into the base weights and unloaded so that the
    resulting model is a plain AutoModelForCausalLM — making it easy to save
    and reload later with AutoModelForCausalLM.from_pretrained.
    """
    model_name = model_path if model_path else "Qwen/Qwen2.5-1.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    adapter_cfg = os.path.join(model_name, "adapter_config.json") if model_path else None
    if adapter_cfg and os.path.isfile(adapter_cfg):
        # PEFT / LoRA adapter — merge before training
        from peft import AutoPeftModelForCausalLM

        print(f"  [MLM] Detected LoRA adapter at '{model_name}' — merging weights …")
        peft_model = AutoPeftModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )
        model = peft_model.merge_and_unload()
        print("  [MLM] Adapters merged and unloaded.")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )

    return tokenizer, model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_clm(model, tokenizer, loader, device, epochs, lr, accumulation_steps, warmup_ratio):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    if device.type == "cuda":
        scaler = GradScaler("cuda")
    else:
        scaler = GradScaler("cpu", enabled=False)

    total_steps = max(1, (len(loader) // accumulation_steps) * epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        step = 0

        pbar = tqdm(loader, desc=f"  [MLM] Epoch {epoch}/{epochs}", leave=True)
        for step, batch in enumerate(pbar, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )
                loss_val = out.loss / accumulation_steps

            scaler.scale(loss_val).backward()

            if step % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss_val.item() * accumulation_steps
            pbar.set_postfix(
                loss=f"{total_loss / step:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        # Flush any remaining partial accumulation window
        if step % accumulation_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / max(1, step)
        print(f"\n  [MLM] Epoch {epoch}/{epochs} — avg loss: {avg_loss:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Continued CLM pretraining on unlabeled Wikipedia text."
    )
    p.add_argument("--output_dir", default="./mlm_adapted",
                   help="Where to save the adapted model checkpoint.")
    p.add_argument("--start_from", default=None,
                   help="Base model path or prior checkpoint (plain or LoRA).")
    p.add_argument("--languages", default="hi,kn",
                   help="Comma-separated Wikipedia language codes, e.g. 'hi,kn'.")
    p.add_argument("--max_samples", type=int, default=5000,
                   help="Max Wikipedia articles to fetch per language.")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--accumulation_steps", type=int, default=4)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"[MLM Pretraining] Device: {device}")

    langs = [l.strip() for l in args.languages.split(",") if l.strip()]
    print(f"[MLM Pretraining] Languages: {langs}")

    # Load model
    print(f"[MLM Pretraining] Loading model from: {args.start_from or 'Qwen/Qwen2.5-1.5B'}")
    tokenizer, model = load_model_for_pretraining(args.start_from)
    model = model.to(device)

    # Fetch corpus
    texts = fetch_wikipedia_texts(langs, args.max_samples, args.seed)

    # Build dataset & loader
    dataset = TextDataset(texts, tokenizer, args.max_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, pad_id=tokenizer.pad_token_id),
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    # Train
    train_clm(
        model, tokenizer, loader, device,
        epochs=args.epochs,
        lr=args.lr,
        accumulation_steps=args.accumulation_steps,
        warmup_ratio=args.warmup_ratio,
    )

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n[MLM Pretraining] Saved adapted model → {args.output_dir}")


if __name__ == "__main__":
    main()
