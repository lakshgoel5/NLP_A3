import os
import json
import random
import argparse

import torch
try:
    from torch.amp import GradScaler, autocast  # PyTorch >= 1.10
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

MODEL_NAME = "Qwen/Qwen2.5-1.5B"

LANG_CONFIGS = {
    "hi":  "20231101.hi",
    "kn":  "20231101.kn",
    "or":  "20231101.or",
    "tcy": "20231101.tcy",
}


def load_base_model(pretrained_dir=None):
    model_name = MODEL_NAME
    if pretrained_dir:
        model_name = pretrained_dir
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Qwen2.5 uses eos as pad by default; set explicitly
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16, # save memory # DESIGN (bfloat16 might be stable on A100)
    )

    return tokenizer, model

# freeze everything, inject tiny trainable matrices at specific places
def lora(model, lora_rank, lora_alpha, lora_dropout):
    # output = W·x + (lora_alpha/r) × B·A·x
    lora_config = LoraConfig(
        r=lora_rank, #DESIGN try 8,16,32,64
        lora_alpha=lora_alpha, #DESIGN Try sqrt(2)*r, 2*r
        lora_dropout=lora_dropout, #DESIGN 0.05-0.1
        # DESIGN: Q1 only used q_proj+v_proj (feature extraction), generation benefits from k_proj+o_proj too
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", #DESIGN "lora_only" or "all" can sometimes improve performance
        task_type=TaskType.CAUSAL_LM, #TaskType.CAUSAL_LM is a configuration constant used in the Hugging Face PEFT (Parameter-Efficient Fine-Tuning) library to specify that a model is being used for Causal Language Modeling.
    )

    # get_peft_model() is a Hugging Face PEFT library function that wraps a pre-trained base model with a PeftConfig to create a trainable PeftModel
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def load_wiki_texts(lang_codes, max_samples_per_lang, seed=42):
    all_texts = []
    for lang in lang_codes:
        cfg = LANG_CONFIGS.get(lang)
        if cfg is None:
            print(f"Unknown lang: {lang}, skipping")
            continue
        print(f"Loading Wikipedia ({lang}) ...")
        try:
            data_set = load_dataset("wikimedia/wikipedia", cfg, split="train", streaming=True)
            texts = []
            for example in data_set:
                if len(example["text"].strip()) > 200:  # skip stubs
                    texts.append(example["text"])
                if len(texts) >= max_samples_per_lang:
                    break
            print(f"  {lang}: {len(texts)} articles")
            all_texts.extend(texts)
        except Exception as e:
            print(f"  Failed to load {lang}: {e}")
    random.Random(seed).shuffle(all_texts)
    return all_texts

class IndicCLMDataset(Dataset):
    def __init__(self, texts, tokenizer, chunk_size=256):
        self.chunks = []
        self.build(texts, tokenizer, chunk_size)

    def build(self, texts, tokenizer, chunk_size):
        print(f"  Tokenising {len(texts)} articles")
        all_ids = []
        # Concatenate all articles into one long token stream
        for text in tqdm(texts, leave=False):
            ids = tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
            all_ids.extend(ids)
            all_ids.append(tokenizer.eos_token_id)

        # Slice that stream into fixed-size chunks
        for start in range(0, len(all_ids) - chunk_size, chunk_size):
            chunk = all_ids[start : start + chunk_size]
            self.chunks.append(torch.tensor(chunk, dtype=torch.long))

        print(f"  tokenised into {len(self.chunks)} chunks of {chunk_size} tokens")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        ids = self.chunks[idx]
        return {"input_ids": ids, "labels": ids.clone()}

def collate_function(batch):
    input_ids = torch.stack([x["input_ids"] for x in batch])
    labels = torch.stack([x["labels"] for x in batch])
    return {"input_ids": input_ids, "labels": labels}

def train(model, loader, optimizer, scheduler, scaler, device, epochs, accumulation):
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"CPT Epoch {epoch}/{epochs}")
        for step, batch in enumerate(pbar, 1):
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                out = model(input_ids=input_ids, labels=labels)
                loss = out.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if step % accumulation != 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            pbar.set_postfix(
                loss=f"{total_loss / step:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        print(f"Epoch {epoch} avg loss: {total_loss/len(loader):.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="./output")
    p.add_argument("--config_path", default="./pretrain_config.json")
    p.add_argument("--pretrained_dir", default=None, help="Path to CPT-adapted model from pretrain.py")

    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    #------Hyper parameters--------
    with open(args.config_path) as f:
        config = json.load(f)

    epochs = config["epochs"]
    lr = config["lr"]
    weight_decay = config["weight_decay"]
    warmup_ratio = config["warmup_ratio"]
    accumulation = config.get("accumulation_steps", 1)
    batch_size = config["batch_size"]

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    # ---------- Model ----------
    tokenizer, base_model = load_base_model(args.pretrained_dir)
    # this is only model now
    model = lora(base_model, config["lora_rank"], config["lora_alpha"], config["lora_dropout"])
    model = model.to(device)


    # ---------- Data ----------
    langs = config["langs"]
    texts = load_wiki_texts(langs, config["max_samples_per_lang"], seed=config["seed"])
    dataset = IndicCLMDataset(texts, tokenizer, chunk_size=config["chunk_size"])
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_function,
        num_workers=4,
        pin_memory=True,
    )

    # ---------Optimiser---------
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)

    #-----scaler for float16 error------
    scaler = GradScaler("cpu", enabled=False) if device.type not in ("cuda",) else GradScaler("cuda")

    # ---------Scheduler----------
    total_steps = (len(train_loader) // accumulation) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ---------train------------
    print(f"\nStarting CPT: {epochs} epoch(s), {total_steps} steps, {len(dataset)} chunks from {len(texts)} articles")
    train(model, train_loader, optimizer, scheduler, scaler, device, epochs, accumulation)

    # ---- Merge LoRA weights and save -----
    print("\nMerging LoRA weights into base model")
    merged = model.merge_and_unload()

    # To load a tokenizer successfully from a local directory, that folder must contain the necessary configuration and vocabulary files, typically saved using the save_pretrained() method
    # The directory should include files like 
    # tokenizer_config.json, 
    # special_tokens_map.json, 
    # and the specific vocabulary file for that model (e.g., vocab.txt or tokenizer.json).
    merged.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved adapted model → {args.output_dir}")
    print("\nNow run Q1/Q2 train.py with  --pretrained_dir", args.output_dir)


if __name__ == "__main__":
    main()