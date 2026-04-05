import os
import json
import argparse
import torch

from dataset import build_label_map, NUM_CLASSES, RDataset
from model import load_base_model, lora

from transformers import get_cosine_schedule_with_warmup


# Evaluated on English, Hindi, Kannada NYT-10 test set
def main():
    p = argparse.ArgumentParser()
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

    #----------Hyperparameters-----------
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    lr = config["learning_rate"]
    max_len = config["max_len"]
    weight_decay = config.get("weight_decay", 0.01)
    accumulation = config.get("accumulation_steps", 1)

    warmup_ratio = config["warmup_ratio"]

    # ---------Data Paths--------
    file_paths = [
        "../en_sft_dataset/train.jsonl",
        "../sft_dataset/hi_train.jsonl",
        "../sft_dataset/kn_train.jsonl",
    ]
    val_file_path = "../en_sft_dataset/valid.jsonl"

    map_paths = [
        "../sft_dataset/hi_map.json",
        "../sft_dataset/kn_map.json",
    ]

    train_files = [f for f in file_paths if os.path.isfile(f)]
    train_maps = [f for f in map_paths if os.path.isfile(f)]
    print(f"Training files: {train_files}\n")
    print(f"Training maps: {train_maps}\n")

    # -------labels--------
    label2id, id2label = build_label_map()
    num_classes = NUM_CLASSES

    #--------model-------
    tokenizer, base_model = load_base_model()
    base_model = lora(base_model, config["lora_rank"], config["lora_alpha"], config["lora_dropout"])

    #-------dataset--------
    # A DataLoader requires an object that follows a specific protocol (the Dataset) so it knows:
    # How many items exist in total?
    # How do I grab exactly one item?
    train_dataset=RDataset(
        file_paths=train_files,
        tokenizer=tokenizer,
        label2id=label2id,
        map_paths=train_maps,
        max_len=max_len,
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
    loss_function = nn.CrossEntropyLoss() #DESIGN: could use FocalLoss

    # ---------Optimiser---------
    #DEISGN separate learning rates

    # The classifier head is usually randomly initialized. It is "dumb" at the start, while BERT is already "smart." By giving the classifier a 5x higher learning rate, you allow it to catch up and learn the specific task quickly without waiting for the slow-moving LoRA weights.
    classifier_params = list(module.classifier.parameters())
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "classifier" not in n]
    # In LoRA, most of the model (the original BERT/LLM weights) is frozen (requires_grad=False). This filter ensures we only grab the new LoRA weights you've added.

    groups = [
        {"params": lora_params, "lr": lr},
        {"params": classifier_params, "lr": lr * 5}, #DESIGN HYPERPARAMETER
    ]

    # admW mathematically separates the weight decay from the gradient updates, which helps the model generalize much better.
    # Groups: different rules for different layers
    # weight decay: penalty for being too complex. It prevents any single weight from becoming too large and "dominating" the logic. Large weights are a sign that the model is memorizing specific training examples (overfitting) rather than learning general patterns.
    optimizer = AdamW(groups, weight_decay=weight_decay)

    #-----scaler for float16 error------
    scaler = GradScaler("cpu", enabled=False) if device.type not in ("cuda",) else GradScaler("cuda")

    # ---------Scheduler----------
    total_steps = (len(train_loader) // grad_accum) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    #DESIGN alternatives: linear, polynomial, constant-with-warmup
    # num_cycles=0.5 = one half-cosine; try CosineAnnealingWarmRestarts for multi-cycle
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=0.5,
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
            e1_pos = batch["e1_pos"].to(device)
            e1c_pos = batch["e1c_pos"].to(device)
            e2_pos = batch["e2_pos"].to(device)
            e2c_pos = batch["e2c_pos"].to(device)
            labels = batch["labels"].to(device)

            with autocast(device_type=device.type, enabled=(device.type in ("cuda", "mps"))):
                logits = model(input_ids, attention_mask, e1_pos, e1c_pos, e2_pos, e2c_pos)
                loss_val = loss_function(logits, labels)
                loss_val = loss_val / accumulation
            
            scaler.scale(loss).backward() #multiply by a big number

            if step % grad_accum == 0:
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

            total_loss += loss.item() * accumulation
            pbar.set_postfix(loss=f"{total_loss / step:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_loss = total_loss / len(train_loader)

        
        





if __name__ == "__main__":
    main()