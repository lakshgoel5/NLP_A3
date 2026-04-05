import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

SPECIAL_TOKENS = ["[E1]", "[/E1]", "[E2]", "[/E2]"]

# Hidden state flow:
# Input tokens → Layer 1 hidden_states[0] → ... → Layer 28 hidden_states[-1] [batch, seq_len, 1536]
# We extract entity marker positions from the last hidden layer and feed to a classifier.


def load_base_model():
    model_name = "Qwen/Qwen2.5-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add special tokens
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
 
    # Qwen doesn't have a pad token by default — use eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Auto" means HuggingFace figures out the right model class from the model name. "ForCausalLM" means it's a decoder-only language model (predicts next token).
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16, # save memory # DESIGN (bfloat16 might be stable on A100)
    )
    
    # output_hidden_states must be set on the config, NOT passed to from_pretrained
    model.config.output_hidden_states = True

    # Resize embeddings for new tokens
    model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model

# freeze everything, inject tiny trainable matrices at specific places
def lora(model, lora_rank, lora_alpha, lora_dropout):
    # output = W·x + (lora_alpha/r) × B·A·x
    lora_config = LoraConfig(
        r=lora_rank, #DESIGN try 8,16,32,64
        lora_alpha=lora_alpha, #DESIGN Try sqrt(2)*r, 2*r
        lora_dropout=lora_dropout, #DESIGN 0.05-0.1
        target_modules=["q_proj", "v_proj"], #DESIGN
        bias="none", #DESIGN "lora_only" or "all" can sometimes improve performance
        task_type=TaskType.FEATURE_EXTRACTION, #extracting hidden states
    )

    # get_peft_model() is a Hugging Face PEFT library function that wraps a pre-trained base model with a PeftConfig to create a trainable PeftModel
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model