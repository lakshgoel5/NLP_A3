from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

SPECIAL_TOKENS = ["[E1]", "[/E1]", "[E2]", "[/E2]"]


# hidden = outputs.hidden_states[-1]  # [batch, seq_len, 1536]
# extract E1 and E2 positions → feed to classifier

# Input tokens
# ↓
# Layer 1  → hidden_states[0]   [batch, seq_len, 1536]
#  ↓
# Layer 2  → hidden_states[1]   [batch, seq_len, 1536]
#  ↓
# ...
#  ↓
# Layer 28 → hidden_states[-1]  [batch, seq_len, 1536]  ← we use this
#  ↓
# Logits (next token scores)    [batch, seq_len, 151936] ← we ignore this

def load_base_model():
    model_name = "Qwen/Qwen2.5-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add special tokens
    tokenizer.add_special_tokens({"special_tokens": SPECIAL_TOKENS})
 
    # Qwen doesn't have a pad token by default — use eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Auto" means HuggingFace figures out the right model class from the model name. "ForCausalLM" means it's a decoder-only language model (predicts next token).
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16, # save memory # DESIGN
        output_hidden_states=True, # we need hidden states for classification, we'll feed last layer output to linear classifier and 
    )

    # Resize embeddings for new tokens
    model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model

# freeze everything, inject tiny trainable matrices at specific places
def lora(model, lora_rank, lora_alpha, lora_dropout):
    # output = W·x + (lora_alpha/r) × B·A·x
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout, #DESIGN
        target_modules=["q_proj", "v_proj"], #DESIGN
        bias="none", #DESIGN
        task_type=TaskType.FEATURE_EXTRACTION, #extracting hidden states
    )

    # get_peft_model() is a Hugging Face PEFT library function that wraps a pre-trained base model with a PeftConfig to create a trainable PeftModel
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model