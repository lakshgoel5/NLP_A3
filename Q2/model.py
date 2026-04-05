import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, AutoPeftModelForCausalLM

# Hidden state flow:
# Input tokens → Layer 1 hidden_states[0] → ... → Layer 28 hidden_states[-1] [batch, seq_len, 1536]
# We extract entity marker positions from the last hidden layer and feed to a classifier.


def load_base_model():
    model_name = "Qwen/Qwen2.5-1.5B"
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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", #DESIGN "lora_only" or "all" can sometimes improve performance
        task_type=TaskType.CAUSAL_LM, #TaskType.CAUSAL_LM is a configuration constant used in the Hugging Face PEFT (Parameter-Efficient Fine-Tuning) library to specify that a model is being used for Causal Language Modeling.
    )

    # get_peft_model() is a Hugging Face PEFT library function that wraps a pre-trained base model with a PeftConfig to create a trainable PeftModel
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def load_trained_model(output_dir, device):
    # AutoPeftModelForCausalLM reads adapter_config.json to find the
    # base model name, downloads it if needed, and applies the saved adapters —
    # avoids duplicating that logic here.

    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # #DESIGN: left-pad for generation (labels align at right)

    model = AutoPeftModelForCausalLM.from_pretrained(
        output_dir,
        dtype=torch.float16,
    )
    model = model.to(device)
    model.eval()
    return tokenizer, model