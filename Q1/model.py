import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
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

    model = AutoModel.from_pretrained(
        model_name,
        dtype=torch.float16, # save memory # DESIGN (bfloat16 might be stable on A100)
    )

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

class RelationClassifier(nn.Module):
    def __init__(self, base_model, vector_dim, num_classes, entity_repr):
        super().__init__()
        self.base_model = base_model
        self.vector_dim = vector_dim
        self.num_classes = num_classes
        self.entity_repr = entity_repr

        if entity_repr == "concat_four":
            self.classifier = nn.Linear(4 * vector_dim, num_classes)
        elif entity_repr == "concat_two":
            self.classifier = nn.Linear(2 * vector_dim, num_classes)
        elif entity_repr == "mean_four":
            self.classifier = nn.Linear(vector_dim, num_classes)
        elif entity_repr == "mean_two":
            self.classifier = nn.Linear(vector_dim, num_classes)
        elif entity_repr == "last_token":
            self.classifier = nn.Linear(vector_dim, num_classes)
        else:
            raise ValueError("Invalid entity_repr")

    def forward(self, input_ids, attention_mask, e1_pos, e1_end_pos, e2_pos, e2_end_pos):
        outputs = self.base_model(input_ids, attention_mask)
        hidden_states = outputs.last_hidden_state  #DESIGN: try outputs.hidden_states[-2] (second-to-last) or average last N layers

        batch_size = hidden_states.shape[0]

        batch_vector = torch.arange(batch_size)

        # hidden_states is 3D tensor of dim (B, seq_len, 1536)
        # each token's hidden state has attended to every other token through self-attention

        if self.entity_repr == "concat_four":
            e1_repr  = hidden_states[batch_vector, e1_pos] #(B, 1536)
            e1_end_repr = hidden_states[batch_vector, e1_end_pos]
            e2_repr  = hidden_states[batch_vector, e2_pos]
            e2_end_repr = hidden_states[batch_vector, e2_end_pos]
            combined = torch.cat([e1_repr, e1_end_repr, e2_repr, e2_end_repr], dim=-1)
        elif self.entity_repr == "concat_two":
            e1_repr  = hidden_states[batch_vector, e1_pos]
            e2_repr  = hidden_states[batch_vector, e2_pos]
            combined = torch.cat([e1_repr, e2_repr], dim=-1)
        elif self.entity_repr == "mean_four":
            e1_repr  = hidden_states[batch_vector, e1_pos]
            e1_end_repr = hidden_states[batch_vector, e1_end_pos]
            e2_repr  = hidden_states[batch_vector, e2_pos]
            e2_end_repr = hidden_states[batch_vector, e2_end_pos]
            combined = (e1_repr + e1_end_repr + e2_repr + e2_end_repr) / 4
        elif self.entity_repr == "mean_two":
            e1_repr  = hidden_states[batch_vector, e1_pos]
            e2_repr  = hidden_states[batch_vector, e2_pos]
            combined = (e1_repr + e2_repr) / 2
        elif self.entity_repr == "last_token":
            seq_lens = attention_mask.sum(dim=1) - 1
            combined = hidden_states[batch_vector, seq_lens]
        else:
            raise ValueError("Invalid entity_repr")

        logits = self.classifier(combined)
        return logits