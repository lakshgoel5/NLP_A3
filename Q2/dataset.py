import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

ALL_RELATION_LABELS = [
    "NA",
    "/business/company/advisors",
    "/business/company/founders",
    "/business/company/industry",
    "/business/company/major_shareholders",
    "/business/company/place_founded",
    "/business/company_shareholder/major_shareholder_of",
    "/business/person/company",
    "/location/administrative_division/country",
    "/location/country/administrative_divisions",
    "/location/country/capital",
    "/location/location/contains",
    "/location/neighborhood/neighborhood_of",
    "/people/deceased_person/place_of_death",
    "/people/ethnicity/geographic_distribution",
    "/people/ethnicity/people",
    "/people/person/children",
    "/people/person/ethnicity",
    "/people/person/nationality",
    "/people/person/place_lived",
    "/people/person/place_of_birth",
    "/people/person/profession",
    "/people/person/religion",
    "/sports/sports_team/location",
    "/sports/sports_team_location/teams",
]

NUM_CLASSES = len(ALL_RELATION_LABELS)

def build_label_map():
    label2id = {l: i for i, l in enumerate(ALL_RELATION_LABELS)}
    id2label  = {i: l for i, l in enumerate(ALL_RELATION_LABELS)}
    return label2id, id2label


class SFTDataset(Dataset):
    def __init__(self, file_paths, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        self.load_data(file_paths)

    def load_data(self, file_paths):
        file_to_invmap = {}
        if map_paths:
            for map_path in map_paths:
                with open(map_path, "r", encoding="utf-8") as f:
                    forward_map = json.load(f)
                    inv = {v: k for k, v in forward_map.items()}
                    lang_code = map_path.split("/")[-1].split("_")[0] #DEBUG: Hardcoaded name of langauge
                    for fp in file_paths:
                        if lang_code in fp:
                            file_to_invmap[fp] = inv

        skipped = 0
        for file_path in file_paths:
            inv_map = file_to_invmap.get(file_path, None)
            
            with open(file_path, "r", encoding="utf-8") as f:
                for line in tqdm(f, desc=file_path.split("/")[-1], leave=False):
                    line = line.strip()
                    if not line:
                        continue

                    data = json.loads(line) # This converts the JSON-formatted string into a Python dictionary
                    sent = data["sentText"]
                    for rm in data.get("relationMentions", []):
                        em1 = rm["em1Text"]
                        em2 = rm["em2Text"]
                        raw_label = rm["label"]

                        if inv_map is not None:
                            raw_label = inv_map.get(raw_label, raw_label)

                        if raw_label not in self.label2id:
                            skipped += 1
                            print(f"Skipping {raw_label}")
                            continue
                        label_id = self.label2id[raw_label]

                        marked = self.make_prompt(sent, em1, em2) # This function will make prompt for model
                        if marked is None:
                            skipped += 1
                            continue
                        
                        enc = self.tokenizer(
                            marked,
                            max_length=self.max_length,
                            truncation=True,
                            padding=False, # Because sentences in a batch have different lengths. collate later efficiently
                            return_tensors=None, # return plain Python lists, not torch tensors # Because sentences in a batch have different lengths. 
                        )
                        # output is a dictionary with keys "input_ids" and "attention_mask"
                        ids = enc["input_ids"]

                        self.examples.append({
                            "input_ids": torch.tensor(ids, dtype=torch.long), #for whole numbers used as INDICES
                            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long), #for whole numbers used as INDICES
                            "label": label_id,
                        })

        print(f"Loaded {len(self.examples)} examples, skipped {skipped}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def find_token_position(self, ids, token_id):
        try:
            return ids.index(token_id)
        except ValueError:
            return None
    
    def make_prompt(self, sent_text, em1, em2):
        return (
            f'Entity 1: {em1}\n'
            f'Entity 2: {em2}\n'
            f'Sentence: {sent_text}\n'
            f'Relation: '
        )

def collate_fn(batch, pad_id):
    max_len = max(x["input_ids"].size(0) for x in batch)
    input_ids_padded = []
    attn_mask_padded = []
    for x in batch:
        L = x["input_ids"].size(0)
        pad_len = max_len - L
        input_ids_padded.append(
            torch.cat([x["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)])
        )
        attn_mask_padded.append(
            torch.cat([x["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
    return {
        "input_ids": torch.stack(input_ids_padded),
        "attention_mask": torch.stack(attn_mask_padded),
        "e1_pos": torch.tensor([x["e1_pos"] for x in batch], dtype=torch.long),
        "e1_end_pos": torch.tensor([x["e1_end_pos"] for x in batch], dtype=torch.long),
        "e2_pos": torch.tensor([x["e2_pos"] for x in batch], dtype=torch.long),
        "e2_end_pos": torch.tensor([x["e2_end_pos"] for x in batch], dtype=torch.long),
        "labels": torch.tensor([x["label"] for x in batch], dtype=torch.long),
    }