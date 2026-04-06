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


class RelationDataset(Dataset):
    def __init__(self, file_paths, tokenizer, label2id, map_paths=None, max_length=256):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.examples = []
        self.load_data(file_paths, map_paths)

    def load_data(self, file_paths, map_paths):
        e1_id = self.tokenizer.convert_tokens_to_ids("[E1]")
        e1_end_id = self.tokenizer.convert_tokens_to_ids("[/E1]")
        e2_id = self.tokenizer.convert_tokens_to_ids("[E2]")
        e2_end_id = self.tokenizer.convert_tokens_to_ids("[/E2]")

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
                content = f.read()
            decoder = json.JSONDecoder()
            buf = content.lstrip()
            records = []
            while buf:
                try:
                    obj, idx = decoder.raw_decode(buf)
                    records.append(obj)
                    buf = buf[idx:].lstrip()
                except json.JSONDecodeError:
                    break
            for data in tqdm(records, desc=file_path.split("/")[-1], leave=False):
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

                        marked = self.mark_entities(sent, em1, em2) # This function will add special tokens around the entities
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

                        # Locate special token positions
                        e1_pos = self.find_token_position(ids, e1_id)
                        e1_end_pos = self.find_token_position(ids, e1_end_id)
                        e2_pos = self.find_token_position(ids, e2_id)
                        e2_end_pos = self.find_token_position(ids, e2_end_id)

                        if e1_pos is None or e1_end_pos is None or e2_pos is None or e2_end_pos is None:
                            skipped += 1
                            continue

                        self.examples.append({
                            "input_ids": torch.tensor(ids, dtype=torch.long), #for whole numbers used as INDICES
                            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long), #for whole numbers used as INDICES
                            "e1_pos": e1_pos,
                            "e1_end_pos": e1_end_pos,
                            "e2_pos": e2_pos,
                            "e2_end_pos": e2_end_pos,
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
    
    def mark_entities(self, sent_text, em1, em2):
        pos1 = sent_text.find(em1) # searches the string for substring and returns the character index of its first occurrence.
        pos2 = sent_text.find(em2)

        if pos1 == -1 or pos2 == -1:
            return None

        # Build a list of (start, end, open_tag, close_tag) sorted right-to-left
        spans = [
            (pos1, pos1 + len(em1), "[E1]", "[/E1]"),
            (pos2, pos2 + len(em2), "[E2]", "[/E2]"),
        ]

        # Sort spans by start position in descending order
        spans.sort(key=lambda x: x[0], reverse=True)

        # Insert tags into the sentence
        marked_sent = sent_text
        for start, end, open_tag, close_tag in spans:
            marked_sent = marked_sent[:start] + open_tag + marked_sent[start:end] + close_tag + marked_sent[end:]

        return marked_sent


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