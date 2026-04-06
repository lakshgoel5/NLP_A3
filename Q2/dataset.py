import os
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
    def __init__(self, file_paths, tokenizer, max_length=256, map_paths=None, indic_repeat=1):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id, self.id2label = build_label_map()
        self.examples = []
        self.load_data(file_paths, map_paths or [], indic_repeat)

    def load_data(self, file_paths, map_paths, indic_repeat=1):
        # Build inverse maps: indic label -> English label, keyed by file path
        file_to_invmap = {}
        for map_path in map_paths:
            with open(map_path, "r", encoding="utf-8") as f:
                forward_map = json.load(f)
            inv = {v: k for k, v in forward_map.items()}
            lang_code = os.path.basename(map_path).split("_")[0]
            for fp in file_paths:
                if lang_code in os.path.basename(fp):
                    file_to_invmap[fp] = inv

        skipped = 0
        for file_path in file_paths:
            inv_map = file_to_invmap.get(file_path, None)
            repeat = indic_repeat if inv_map is not None else 1

            file_examples = []
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
            for data in tqdm(records, desc=os.path.basename(file_path), leave=False):
                    sent = data["sentText"]
                    for rm in data.get("relationMentions", []):
                        em1 = rm["em1Text"]
                        em2 = rm["em2Text"]
                        raw_label = rm["label"]

                        # Convert indic label to English using inverse map
                        if inv_map is not None:
                            raw_label = inv_map.get(raw_label, raw_label)

                        if raw_label not in self.label2id:
                            skipped += 1
                            continue

                        prompt = self.make_prompt(sent, em1, em2)
                        completion = raw_label + self.tokenizer.eos_token

                        # Tokenize prompt and completion separately to mask prompt in labels
                        prompt_ids = self.tokenizer(
                            prompt,
                            truncation=True,
                            max_length=self.max_length - 20, # leave room for completion
                            add_special_tokens=True,
                        )["input_ids"]

                        completion_ids = self.tokenizer(
                            completion,
                            add_special_tokens=False,
                        )["input_ids"]

                        input_ids = prompt_ids + completion_ids
                        labels = [-100] * len(prompt_ids) + completion_ids

                        # Truncate if still over max_length
                        if len(input_ids) > self.max_length:
                            input_ids = input_ids[:self.max_length]
                            labels = labels[:self.max_length]

                        file_examples.append({
                            "input_ids": torch.tensor(input_ids, dtype=torch.long),
                            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
                            "labels": torch.tensor(labels, dtype=torch.long),
                        })

            self.examples.extend(file_examples * repeat)

        print(f"Loaded {len(self.examples)} examples (with repeats), skipped {skipped}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def make_prompt(self, sent_text, em1, em2):
        return (
            f'Entity 1: {em1}\n'
            f'Entity 2: {em2}\n'
            f'Sentence: {sent_text}\n'
            f'Relation: '
        )

def collate_fn(batch, pad_id):
    max_len = max(x["input_ids"].size(0) for x in batch)
    input_ids_padded, attn_mask_padded, labels_padded = [], [], []
    for x in batch:
        L = x["input_ids"].size(0)
        pad_len = max_len - L
        input_ids_padded.append(torch.cat([x["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)]))
        attn_mask_padded.append(torch.cat([x["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
        labels_padded.append(torch.cat([x["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(input_ids_padded),
        "attention_mask": torch.stack(attn_mask_padded),
        "labels": torch.stack(labels_padded),
    }