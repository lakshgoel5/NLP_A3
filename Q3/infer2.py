from vllm import LLM, SamplingParams
from typing import List, Dict, Tuple
import os
import random
import difflib
import numpy as np
from tqdm import tqdm
import json
import argparse

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss

TRAIN_FILE = "../en_sft_dataset/train.jsonl"
MAP_DIR = "../sft_dataset"

MODEL = "/home/scai/msr/aiy247541/scratch/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# All Indic training files with their label maps
INDIC_LANGS = ["hi", "kn", "or", "tcy"]

ALL_LABELS = [
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

SYSTEM_PROMPT_TEMPLATE = """You are a relation extraction system. Given a sentence and two entities, output the relation between them.
You MUST output EXACTLY ONE label from the list below — nothing else, no explanation, no punctuation:
{label_list}"""


def closest_label(pred: str, valid_labels: list) -> str:
    pred = pred.strip()
    if pred in valid_labels:
        return pred
    for label in valid_labels:
        if pred.startswith(label):
            return label
    matches = difflib.get_close_matches(pred, valid_labels, n=1, cutoff=0.0)
    return matches[0] if matches else "NA"


def load_demo_data(file_path, inv_label_map=None):
    data_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for data in tqdm([json.loads(l) for l in lines if l.strip()], desc=file_path.split("/")[-1], leave=False):
        sent = data["sentText"]
        for rm in data.get("relationMentions", []):
            em1 = rm["em1Text"]
            em2 = rm["em2Text"]
            raw_label = rm["label"]
            if not raw_label or raw_label == "NA":
                continue
            if inv_label_map:
                raw_label = inv_label_map.get(raw_label, raw_label)
            if raw_label not in ALL_LABELS:
                continue
            data_list.append({
                "sentText": sent,
                "em1Text": em1,
                "em2Text": em2,
                "label": raw_label
            })
    return data_list


def load_test_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def split_for_local_eval(file_path, inv_label_map=None, seed=42):
    """Split an Indic train file 80/20. Returns (demo_list, test_records).
    demo_list: flat dicts for the demo pool (80%)
    test_records: nested JSONL records for inference (20%)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    random.Random(seed).shuffle(lines)
    n_test = max(1, int(len(lines) * 0.2))
    train_lines = lines[n_test:]   # 80%
    test_lines  = lines[:n_test]   # 20%

    demo_data = []
    for data in [json.loads(l) for l in train_lines]:
        sent = data["sentText"]
        for rm in data.get("relationMentions", []):
            em1, em2, raw_label = rm["em1Text"], rm["em2Text"], rm["label"]
            if not raw_label or raw_label == "NA":
                continue
            if inv_label_map:
                raw_label = inv_label_map.get(raw_label, raw_label)
            if raw_label not in ALL_LABELS:
                continue
            demo_data.append({"sentText": sent, "em1Text": em1, "em2Text": em2, "label": raw_label})

    test_data = [json.loads(l) for l in test_lines]
    return demo_data, test_data


def load_label_map(lang: str) -> dict:
    if lang == "en":
        return {}
    path = os.path.join(MAP_DIR, f"{lang}_map.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class Retriever:
    def __init__(self, demo_data, retrieval_type, embed_model):
        self.demo_data = demo_data
        self.retrieval_type = retrieval_type
        self.rng = random.Random(42)

        self.by_label = {}
        for ex in demo_data:
            label = ex["label"]
            if label not in self.by_label:
                self.by_label[label] = []
            self.by_label[label].append(ex)

        if retrieval_type == "similarity":
            self.build_faiss_index(embed_model)

    def build_faiss_index(self, embed_model):
        print(f"[retriever] loading embedder: {embed_model}")
        self.embedder = SentenceTransformer(embed_model)
        texts = [f"{ex['em1Text']} {ex['em2Text']} {ex['sentText']}" for ex in self.demo_data]
        print(f"[retriever] encoding {len(texts)} pool texts...")
        embs = self.embedder.encode(
            texts, batch_size=512, show_progress_bar=True,
            normalize_embeddings=True, convert_to_numpy=True,
        ).astype(np.float32)
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)
        print(f"[retriever] index built: {self.index.ntotal} vectors, dim={embs.shape[1]}")

    def retrieve_batch(self, queries, k):
        if self.retrieval_type == "similarity":
            return self.retrieve_similarity_batch(queries, k)
        elif self.retrieval_type == "stratified":
            return [self.retrieve_stratified(k) for _ in queries]
        else:
            return [self.rng.sample(self.demo_data, k) for _ in queries]

    def retrieve_stratified(self, k):
        labels = list(self.by_label.keys())
        self.rng.shuffle(labels)
        demos = []
        i = 0
        while len(demos) < k:
            lbl = labels[i % len(labels)]
            demos.append(self.rng.choice(self.by_label[lbl]))
            i += 1
        return demos

    def retrieve_similarity_batch(self, queries, k):
        texts = [f"{em1} {em2} {sent}" for em1, em2, sent in queries]
        q_embs = self.embedder.encode(
            texts, batch_size=512, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=True,
        ).astype(np.float32)
        _, idxs = self.index.search(q_embs, k)
        return [[self.demo_data[i] for i in row] for row in idxs]


# Context-aware prompt builder: drops demos from the end (least relevant first)
# until the tokenized prompt fits within max_prompt_tokens.
# Indic scripts tokenize into 2-4x more tokens than English, so without this
# long Indic sentences silently truncate inside vLLM.
def build_prompt(tokenizer, system_prompt, demos, em1, em2, sent, max_prompt_tokens=3500):
    for n in range(len(demos), -1, -1):
        messages = [{"role": "system", "content": system_prompt}]
        for d in demos[:n]:
            example = f"Entity 1: {d['em1Text']}\nEntity 2: {d['em2Text']}\nSentence: {d['sentText']}\nRelation:"
            messages.append({"role": "user", "content": example})
            messages.append({"role": "assistant", "content": d["label"]})
        query = f"Entity 1: {em1}\nEntity 2: {em2}\nSentence: {sent}\nRelation:"
        messages.append({"role": "user", "content": query})
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if len(tokenizer.encode(prompt)) <= max_prompt_tokens:
            return prompt
    return prompt  # zero-shot fallback (n=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", required=True, choices=["en", "hi", "kn", "or", "tcy"])
    p.add_argument("--test_file", default=None)
    p.add_argument("--output_dir", default="./output")
    p.add_argument("--model_path", default=None, help="Local path to model weights (overrides MODEL constant)")
    p.add_argument("--num_demos", type=int, default=8)
    p.add_argument("--retrieval", choices=["similarity", "stratified", "random", "auto"], default="auto")
    p.add_argument("--holdout_eval", action="store_true",
                   help="Split Indic train 80/20; use 20%% as test (local eval only)")

    args = p.parse_args()

    # With all Indic data now in the pool, similarity retrieval works for every language
    if args.retrieval == "auto":
        args.retrieval = "similarity"

    model_name = args.model_path if args.model_path else MODEL
    print(f"[cfg] lang={args.lang}  retrieval={args.retrieval}  k={args.num_demos}")
    print(f"[cfg] model={model_name}")

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load English demo pool
    demo_data = load_demo_data(TRAIN_FILE)
    print(f"[demo] English: {len(demo_data)} examples")

    # Load ALL Indic training data into demo pool (not just hi/kn).
    # This means or/tcy test queries also find same-script demos via similarity.
    holdout_test = None
    for lang_code in INDIC_LANGS:
        indic_train = os.path.join(MAP_DIR, f"{lang_code}_train.jsonl")
        indic_map_path = os.path.join(MAP_DIR, f"{lang_code}_map.json")
        if os.path.isfile(indic_train) and os.path.isfile(indic_map_path):
            with open(indic_map_path, "r", encoding="utf-8") as f:
                fwd = json.load(f)
            inv_map = {v: k for k, v in fwd.items()}
            if args.holdout_eval and lang_code == args.lang:
                indic_demos, holdout_test = split_for_local_eval(indic_train, inv_map)
                print(f"[holdout] {lang_code}: {len(indic_demos)} demo / {len(holdout_test)} test")
            else:
                indic_demos = load_demo_data(indic_train, inv_label_map=inv_map)
            demo_data.extend(indic_demos)
            print(f"[demo] {lang_code}: +{len(indic_demos)} examples → total {len(demo_data)}")

    if holdout_test is not None:
        test_data = holdout_test
    else:
        if args.test_file is None:
            raise ValueError("--test_file is required when not using --holdout_eval")
        test_data = load_test_data(args.test_file)
    print(f"[test] {len(test_data)} sentences")

    forward_map = load_label_map(args.lang)
    print(f"[map] {len(forward_map)} entries for {args.lang}")

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(label_list="\n".join(sorted(ALL_LABELS)))

    retriever = Retriever(demo_data, args.retrieval, embed_model=EMBED_MODEL)

    queries = []
    index_map = []
    for sent_idx, data in enumerate(tqdm(test_data, desc="build prompts")):
        sent = data["sentText"]
        for m_idx, rm in enumerate(data["relationMentions"]):
            queries.append((rm["em1Text"], rm["em2Text"], sent))
            index_map.append((sent_idx, m_idx))

    demos_per_query = retriever.retrieve_batch(queries, args.num_demos)

    prompts = []
    for (em1, em2, sent), demos in zip(queries, demos_per_query):
        prompts.append(build_prompt(tokenizer, system_prompt, demos, em1, em2, sent))

    print(f"[prompts] built {len(prompts)} prompts")
    print(f"[prompts] sample:\n{prompts[0][:800]}...\n")

    print("[vllm] loading model...")
    llm = LLM(model=model_name, dtype="float16", trust_remote_code=True, max_model_len=8192)

    # Guided decoding: constrains vLLM to output exactly one of the 24 valid labels.
    # Eliminates label-snapping errors entirely — model cannot produce an invalid string.
    # Falls back to greedy + stop tokens if the installed vLLM version is too old.
    try:
        sampling_params = SamplingParams(temperature=0, max_tokens=64, guided_choice=ALL_LABELS)
        print("[vllm] using guided decoding")
    except TypeError:
        sampling_params = SamplingParams(temperature=0, max_tokens=64, stop=["\n", "<|eot_id|>"])
        print("[vllm] guided decoding not supported, using greedy + stop tokens")

    print("[vllm] generating...")
    outputs = llm.generate(prompts, sampling_params)
    raw_preds = [o.outputs[0].text for o in outputs]

    # Pre-fill every mention with NA so the output is always valid
    results = []
    for record in test_data:
        results.append({
            "articleId": record.get("articleId", ""),
            "sentId": record.get("sentId", ""),
            "sentText": record["sentText"],
            "relationMentions": [
                {"em1Text": rm["em1Text"], "em2Text": rm["em2Text"], "label": "NA"}
                for rm in record.get("relationMentions", [])
            ],
        })

    for (sent_idx, m_idx), raw in zip(index_map, raw_preds):
        en_label = closest_label(raw, ALL_LABELS)
        label = forward_map.get(en_label, en_label) if forward_map else en_label
        results[sent_idx]["relationMentions"][m_idx]["label"] = label

    out_path = os.path.join(args.output_dir, f"Q3_{args.lang}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[done] saved {len(results)} predictions → {out_path}")

    if args.holdout_eval and holdout_test is not None:
        ref_path = os.path.join(args.output_dir, f"holdout_ref_{args.lang}.jsonl")
        with open(ref_path, "w", encoding="utf-8") as f:
            for r in test_data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[holdout] reference saved to {ref_path}")


if __name__ == "__main__":
    main()
