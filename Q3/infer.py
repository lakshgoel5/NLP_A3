from vllm import LLM, SamplingParams
from typing import List
import os
import random
import difflib
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
import json

import argparse

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss

# What AutoTokenizer.from_pretrained(MODEL) does
# It downloads (or loads from cache) the tokenizer that ships with Llama-3.1-8B-Instruct. A tokenizer is a small object (~5 MB of files: tokenizer.json, special_tokens_map.json, tokenizer_config.json) that knows two things:

# How to turn text into token IDs and back — the vocabulary and merge rules.
# How to format a list of chat messages into the exact string the model was trained to expect — the chat template.

# Llama-3.1-Instruct wasn't trained on plain text. It was trained on text formatted with specific special tokens that mark where the system prompt starts, where each user turn begins, where the assistant should respond, and so on. 

# messages = [
#     {"role": "system",    "content": "You are a relation extraction system..."},
#     {"role": "user",      "content": "Entity 1: Paris\nEntity 2: France\n..."},
#     {"role": "assistant", "content": "/location/country/capital"},
#     {"role": "user",      "content": "Entity 1: ...\n..."},  # the query
# ]

# prompt_string = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,            # we want the string, not token IDs
#     add_generation_prompt=True, # append the "assistant turn starts here" tokens
# )

TRAIN_FILE = "../en_sft_dataset/train.jsonl" 
MAP_DIR = "../sft_dataset"

MODEL = "/home/scai/msr/aiy247541/scratch/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

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

rng = random.Random(42)

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

class Retriever:
    def __init__(self, demo_data, retrieval_type, embed_model):
        self.demo_data = demo_data
        self.retrieval_type = retrieval_type
        self.embed_model = embed_model
        self.rng = random.Random(42)

        self.by_label = {} # Group examples by Label

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

        _, idxs = self.index.search(q_embs, k)  # shape (N, k)
        
        return [[self.demo_data[i] for i in row] for row in idxs]

def load_demo_data(file_path, inv_label_map=None):

    data_list = []

    ##SUBMISSION reading line wise
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for data in tqdm([json.loads(l) for l in lines if l.strip()], desc=file_path.split("/")[-1], leave=False): #if l.strip() A filter that skips empty lines or lines containing only whitespace
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
    # Nested test data
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

def build_prompt(tokenizer, system_prompt, demos, em1, em2, sent):
    # a prompt for each sample to be inferred
    messages = [{"role": "system", "content": system_prompt}]

    # add demos as few-shot examples
    for d in demos:
        example = (f"Entity 1: {d['em1Text']}\nEntity 2: {d['em2Text']}\nSentence: {d['sentText']}\nRelation:")
        messages.append({"role": "user", "content": example})
        messages.append({"role": "assistant", "content": d["label"]})

    # add the query
    query = f"Entity 1: {em1}\nEntity 2: {em2}\nSentence: {sent}\nRelation:"
    messages.append({"role": "user", "content": query})

    # apply chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", required=True, choices=["en", "hi", "kn", "or", "tcy"])
    p.add_argument("--test_file", default=None)
    p.add_argument("--output_dir", default="./output")
    p.add_argument("--holdout_eval", action="store_true",
                   help="Split Indic train 80/20; use 20%% as test (local eval only)")

    #---Test time----
    p.add_argument("--num_demos", type=int, default=8)
    p.add_argument("--retrieval", choices=["similarity", "stratified", "random"], default="similarity")

    args = p.parse_args()

    model_name = MODEL
    print(f"[cfg] lang={args.lang}  retrieval={args.retrieval}  k={args.num_demos}")
    print(f"[cfg] model={model_name}")

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    #-------load eng data------
    read_data = load_demo_data(TRAIN_FILE)
    print(f"[demo] {len(read_data)} examples, {len(set(e['label'] for e in read_data))} unique labels")

    # For languages with labeled training data, add their examples to the demo pool.
    # This gives the similarity retriever in-language examples to match against.
    holdout_test = None
    if args.lang in ("hi", "kn"):
        indic_train = os.path.join(MAP_DIR, f"{args.lang}_train.jsonl")
        indic_map_path = os.path.join(MAP_DIR, f"{args.lang}_map.json")
        if os.path.isfile(indic_train) and os.path.isfile(indic_map_path):
            with open(indic_map_path, "r", encoding="utf-8") as f:
                fwd = json.load(f)
            inv_map = {v: k for k, v in fwd.items()}
            if args.holdout_eval:
                indic_demos, holdout_test = split_for_local_eval(indic_train, inv_map)
                print(f"[holdout] {args.lang}: {len(indic_demos)} demo / {len(holdout_test)} test")
            else:
                indic_demos = load_demo_data(indic_train, inv_label_map=inv_map)
            read_data.extend(indic_demos)
            print(f"[demo] added {len(indic_demos)} {args.lang} examples → total {len(read_data)}")

    #-------load test data------
    if holdout_test is not None:
        test_data = holdout_test
    else:
        if args.test_file is None:
            raise ValueError("--test_file is required when not using --holdout_eval")
        test_data = load_test_data(args.test_file)
    print(f"[test] {len(test_data)} sentences")

    #--------A class that gets demos based on query---------
    #LLM outputs eng labels, so need translation
    forward_map = load_label_map(args.lang)
    print(f"[map] {len(forward_map)} entries for {args.lang}")
    if forward_map:
        print(f"[map] Entry sample: {list(forward_map.items())[0]}")

    #--------read test and build prompts------
    # Use ALL_LABELS directly so every valid label appears in the prompt,
    # even if a rare label has zero examples in the training pool.
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(label_list="\n".join(sorted(ALL_LABELS)))

    retriever = Retriever(read_data, args.retrieval, embed_model=EMBED_MODEL)

    queries = []
    index_map = [] # for later mapping back vllm output for nesting data

    # get k demos using 
    # format them in prompt
    for sent_idx, data in enumerate(tqdm(test_data, desc="build prompts")):
        sent = data["sentText"]
        for m_idx, rm in enumerate(data["relationMentions"]):
            queries.append((rm["em1Text"], rm["em2Text"], sent))
            index_map.append((sent_idx, m_idx))

    # test_data[sent_idx]["relationMentions"][m_idx] ↔ prompts[i]. 
    # When vllm returns a flat list, zip(index_map, predictions) gives you back the (sent_idx, m_idx) address for each one.

    # Expensive work on entire batch rather than on individual query
    demos_per_query = retriever.retrieve_batch(queries, args.num_demos)

    print(f"[debug] demos for query 0: {[d['label'] for d in demos_per_query[0]]}")

    prompts = []
    # Make prompts using the retrieved demos
    for (em1, em2, sent), demos in zip(queries, demos_per_query):
        prompts.append(build_prompt(tokenizer, system_prompt, demos, em1, em2, sent))

    print(f"[prompts] built {len(prompts)} prompts")
    print(f"[prompts] sample:\n{prompts[0][:800]}...\n")

    #-------Batch all prompts---------
    # pack all prompts together for speed
    # Calling it once with 2000 prompts is ~50× faster than calling it 2000 times with 1 prompt each. 

    #--------Pass to VLLM-----------
    print("[vllm] loading model...")
    llm = LLM(model=model_name, dtype="float16", trust_remote_code=True, max_model_len=8192)
    sampling_params = SamplingParams(temperature=0, max_tokens=64, stop=["\n", "<|eot_id|>"])
    print("[vllm] generating...")
    outputs = llm.generate(prompts, sampling_params)
    raw_preds = [o.outputs[0].text for o in outputs]

    #-----Decode output and write to file-------------
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
    print(f"[done] saved {len(results)} predictions to {out_path}")

    if args.holdout_eval and holdout_test is not None:
        ref_path = os.path.join(args.output_dir, f"holdout_ref_{args.lang}.jsonl")
        with open(ref_path, "w", encoding="utf-8") as f:
            for r in test_data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[holdout] reference saved to {ref_path}")

if __name__ == "__main__":
    main()