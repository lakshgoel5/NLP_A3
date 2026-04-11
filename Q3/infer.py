# from vllm import LLM, SamplingParams
from typing import List
import os
import random
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

MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
rng = random.Random(42)

SYSTEM_PROMPT_TEMPLATE = """You are a relation extraction system. Given a sentence and two entities, output the relation between them.
You MUST output EXACTLY ONE label from the list below — nothing else, no explanation, no punctuation:
{label_list}"""

def generate_vllm_responses(prompts: List[str], model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct") -> List[str]:
    """
    Takes a list of prompt strings and returns a list of generated text strings using vLLM.
    """
    # 1. Initialize the LLM (This loads the model weights into GPU memory)
    print(f"Loading model '{model_name}'...")
    llm = LLM(model=model_name, dtype="float16", trust_remote_code=True, max_model_len=8192)

    # 2. Define sampling parameters
    # You can tweak temperature, max_tokens, top_p, etc. here.
    sampling_params = SamplingParams(temperature=0.7, max_tokens=1024)

    # 3. Run batch generation
    # vLLM automatically handles batching the prompts for maximum GPU throughput
    print("Generating responses...")
    outputs = llm.generate(prompts, sampling_params)

    # 4. Extract the text from the vLLM RequestOutput objects
    # outputs[0] refers to the best completion (if you generated multiple completions per prompt)
    generated_texts = [output.outputs[0].text for output in outputs]

    return generated_texts

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

def load_demo_data(file_path):

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
    p.add_argument("--test_file", required=True)
    p.add_argument("--output_dir", default="./output")

    #---Test time----
    p.add_argument("--num_demos", type=int, default=8)
    p.add_argument("--retrieval", choices=["similarity", "stratified", "random", "auto"], default="auto")

    args = p.parse_args()

    if args.retrieval == "auto":
        args.retrieval = "similarity" if args.lang in ("en", "hi") else "stratified"

    print(f"[cfg] lang={args.lang}  retrieval={args.retrieval}  k={args.num_demos}")

    os.makedirs(args.output_dir, exist_ok=True)

    #-------load eng data------
    read_data = load_demo_data(TRAIN_FILE)
    print(f"[demo] {len(read_data)} examples, {len(set(e['label'] for e in read_data))} unique labels")
    print(f"[demo] sample: {read_data[0]}")


    #-------load test data------
    test_data = load_test_data(args.test_file)
    print(f"[test] {len(test_data)} sentences")

    #--------A class that gets demos based on query---------
    #LLM outputs eng labels, so need translation
    forward_map = load_label_map(args.lang)
    print(f"[map] {len(forward_map)} entries for {args.lang}")
    print(f"[map] Entry sample: {list(forward_map.items())[0]}")

    #--------read test and build prompts------
    #System prompt
    all_labels = sorted({e["label"] for e in read_data})
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(label_list="\n".join(all_labels))

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

    
    #-----Decode output and write to file-------------

    



if __name__ == "__main__":
    # List of input strings
    # my_prompts = [
    #     "The capital of France is",
    #     "Write a haiku about a GPU:",
    #     "Explain the theory of relativity in one sentence:"
    # ]
    
    # # Get the outputs
    # results = generate_vllm_responses(my_prompts, model_name = "/home/scai/msr/aiy247541/scratch/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659")
    
    # # Print the results
    # for i, (prompt, response) in enumerate(zip(my_prompts, results)):
    #     print(f"\n--- Prompt {i+1} ---")
    #     print(f"Input: {prompt}")
    #     print(f"Output: {response}")

    main()