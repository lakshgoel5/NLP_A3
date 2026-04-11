# from vllm import LLM, SamplingParams
from typing import List
import os
from tqdm import tqdm
import json

import argparse

TRAIN_FILE = "../en_sft_dataset/train.jsonl" 
MAP_DIR = "../sft_dataset"

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

    #--------read test and build prompts------
    # get k demos using 
    # format them in prompt


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