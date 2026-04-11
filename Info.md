# NLP Assignment 3: Relation Extraction and Language Adaptation

This project focuses on Relation Extraction (RE) through various techniques including Supervised Fine-Tuning (SFT), Continued Pre-training, and In-Context Learning (ICL). It evaluates performance on datasets across different languages.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Question 1: English Supervised Fine-Tuning
Run the training script to fine-tune on the English dataset:
```bash
./Q1/train.sh
```
Run inference:
```bash
./Q1/infer.sh
```

### Question 2: Language Adaptation
Run the training script for language-specific adaptation:
```bash
./Q2/train.sh
```
Run inference:
```bash
./Q2/infer.sh
```

### Question 3: In-Context Learning (Few-Shot)
Run the ICL inference script:
```bash
./Q3/infer.sh
```

## Evaluation

To evaluate your predictions against ground truth labels:
```bash
python eval.py <path_to_predictions.jsonl> <path_to_reference.jsonl>
```

Testing with sample data:
```bash
python eval.py sample_prediction.jsonl sample_reference.jsonl
```

## Time Limits
- Training: 480 minutes i.e. 8hrs
- Inference: per test file i.e. 500 samples: 30 minutes

## Allowed Libraries

| Library | Purpose |
|---------|---------|
| PyTorch | Core deep learning |
| HuggingFace Transformers | Load models, tokenizers |
| HuggingFace PEFT | LoRA adapters |
| vllm | Fast inference for Q3 |
| FAISS | Vector similarity search (likely for retrieval in Q2/Q3) |
| NumPy, scikit-learn | Data processing, metrics |

## Not Allowed

| Restriction | Implication |
|-------------|-------------|
| No pre-trained RE models | Can't use models already trained on relation extraction datasets |
| Only given data | No external datasets, no scraping |
| No translation at inference | Can't translate Hindi/Kannada sentences to English before predicting |

## Methodology

If no relation holds predict NA.
NA not counted in F1 scores.

Training files also have Named Entity Recognition data

Q1 & Q2 use a smaller 1.5B model, suitable for fine-tuning with a classification head
Q3 uses a larger 8B instruct model, no fine-tuning, just inference via vllm

### Micro F1
**Intuition:** Pool all predictions together, then compute one global F1.

```
Imagine you have 3 relation types: 
/business/person/company  → 1000 examples  (frequent) 
/location/city/country    →  500 examples  (medium) 
/sports/player/team       →   10 examples  (rare)
```
Micro-F1 counts raw TP, FP, FN across all classes, so the 1000-example class dominates. A model that's great on frequent classes but terrible on rare ones will still score high.

### Macro F1
**Intuition:** Compute F1 separately for each class, then take a simple average. Rare classes get equal weight.
```
F1(/business/person/company) = 0.90
F1(/location/city/country)   = 0.80
F1(/sports/player/team)      = 0.20
                               ────
Macro-F1                     = 0.63  ← rare class drags it down!
```

| | Micro-F1 | Macro-F1 |
|--|----------|----------|
| Weighting | By instance count | Equal per class |
| Dominated by | Frequent classes | Rare classes |
| Good model must... | Get common relations right | Get all relations right |
| Use when... | Overall accuracy matters | Class balance matters |

Evaluation script ignores extra pairs. Only GT pairs are scored. \
So model is penalized for missing pairs but not rewarded for inventing extra ones — which discourages hallucination.

# Task 1
Qwen is a decoder-only transformer (like GPT) made by Alibaba. The 1.5B version has 1.5 billion parameters. "Decoder-only" means it reads left to right and predicts the next token.
It is trained on massive multilingual web text — including Hindi, Kannada, and other Indian languages \

When you feed it a sentence, it produces hidden states — one vector per token:

```
Input:  "[E1] India [/E1] contains [E2] Hyderabad [/E2]"
         tok1  tok2  tok3   tok4    tok5    tok6    tok7

Hidden states (from last layer):
         h1    h2    h3     h4      h5      h6      h7
         
Each h = vector of size 1536 (Qwen2.5-1.5B hidden dim)
```

## What vector do you feed into the classifier?
### Option A
Use h_last and feed to classifier. \
**Problem**: the last token represents the whole sentence, not specifically about E1 and E2. It loses entity-specific info.

### Option B
Concatenate E1 and E2 token states. \
Why this works: by the time the model processes [E1], it has attended to everything before it (the sentence context). So h_[E1] encodes "what E1 means in this sentence". Same for E2. Concatenating both gives the classifier:

- Info about E1 in context
- Info about E2 in context
- Implicitly, the relationship between them

### Option C
Concatenate all four boundary tokens.

```
[h_[E1] | h_[/E1] | h_[E2] | h_[/E2]] → classifier
```

This captures both the start and end context of each entity span.

# Task 3
The defining constraint of Task 3 is that there are no gradient updates; you cannot "train" or fine-tune the model. Instead, you must guide the model to the correct answer by providing high-quality instructions and examples within the input text itself.

To succeed in this task, we need to focus on Prompt Engineering and Example Selection.

Model can read 128k tokens.

## Demonstration Selection
- Balancing the number of examples with the model's context window limits
- Deciding whether to use English examples, target-language examples, or a mix for unseen languages like Oriya
- Comparing random sampling, stratified selection (by relation type), and similarity-based retrieval using FAISS

## FAISS
- How it works: You convert your test sentence into a "vector" (a list of numbers representing its meaning).

- The goal: You search your training data (like the English or Hindi sets) for the sentences that are most "similar" to your test sentence and use those as your few-shot demonstrations.

## Few shot learning
- examples you include in your prompt to show the model how to perform a task

## Cross lingual
Even though "apple" in English and "सेब" (seb) in Hindi look completely different, a multilingual model maps them to nearly the same point in a mathematical "meaning space" 📍. This allows us to compare the intent of a sentence rather than just the words

## Input

Input Sentence: "उत्तरी कैरोलिना ईस्टर्न म्यूजिक फेस्टिवल ग्रीन्सबोरो , 25 जून-30 जुलाई।" \
Query Entities: Entity 1: "उत्तरी कैरोलिना", Entity 2: "ग्रीन्सबोरो" \
Output JSON: {"relationMentions": [{"em1Text": "उत्तरी कैरोलिना", "em2Text": "ग्रीन्सबोरो", "label": "/स्थान/स्थान/शामिल_है"}]}