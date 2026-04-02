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

## Methodology

If no relation holds predict NA.
NA not counted in F1 scores.

Training files also have Named Entity Recognition data

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