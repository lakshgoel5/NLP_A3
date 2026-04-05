Even though I have taken help from the following students and LLMs in terms of discussing ideas and coding practices, all my code is written by me.

Students: 
- Neelkanth Mishra

LLMs: 
- Gemini

# Approach
## Q1
It is a supervised classification problem. Given a sentence and two highlighted entities, classify what relationship holds between them.

Classes of relations are given in English in file _____, and their translations are given in the respective languages.

Why Unsupervised data? This is for Pre-training on large unlabelled corpus to learn the language

Oriya and Tulu have very little data compared to Hindi/Kannada. They should be treated as zero/few-shot languages. My model must generalize to them without being able to rely on meaningful training signal.

Hindi & Kannada — few-shot (small labelled + large unlabelled)

Base model weights are frozen. Only train LoRA adapter matrices and The linear classification head on top.

## Q2

The core shift from Q1 is that instead of classifying, the model has to *generate* the answer as text. So instead of a classification head reading hidden states, we now use the full LM head and train the model to produce a JSON string like `{"label": "/location/location/contains"}` given the sentence and the two entities.

Because we're generating, we don't need the special entity marker tokens ([E1], [E2] etc.) anymore. The entities are just mentioned explicitly in the prompt. The prompt I went with is:

```
Sentence: <sentence>
Entity 1: <em1>
Entity 2: <em2>
Output: 
```

And the model is trained to complete this with `{"label": "<english_label>"}`. I chose JSON output because it makes post-processing clean and it's what the assignment asks for.

**Loss masking.** During training, we don't want the model to "learn" the fixed prompt template — it's the same every time and memorizing it wastes capacity. So we compute loss only on the completion tokens (the JSON part) and mask the prompt tokens with -100. This is standard SFT practice.

**Label language.** During training the model always predicts English labels, regardless of what language the sentence is in. Native labels from Hindi/Kannada files are translated back to English using the inverse of the provided map files. At inference time, once we have the English label, we translate it forward to whatever the target language is. This keeps training simple — one consistent output format — and the mapping is just a dictionary lookup at the end.

**Joint training.** I train on all available data together — English (56k) + Hindi (200) + Kannada (200) + Oriya (89) + Tulu (95). Since Indic data is tiny compared to English, I repeat Indic examples 5x so they aren't completely drowned out. Validation is done on the English val set to track loss.

Results (to be filled after run):
| Language | Macro-F1 | Micro-F1 |
|----------|----------|----------|
| English  |          |          |
| Hindi    |          |          |
| Kannada  |          |          |
| Oriya    |          |          |
| Tulu     |          |          |

**Oriya and Tulu.** These are the hard languages — very little labeled data and no large pre-training corpus available (unlike Hindi/Kannada which have Wikipedia). They are essentially zero-shot languages. The hope is that the model has seen enough multilingual text during its own pre-training (Qwen2.5 is trained on multilingual data) that it can generalize. We still include the small or/tcy training sets in Stage 2 to give it some signal.

**Post-processing.** Generation isn't guaranteed to produce exactly a valid label. The post-processing does: (1) try to JSON-parse the output and extract the "label" key, (2) if that fails, search for any known label string occurring anywhere in the output and pick the longest match, (3) if nothing matches, default to NA. The substring fallback handles cases where the model wraps extra text around the JSON.