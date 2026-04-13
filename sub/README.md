Even though I have taken help from the following students and LLMs in terms of discussing ideas and coding practices, all my code is written by me.

Students: 
- Neelkanth Mishra

LLMs: 
- Gemini
- Claude

# Approach

## Q1 — Encoder + Classification Head (Qwen2.5-1.5B + LoRA)

### What I built
A relation classifier on top of a frozen encoder. The base model is Qwen/Qwen2.5-1.5B with LoRA adapters injected into the query and value projections. A linear classification head reads entity representations from the last hidden layer and predicts one of 24 relation classes.

### Entity representation
Rather than using the [CLS] token or sentence mean pooling, I wrap each entity mention with special tokens — `[E1]...[/E1]` and `[E2]...[/E2]` — inserted directly into the sentence text at the character positions of the mentions. At inference, I locate the positions of these four marker tokens in the token sequence and extract their hidden states. The final representation fed to the classifier is the concatenation of all four markers (`concat_four`), giving the model access to both the start and end boundaries of each entity. This was chosen over simpler alternatives (e.g. just start tokens, or mean pooling over the span) because it captures span extent explicitly.

### Continued Pre-training (CPT)
I experimented with continued pre-training on the Indic unsupervised corpora before fine-tuning, using 500 samples per language (Hindi, Kannada, Odia, Tulu) with a causal LM objective. The idea was to adapt Qwen's representations to the Indic scripts before the supervised phase. In practice, the downstream classification F1 did not improve — likely because 500 samples is too small to meaningfully shift the model's representations, and Qwen2.5 already has reasonable multilingual coverage. The CPT step was dropped and the final model is trained directly on the labeled data.

### Training
- **LoRA**: rank 32, alpha 64, targeting `q_proj` and `v_proj`. The base model weights are fully frozen; only the LoRA matrices and the classifier head are trained.
- **Joint multilingual training**: English (56k) + Hindi (200) + Kannada (200) + Odia (89) + Tulu (95). Indic examples are repeated 5× to prevent them being overwhelmed by English data.
- **Label mapping**: The Indic training files use native-language label strings. These are mapped back to English labels (using the inverse of the provided map files) before training, so the classifier always operates on a single consistent label space. At inference, the predicted English label is translated forward to the target language using the same map.
- **Odia and Tulu**: These are effectively zero-shot languages — the small labeled sets are included for a marginal signal but the model has to rely on the multilingual representations learned during Qwen's pre-training.

---

## Q2 — Generative SFT (Qwen2.5-1.5B + LoRA)

### What I built
A sequence-to-sequence formulation of the same task. Instead of a classification head, the model is fine-tuned to generate the relation label as plain text. The same base model and LoRA setup is used, but here the full causal LM head is kept and the model is trained with next-token prediction loss on the label string.

### Prompt format
```
Entity 1: <em1>
Entity 2: <em2>
Sentence: <sentence>
Relation: <label>
```
The entities are mentioned explicitly in the prompt, so there is no need for special marker tokens. The model is trained to complete "Relation: " with the English label string.

### Continued Pre-training (CPT)
Same experiment as Q1: continued pre-training on 500 samples per Indic language (Hindi, Kannada, Odia, Tulu) with a causal LM objective before supervised fine-tuning. The results did not improve over the baseline — the sample size is too small to shift token distributions meaningfully, and the generation loss on Indic text did not translate to better relation label generation. The CPT step was dropped; the final model is fine-tuned directly from the base Qwen2.5-1.5B checkpoint.

### Key design choices
- **Loss masking**: Loss is computed only on the label tokens (the completion), not on the prompt. The prompt template is fixed and uninformative for training — masking it (setting those positions to -100) focuses the gradient on what actually varies: the label.
- **Label language**: The model always predicts English labels regardless of sentence language. Indic labels from the training files are inverted to English using the provided maps before constructing training examples. Forward translation is applied at inference time as a dictionary lookup. This keeps the output space consistent across languages and avoids the model having to learn separate label vocabularies.
- **Indic oversampling**: Same 5× repeat as Q1, for the same reason.
- **Post-processing**: Generated text is snapped to the nearest valid label using prefix matching followed by difflib fuzzy matching as a fallback. This handles cases where the model generates minor variations or trailing tokens.

---

## Q3 — In-Context Learning (Meta-Llama-3.1-8B-Instruct)

### What I built
A zero-/few-shot inference pipeline using Meta-Llama-3.1-8B-Instruct served via vLLM. No fine-tuning is performed. For each test instance, a set of labeled demonstrations is retrieved from the training pool and formatted into a chat prompt, which the model completes with a relation label.

### Demo retrieval
Each test query (em1, em2, sentence) is encoded with `paraphrase-multilingual-MiniLM-L12-v2` and the k=8 nearest neighbors are retrieved from a FAISS index built over the entire demo pool. This retrieval model is multilingual — it maps semantically similar sentences from different languages close together in embedding space. This means that even for Odia or Tulu queries (where same-language training data is sparse), semantically relevant English and cross-lingual examples surface as demos.

The demo pool contains English training data plus all available Indic training files (hi, kn, or, tcy), with Indic labels mapped back to English for consistency. Loading all languages into the pool means that similarity retrieval can also surface same-script examples for Indic queries, which is better than falling back to random or stratified sampling.

### Prompt construction
A chat template is used (via `apply_chat_template`) with a system prompt that lists all 24 valid labels and instructs the model to output exactly one. Each demo is a user/assistant turn pair. The query is the final user turn.

For long Indic sentences, Indic scripts tokenize 2–4× more tokens per character than English, which can silently truncate prompts inside vLLM. To handle this, the prompt builder checks the tokenized length and drops the least-relevant demos (from the end) until the prompt fits within a 3500-token budget.

### Decoding
vLLM's guided decoding (`guided_choice`) is used to constrain the model's output to exactly one of the 24 valid labels. This eliminates the label-snapping post-processing step entirely — the model cannot generate an invalid string. A fallback to greedy sampling with stop tokens is included for older vLLM versions that do not support guided decoding.

### Cross-lingual generalization
The approach relies entirely on the multilingual representations of the embedding model (for retrieval) and Llama's multilingual pre-training (for relation understanding). No language-specific adaptation is performed at inference time beyond translating the predicted English label to the target language using the provided maps.
