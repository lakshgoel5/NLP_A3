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