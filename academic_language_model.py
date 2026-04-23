# =========================================
# STEP 1 — IMPORTS
# =========================================

import os
import re
import time
import json
import psutil
import numpy as np
import pandas as pd

from collections import Counter, defaultdict
from farasa.segmenter import FarasaSegmenter


# =========================================
# STEP 2 — MONITOR
# =========================================

def monitor(step, start):
    print(f"\n[{step}] Time: {time.time()-start:.2f}s | CPU: {psutil.cpu_percent()}% | RAM: {psutil.virtual_memory().percent}%")


# =========================================
# STEP 3 — LOAD DATASET
# =========================================

start = time.time()

dataset_paths = [
    r"documents\corpus-txt",
    r"documents\texts-txt",
    r"documents\texts-txt\msa\كتب حديثة",
     r"documents\texts-txt\msa\منوع",
    r"documents\texts-txt\msa\enfal.de"


]

articles = []

for path in dataset_paths:
    print(f"\nLoading from: {path}")

    if not os.path.exists(path):
        print("❌ Path not found:", path)
        continue

    files = [f for f in os.listdir(path) if f.endswith(".txt")]
    print("Files:", len(files))

    for file in files:
        with open(os.path.join(path, file), "r", encoding="utf-8") as f:
            articles.append(f.read())

print("Total articles:", len(articles))

monitor("Dataset Loaded", start)


# =========================================
# STEP 4 — NORMALIZATION
# =========================================

start = time.time()

def normalize(text):
    text = re.sub(r"[ًٌٍَُِّْـ]", "", text)
    text = re.sub(r"(.)\1+", r"\1", text)
    return text

articles = [normalize(a) for a in articles]

monitor("Normalization Done", start)


# =========================================
# STEP 5 — FARASA
# =========================================

start = time.time()
segmenter = FarasaSegmenter(interactive=False)
monitor("Farasa Initialized", start)


# =========================================
# STEP 6 — CLEAN + CONTROLLED STEMMING
# =========================================

def clean(tokens):
    return [re.sub(r"[^\w+]", "", t).strip() for t in tokens if t.strip()]


def stem(token):

    # SAFE PREFIXES
    safe_prefixes = ["وال", "بال", "كال", "فال", "لل", "ال"]
    for p in safe_prefixes:
        if token.startswith(p) and len(token) > len(p) + 2:
            token = token[len(p):]
            return token

    # WEAK PREFIXES (controlled)
    weak_prefixes = ["و", "ف"]
    for p in weak_prefixes:
        if token.startswith(p) and len(token) > 4:
            token = token[len(p):]
            return token

    # SUFFIXES
    suffixes = ["يات", "ات", "ون", "ين", "ان", "ة", "ه", "ها", "ك", "ي"]
    for s in suffixes:
        if token.endswith(s) and len(token) > len(s) + 2:
            token = token[:-len(s)]
            return token

    return token


# =========================================
# STEP 7 — TOKENIZATION
# =========================================

start = time.time()

tokenized_articles = []

for i, article in enumerate(articles):

    print(f"Processing {i+1}/{len(articles)}")

    article = article[:20000]

    segmented = segmenter.segment(article)
    tokens = segmented.split()

    tokens = clean(tokens)
    tokens = [stem(t) for t in tokens]

    tokenized_articles.append(tokens)

monitor("Tokenization + Stemming", start)


# =========================================
# STEP 8 — VOCABULARY
# =========================================

start = time.time()

all_tokens = [t for a in tokenized_articles for t in a]

word_counts = Counter(all_tokens)
vocab = sorted(word_counts.keys())

print("Vocabulary size:", len(vocab))

with open("vocabulary.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False)

monitor("Vocabulary Built", start)


# =========================================
# STEP 9 — NEXT WORD MODEL
# =========================================

start = time.time()

transitions = defaultdict(Counter)

for article in tokenized_articles:
    for i in range(len(article)-1):
        transitions[article[i]][article[i+1]] += 1

prob = {}

for w1, next_words in transitions.items():
    total = sum(next_words.values())
    prob[w1] = {w2: c/total for w2, c in next_words.items()}

with open("next_word_probabilities.json", "w", encoding="utf-8") as f:
    json.dump(prob, f, ensure_ascii=False)

monitor("Model Built", start)


# =========================================
# STEP 10 — MATRIX
# =========================================

start = time.time()

TOP_N = 500
top_words = [w for w, _ in word_counts.most_common(TOP_N)]
idx = {w:i for i,w in enumerate(top_words)}

matrix = np.zeros((TOP_N, TOP_N))

for article in tokenized_articles:
    for i in range(len(article)-1):
        w1, w2 = article[i], article[i+1]
        if w1 in idx and w2 in idx:
            matrix[idx[w1], idx[w2]] += 1

for i in range(TOP_N):
    s = matrix[i].sum()
    if s > 0:
        matrix[i] /= s

pd.DataFrame(matrix, index=top_words, columns=top_words)\
  .to_csv("word_word_probability_matrix.csv", encoding="utf-8-sig")

monitor("Matrix Built", start)

print("\nDONE ✅")