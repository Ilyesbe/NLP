import os, re, time, json, psutil
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from farasa.segmenter import FarasaSegmenter


def monitor(step, start):
    print(f"[{step}] {time.time()-start:.2f}s | CPU {psutil.cpu_percent()}% | RAM {psutil.virtual_memory().percent}%")


segmenter = FarasaSegmenter(interactive=False)

paths = [r"documents\corpus-txt",
         r"documents\texts-txt", 
         r"documents\texts-txt\msa\كتب حديثة",
         r"documents\texts-txt\msa\منوع",
         r"documents\texts-txt\msa\enfal.de"
         ]

word_counts = Counter()
transitions = defaultdict(Counter)

MAX_LEN = 15000
MIN_FREQ = 3


def normalize(t):
    t = re.sub(r"[ًٌٍَُِّْـ]", "", t)
    return re.sub(r"(.)\1+", r"\1", t)


def clean(tokens):
    return [re.sub(r"[^\w+]", "", x).strip() for x in tokens if x.strip()]


def stem(token):

    safe_prefixes = ["وال","بال","كال","فال","لل","ال"]
    for p in safe_prefixes:
        if token.startswith(p) and len(token) > len(p)+2:
            return token[len(p):]

    weak_prefixes = ["و","ف"]
    for p in weak_prefixes:
        if token.startswith(p) and len(token) > 4:
            return token[len(p):]

    suffixes = ["يات","ات","ون","ين","ان","ة","ه","ها","ك","ي"]
    for s in suffixes:
        if token.endswith(s) and len(token) > len(s)+2:
            return token[:-len(s)]

    return token


# =========================================
# MAIN LOOP (STREAMING)
# =========================================

start = time.time()
count = 0

for path in paths:

    if not os.path.exists(path):
        print("❌ Missing:", path)
        continue

    files = [f for f in os.listdir(path) if f.endswith(".txt")]

    for f in files:
        count += 1

        try:
            text = open(os.path.join(path,f), encoding="utf-8").read()
        except:
            continue

        text = normalize(text[:MAX_LEN])

        try:
            tokens = segmenter.segment(text).split()
        except:
            continue

        tokens = clean(tokens)
        tokens = [stem(t) for t in tokens]

        word_counts.update(tokens)

        for i in range(len(tokens)-1):
            transitions[tokens[i]][tokens[i+1]] += 1

        if count % 200 == 0:
            print("Processed:", count)

monitor("Processed", start)


# =========================================
# FILTER VOCAB
# =========================================

vocab = {w for w,c in word_counts.items() if c >= MIN_FREQ}
print("Filtered vocab size:", len(vocab))


# =========================================
# MODEL
# =========================================

prob = {}

for w1, nxt in transitions.items():
    if w1 not in vocab:
        continue

    total = sum(c for w2,c in nxt.items() if w2 in vocab)
    if total == 0:
        continue

    prob[w1] = {w2:c/total for w2,c in nxt.items() if w2 in vocab}

json.dump(prob, open("optimized_model.json","w",encoding="utf-8"), ensure_ascii=False)


# =========================================
# MATRIX
# =========================================

TOP = 300
top = [w for w,_ in word_counts.most_common(TOP)]
idx = {w:i for i,w in enumerate(top)}

M = np.zeros((TOP,TOP))

for w1,nxt in transitions.items():
    if w1 not in idx:
        continue
    for w2,c in nxt.items():
        if w2 in idx:
            M[idx[w1],idx[w2]] += c

for i in range(TOP):
    s = M[i].sum()
    if s > 0:
        M[i] /= s

pd.DataFrame(M,index=top,columns=top)\
  .to_csv("optimized_matrix.csv",encoding="utf-8-sig")

print("\nOPTIMIZED DONE ✅")