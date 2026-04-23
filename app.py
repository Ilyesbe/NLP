import os
import re
import csv
import json
import numpy as np
from collections import Counter, defaultdict
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_corpus():
    dataset_paths = [
        os.path.join(BASE_DIR, "documents", "corpus-txt"),
        os.path.join(BASE_DIR, "documents", "texts-txt"),
    ]
    articles = []
    for path in dataset_paths:
        if not os.path.exists(path):
            continue
        for f in os.listdir(path):
            if f.endswith(".txt"):
                try:
                    text = open(os.path.join(path, f), encoding="utf-8").read()
                    if text.strip():
                        articles.append(text[:15000])
                except:
                    continue
    return articles


def normalize(text):
    text = re.sub(r"[ً-ْـ]", "", text)
    text = re.sub(r"(.)\1+", r"\1", text)
    return text


def load_matrix_csv(filepath):
    if not os.path.exists(filepath):
        return None, None
    words = []
    rows = []
    with open(filepath, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        col_words = header[1:]
        for row in reader:
            words.append(row[0])
            rows.append([float(x) if x else 0.0 for x in row[1:]])
    return words, np.array(rows, dtype=np.float32)


def compute_statistics(articles):
    stats = {}

    stats["total_documents"] = len(articles)
    total_chars = sum(len(a) for a in articles)
    stats["total_characters"] = total_chars
    stats["avg_doc_length"] = round(total_chars / max(len(articles), 1))

    normalized = [normalize(a) for a in articles]
    norm_chars = sum(len(a) for a in normalized)
    stats["chars_before_norm"] = total_chars
    stats["chars_after_norm"] = norm_chars
    stats["norm_reduction_pct"] = round((1 - norm_chars / max(total_chars, 1)) * 100, 2)

    all_tokens = []
    sentences_all = []
    for a in normalized:
        tokens = a.split()
        all_tokens.extend(tokens)
        sents = re.split(r'[.!?؟،\n]+', a)
        sents = [s.strip() for s in sents if s.strip()]
        sentences_all.extend(sents)

    stats["total_words"] = len(all_tokens)
    stats["unique_words"] = len(set(all_tokens))
    stats["vocab_reduction_pct"] = round((1 - len(set(all_tokens)) / max(len(all_tokens), 1)) * 100, 2)

    char_counter = Counter()
    for a in normalized:
        for ch in a:
            if '؀' <= ch <= 'ۿ':
                char_counter[ch] += 1
    stats["arabic_char_count"] = sum(char_counter.values())
    stats["unique_arabic_chars"] = len(char_counter)

    sent_lengths = [len(s.split()) for s in sentences_all]
    stats["total_sentences"] = len(sent_lengths)
    if sent_lengths:
        stats["avg_sentence_length"] = round(sum(sent_lengths) / len(sent_lengths), 2)
        stats["max_sentence_length"] = max(sent_lengths)
        stats["min_sentence_length"] = min(sent_lengths)
    else:
        stats["avg_sentence_length"] = 0
        stats["max_sentence_length"] = 0
        stats["min_sentence_length"] = 0

    return stats, all_tokens, char_counter, sent_lengths, normalized


def histogram(data, bins):
    counts = [0] * (len(bins) - 1)
    for val in data:
        for i in range(len(bins) - 1):
            if bins[i] <= val < bins[i + 1]:
                counts[i] += 1
                break
        else:
            if val == bins[-1]:
                counts[-1] += 1
    return counts


def build_autoencoder_data(matrix_words, matrix_data, top_n=80):
    if matrix_data is None or len(matrix_data) == 0:
        return None, None, None
    n = min(top_n, len(matrix_data))
    words = matrix_words[:n]
    X = matrix_data[:n, :n].copy()

    input_dim = X.shape[1]
    latent_dim = 2
    hidden_dim = max(16, input_dim // 4)

    np.random.seed(42)
    W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.1
    b1 = np.zeros(hidden_dim, dtype=np.float32)
    W2 = np.random.randn(hidden_dim, latent_dim).astype(np.float32) * 0.1
    b2 = np.zeros(latent_dim, dtype=np.float32)
    W3 = np.random.randn(latent_dim, hidden_dim).astype(np.float32) * 0.1
    b3 = np.zeros(hidden_dim, dtype=np.float32)
    W4 = np.random.randn(hidden_dim, input_dim).astype(np.float32) * 0.1
    b4 = np.zeros(input_dim, dtype=np.float32)

    lr = 0.01
    losses = []

    for epoch in range(300):
        h1 = np.maximum(0, X @ W1 + b1)
        z = h1 @ W2 + b2
        h3 = np.maximum(0, z @ W3 + b3)
        out = h3 @ W4 + b4

        loss = float(np.mean((X - out) ** 2))
        losses.append(loss)

        d_out = 2 * (out - X) / X.shape[0]
        dW4 = h3.T @ d_out
        db4 = d_out.sum(axis=0)
        d_h3 = d_out @ W4.T
        d_h3[h3 <= 0] = 0
        dW3 = z.T @ d_h3
        db3 = d_h3.sum(axis=0)
        d_z = d_h3 @ W3.T
        dW2 = h1.T @ d_z
        db2 = d_z.sum(axis=0)
        d_h1 = d_z @ W2.T
        d_h1[h1 <= 0] = 0
        dW1 = X.T @ d_h1
        db1 = d_h1.sum(axis=0)

        for p, dp in [(W1, dW1), (b1, db1), (W2, dW2), (b2, db2),
                       (W3, dW3), (b3, db3), (W4, dW4), (b4, db4)]:
            p -= lr * np.clip(dp, -1, 1)

    h1 = np.maximum(0, X @ W1 + b1)
    z = h1 @ W2 + b2
    h3 = np.maximum(0, z @ W3 + b3)
    reconstructed = h3 @ W4 + b4
    final_loss = float(np.mean((X - reconstructed) ** 2))

    return words, z.tolist(), {"losses": losses, "final_loss": final_loss}


# =========================================
# LOAD ALL DATA AT STARTUP
# =========================================

print("Loading corpus...")
articles = load_corpus()
print(f"Loaded {len(articles)} articles")

print("Computing statistics...")
stats, all_tokens, char_counter, sent_lengths, normalized_articles = compute_statistics(articles)

prob_model = {}
for model_path in ["optimized_model.json", "next_word_probabilities.json"]:
    full = os.path.join(BASE_DIR, model_path)
    if os.path.exists(full):
        with open(full, encoding="utf-8") as f:
            prob_model = json.load(f)
        print(f"Loaded model from {model_path}: {len(prob_model)} words")
        break

matrix_words, matrix_data = None, None
for mpath in ["optimized_matrix.csv", "word_word_probability_matrix.csv"]:
    full = os.path.join(BASE_DIR, mpath)
    if os.path.exists(full):
        matrix_words, matrix_data = load_matrix_csv(full)
        if matrix_data is not None:
            print(f"Loaded matrix from {mpath}: {matrix_data.shape}")
        break

word_freq = Counter(all_tokens)

print("Training autoencoder...")
ae_words, ae_coords, ae_info = build_autoencoder_data(matrix_words, matrix_data, top_n=80)
print("Autoencoder done.")

matrix_shape_str = f"{matrix_data.shape[0]}x{matrix_data.shape[1]}" if matrix_data is not None else "Non disponible"

pipeline_steps = [
    {"step": "Chargement du corpus",
     "detail": f"{stats['total_documents']} documents, {stats['total_characters']:,} caractères",
     "pct": 100},
    {"step": "Normalisation",
     "detail": f"Réduction de {stats['norm_reduction_pct']}% des caractères (diacritiques supprimés)",
     "pct": stats["norm_reduction_pct"]},
    {"step": "Tokenisation",
     "detail": f"{stats['total_words']:,} mots totaux, {stats['unique_words']:,} uniques",
     "pct": round(stats["unique_words"] / max(stats["total_words"], 1) * 100, 2)},
    {"step": "Vocabulaire",
     "detail": f"Réduction: {stats['vocab_reduction_pct']}% (mots répétés éliminés)",
     "pct": stats["vocab_reduction_pct"]},
    {"step": "Modèle de probabilité",
     "detail": f"{len(prob_model):,} mots avec transitions",
     "pct": round(len(prob_model) / max(stats["unique_words"], 1) * 100, 2)},
    {"step": "Matrice de probabilité",
     "detail": matrix_shape_str,
     "pct": 100 if matrix_data is not None else 0},
]


# =========================================
# ROUTES
# =========================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stats")
def api_stats():
    return jsonify({"stats": stats, "pipeline_steps": pipeline_steps})


@app.route("/api/word_freq")
def api_word_freq():
    top_n = int(request.args.get("n", 30))
    top = word_freq.most_common(top_n)
    return jsonify({
        "words": [w for w, _ in top],
        "counts": [c for _, c in top],
        "total": sum(word_freq.values()),
    })


@app.route("/api/char_heatmap")
def api_char_heatmap():
    arabic_chars = sorted(char_counter.keys(), key=lambda c: -char_counter[c])[:40]
    cooccurrence = defaultdict(lambda: defaultdict(int))

    for article in normalized_articles[:200]:
        for i in range(len(article) - 1):
            c1, c2 = article[i], article[i + 1]
            if c1 in arabic_chars and c2 in arabic_chars:
                cooccurrence[c1][c2] += 1

    chars = arabic_chars[:25]
    matrix = []
    for c1 in chars:
        row = []
        for c2 in chars:
            row.append(cooccurrence[c1][c2])
        matrix.append(row)

    max_val = max(max(r) for r in matrix) if matrix and any(matrix) else 1
    matrix_norm = [[round(v / max_val, 4) for v in row] for row in matrix]

    return jsonify({"chars": chars, "matrix": matrix_norm, "matrix_raw": matrix})


@app.route("/api/sentence_length")
def api_sentence_length():
    if not sent_lengths:
        return jsonify({"bins": [], "counts": [], "stats": {}})

    max_len = min(max(sent_lengths), 100)
    bin_size = max(1, max_len // 20)
    bins = list(range(0, max_len + bin_size + 1, bin_size))
    counts = histogram(sent_lengths, bins)

    return jsonify({
        "bins": [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)],
        "counts": counts,
        "stats": {
            "moyenne": stats["avg_sentence_length"],
            "max": stats["max_sentence_length"],
            "min": stats["min_sentence_length"],
            "total_phrases": stats["total_sentences"],
        }
    })


@app.route("/api/predict")
def api_predict():
    word = request.args.get("word", "").strip()
    if not word or word not in prob_model:
        suggestions = [w for w in prob_model if w.startswith(word)][:10] if word else []
        return jsonify({"found": False, "suggestions": suggestions, "predictions": []})

    next_words = prob_model[word]
    sorted_preds = sorted(next_words.items(), key=lambda x: -x[1])[:15]

    return jsonify({
        "found": True,
        "word": word,
        "predictions": [{"word": w, "probability": round(p * 100, 2)} for w, p in sorted_preds],
        "total_transitions": len(next_words),
    })


@app.route("/api/autoencoder")
def api_autoencoder():
    if ae_words is None:
        return jsonify({"available": False})
    return jsonify({
        "available": True,
        "words": ae_words,
        "coords": ae_coords,
        "losses": ae_info["losses"],
        "final_loss": round(ae_info["final_loss"], 6),
    })


@app.route("/api/vocab_search")
def api_vocab_search():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"results": []})
    results = []
    for w in prob_model:
        if q in w:
            freq = word_freq.get(w, 0)
            n_transitions = len(prob_model.get(w, {}))
            results.append({"word": w, "frequency": freq, "transitions": n_transitions})
            if len(results) >= 20:
                break
    results.sort(key=lambda x: -x["frequency"])
    return jsonify({"results": results})


if __name__ == "__main__":
    print("\n=== NLP Dashboard Ready ===")
    print("http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
