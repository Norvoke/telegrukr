"""
Evaluate keyword, centroid, and hybrid classification strategies on Telegram messages.

Inputs:
  - CSV data: “dev_test_data.csv” with columns:
      • Raw Message: the text to classify
      • Classified: binary ground-truth label (0/1)
  - Keyword files: “keywords/{small,medium,large,xlarge}.txt”
  - Centroid files: “centroids/telegram_centroid_ukraine_{size}.joblib”

Outputs:
  - Prints best centroid‐only and hybrid thresholds for each keyword set
  - Builds `final_df`: a pandas DataFrame summarizing accuracy, precision, recall, and F1
    for baseline, centroid‐only, and hybrid methods on the test split
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
import pymorphy2
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
DATA_PATH    = "dev_test_data.csv"
KEYWORD_DIR  = "keywords"
CENTROID_DIR = "centroids"
DEV_SIZE     = 200
THRESHOLDS   = np.arange(0.1, 1.0, 0.1)

# Initialize resources
model = SentenceTransformer("cointegrated/rubert-tiny2")
morph = pymorphy2.MorphAnalyzer()

# Load and prepare data
df = pd.read_csv(DATA_PATH)
df["embedding"] = df["Raw Message"].apply(lambda x: model.encode([x])[0])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
dev, test = df[:DEV_SIZE], df[DEV_SIZE:]

def load_keywords(path=KEYWORD_DIR):
    """Parse all .txt files into {size: set(keywords)}."""
    kws = {}
    for fn in os.listdir(path):
        if fn.endswith(".txt"):
            size = fn[:-4]
            text = open(os.path.join(path, fn), encoding="utf-8").read()
            obj  = text.split("=",1)[-1].strip()
            kws[size] = set(eval(obj))
    return kws

def load_centroids(path=CENTROID_DIR):
    """Load joblib files into {size: centroid_array}."""
    cents = {}
    for fn in os.listdir(path):
        if fn.endswith(".joblib"):
            size = fn.split("_")[-1].replace(".joblib","")
            cents[size] = joblib.load(os.path.join(path, fn))["centroid"]
    return cents

def lemmatize(text):
    """Return list of pymorphy2 lemmas for a given text string."""
    return [morph.parse(w)[0].normal_form
            for w in re.findall(r"\w+", text.lower())]

def lemmatize_keywords(keywords):
    """Convert each multiword keyword into its normalized lemma sequence."""
    result = set()
    for kw in keywords:
        lem = [morph.parse(w)[0].normal_form for w in kw.split()]
        result.add(" ".join(lem))
    return result

def classify_keyword(msg, kw_lemmas):
    """1 if any keyword-lemma sequence appears in message lemmas."""
    lemmas = lemmatize(msg)
    for kw in kw_lemmas:
        length = len(kw.split())
        for i in range(len(lemmas)-length+1):
            if " ".join(lemmas[i:i+length]) == kw:
                return 1
    return 0

def evaluate(df_subset, method, centroid=None, threshold=None, kw_lemmas=None):
    """Generic evaluation for 'keyword', 'centroid', or 'hybrid'."""
    y_true, preds = df_subset["Classified"].values, []
    for _, row in df_subset.iterrows():
        k_pred = classify_keyword(row["Raw Message"], kw_lemmas) if kw_lemmas else 0
        c_sim  = cosine_similarity([row["embedding"]], [centroid])[0,0] if centroid is not None else 0
        if method == "keyword":
            preds.append(k_pred)
        elif method == "centroid":
            preds.append(int(c_sim > threshold))
        else:  # hybrid
            preds.append(int(k_pred or (c_sim > threshold)))
    return {
        met: score_func(y_true, preds)
        for met, score_func in [
            ("accuracy", accuracy_score),
            ("precision", precision_score),
            ("recall", recall_score),
            ("f1", f1_score)
        ]
    }

# Main loop: tune and evaluate
keywords  = load_keywords()
centroids = load_centroids()
results   = []

for size in ["small","medium","large","xlarge"]:
    cen   = centroids[size]
    kw_lem = lemmatize_keywords(keywords[size])

    best_c_th = max(THRESHOLDS, key=lambda t: evaluate(dev, "centroid", cen, t)["f1"])
    best_h_th = max(THRESHOLDS, key=lambda t: evaluate(dev, "hybrid",   cen, t, kw_lem)["f1"])
    print(f"[{size.upper()}] centroid-only → {best_c_th:.2f}, hybrid → {best_h_th:.2f}")

    row = {"size": size}
    row.update({f"baseline_{k}": v for k,v in evaluate(test,  "keyword",  None, None,   kw_lem).items()})
    row.update({f"centroid_{k}": v for k,v in evaluate(test, "centroid", cen, best_c_th, None).items()})
    row.update({f"hybrid_{k}": v for k,v in evaluate(test,    "hybrid",   cen, best_h_th, kw_lem).items()})
    results.append(row)

final_df = pd.DataFrame(results)
print(final_df)
