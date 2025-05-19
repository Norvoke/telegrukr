import os
import ast
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import joblib

def load_keywords(size):
    """
    Load keywords from keywords/{size}.txt.
    Supports files that are either:
      • A simple newline list of quoted or unquoted words, or
      • A Python set literal, e.g. ukraine_keywords = { "word1", "word2", … }
    Returns a set of lowercase strings.
    """
    path = os.path.join("keywords", f"{size}.txt")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Try to find and eval the set literal after an '='
    if "=" in text:
        try:
            rhs = text.split("=", 1)[1]
            kw_set = ast.literal_eval(rhs)
            return {str(w).lower() for w in kw_set}
        except Exception:
            pass

    # Fallback: treat each non-blank line as one keyword
    kws = set()
    for line in text.splitlines():
        line = line.strip().strip('",\'')
        if line:
            kws.add(line.lower())
    return kws

def is_ukraine_example(example, keywords):
    """
    True if any token or lemma in example['ners'] matches keywords.
    Safely handles example.get('ners') == None.
    """
    for ner in example.get("ners") or []:
        if ner.get("ner", "").lower() in keywords or ner.get("lemma", "").lower() in keywords:
            return True
    return False

def main():
    # 1) Load dataset
    ds = load_dataset("ScoutieAutoML/russian-news-telegram-dataset", split="train")
    texts = ds["text"]

    # 2) Embed all texts once
    model = SentenceTransformer("cointegrated/rubert-tiny2")
    embeddings = model.encode(texts, batch_size=8, show_progress_bar=True)
    model.save("sbert_ukraine_model/")

    # 3) For each keyword‐set size, build & dump centroid
    for size in ("small", "medium", "large", "xlarge"):
        kw = load_keywords(size)
        labels = np.array([int(is_ukraine_example(ex, kw)) for ex in ds])
        pos_embs = embeddings[labels == 1]

        if pos_embs.size == 0:
            print(f"⚠️  Warning: no positive examples for '{size}' → skipping")
            continue

        centroid = pos_embs.mean(axis=0)
        out_path = f"telegram_centroid_ukraine_{size}.joblib"
        joblib.dump({"centroid": centroid}, out_path)
        print(f"→ Wrote centroid for '{size}' → {out_path}")

if __name__ == "__main__":
    main()
