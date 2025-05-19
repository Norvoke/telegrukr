# Telegram Ukraine-Related Message Classifier

This repository contains all the code, data pointers, and models needed to reproduce our pipeline for identifying Ukraine-related messages in Russian-language Telegram posts. It includes keyword-based filtering, embedding via RuBERT-tiny-v2, centroid-based expansion, and combined strategies.

## Prerequisites
- Python 3.8+  
- Access to the internet for initial dataset download and model weights

Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Sources
1. **ScoutieAutoML Dataset**  
   Hugging Face `ScoutieAutoML/russian-news-telegram-dataset` (train split) for historic Telegram posts. Note: data is up to ~5 months old at time of experiments.
2. **Live Scraper Stream**  
   Continuously ingests fresh Telegram messages via the Bot API.  
   - Dashboard: https://telegrasc.fly.dev  
   - JSON API: `https://telegrasc.fly.dev/messages`
3. **Manual Dev/Test Set**  
   400 annotated messages (∼45% Ukraine-related) sampled from the live scraper for tuning and evaluation.

## Keyword Lists
Each of the four files in `keywords/` defines a set of lemmatized war-related terms:
- `small.txt`  
- `medium.txt`  
- `large.txt`  
- `xlarge.txt`  

These are loaded at runtime by `ru_model.py` using Python's `ast.literal_eval` (for set literals) or line-by-line parsing.

## Model Training
Run the `ru_model.py` script to:
1. Load and label the Hugging Face dataset using each keyword set.  
2. Embed all messages with `cointegrated/rubert-tiny2`.  
3. Compute the centroid vector over positively labeled embeddings per keyword size.  
4. Dump four joblib files: `telegram_centroid_ukraine_{size}.joblib`.

```bash
python ru_model.py
```

The first run will also save the SBERT model to `sbert_ukraine_model/`.

## Using the Models
Import your chosen centroid:
```python
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np

# Load SBERT and centroid
sbert = SentenceTransformer("cointegrated/rubert-tiny2")
centroid = joblib.load("telegram_centroid_ukraine_medium.joblib")["centroid"]

# Embed new messages
embs = sbert.encode(["Your text here"], show_progress_bar=False)

distance = np.linalg.norm(embs - centroid, axis=1)
# Compare with your tuned radius threshold
```

## Combined Filtering Strategies
We support three modes:
1. **Keyword only** — match any lemmatized keyword  
2. **Centroid only** — within cosine-distance radius  
3. **Hybrid** — either condition

Thresholds are tuned on our 400-message dev set (∼45% positives).

## Development & Testing
- Ensure `requirements.txt` is up to date.  
- Run `ru_model.py` to regenerate centroids after editing keywords.  
- Add unit tests under `tests/` as needed.

## Contributing
1. Fork this repo.  
2. Create a feature branch.  
3. Open a pull request with a clear description.

## License
This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
