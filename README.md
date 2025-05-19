# Telegram Ukraine-Related Message Classifier

This repository contains code, data pointers, and models for identifying Ukraine-related messages in Russian-language Telegram posts. It supports keyword-based filtering, embedding via RuBERT, centroid-based expansion, and combined strategies, with separate training and evaluation scripts.

## Project Structure
```
project/
├── README.md
├── requirements.txt
├── dev_test_data.csv
├── keywords/
│   ├── small.txt
│   ├── medium.txt
│   ├── large.txt
│   └── xlarge.txt
├── centroids/
│   ├── telegram_centroid_ukraine_small.joblib
│   ├── telegram_centroid_ukraine_medium.joblib
│   ├── telegram_centroid_ukraine_large.joblib
│   └── telegram_centroid_ukraine_xlarge.joblib
├── model.py            # compute and save centroids
└── results.py          # tune thresholds and evaluate
```

## Prerequisites
- Python 3.8+  
- Internet access for initial dataset and model downloads

Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Sources
1. **ScoutieAutoML Dataset**  
   Hugging Face `ScoutieAutoML/russian-news-telegram-dataset` (train split) — up to ~5 months old at experiment time.
2. **Live Scraper Stream**  
   Continuously ingests fresh Telegram messages via Bot API:  
   - Dashboard: https://telegrasc.fly.dev  
   - JSON API: `https://telegrasc.fly.dev/messages`
3. **Manual Dev/Test Set**  
   `dev_test_data.csv` — 400 annotated messages (∼45% Ukraine-related) from the live scraper, used for tuning and evaluation.

## Generating Centroids
Run:
```bash
python model.py
```
This script will:
1. Load and label data from `dev_test_data.csv`.  
2. Load keyword sets from the `keywords/` folder.  
3. Embed messages with RuBERT (`cointegrated/rubert-tiny2`).  
4. Compute and save centroid vectors for each keyword set into `centroids/`.

## Evaluation
Run:
```bash
python results.py
```
This script will:
1. Load dev/test split from `dev_test_data.csv`.  
2. Load centroids from `centroids/`.  
3. Tune best centroid-only and hybrid thresholds on the dev set.  
4. Evaluate baseline (keywords), centroid-only, and hybrid methods on the test set.  
5. Print threshold values and a summary results table.

## Combined Filtering Strategies
- **Keyword only** — matches any lemmatized keyword  
- **Centroid only** — within cosine-distance radius  
- **Hybrid** — either condition  

Thresholds are tuned on the 400-message dev set (∼45% positives).

## Usage Example
```python
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model and centroid
sbert = SentenceTransformer("cointegrated/rubert-tiny2")
centroid = joblib.load("centroids/telegram_centroid_ukraine_medium.joblib")["centroid"]

# Embed new text
emb = sbert.encode(["Your message here"])
distance = np.linalg.norm(emb - centroid, axis=1)
# Compare against chosen threshold
```

## Contributing
1. Fork this repo  
2. Create a feature branch  
3. Submit a pull request  

## License
This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
