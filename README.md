# Nirikshana — Fake vs Real News Detector

A machine learning pipeline that classifies news text as **Fake** or **Real** based on writing style and linguistic patterns, with a Streamlit demo for real-time predictions.

> **Note:** this model detects *stylistic patterns* typical of fake vs. real news writing (tone, phrasing, structure) — it does not fact-check claims or verify events against real-world sources. See [Future Work](#future-work) for planned real-time verification.

## Demo

Paste any news headline or article into the Streamlit app and get an instant prediction with a confidence score.

```bash
uv run streamlit run app.py
```

## Pipeline

```
Raw text
  → clean_text()          (lowercase, remove stopwords, punctuation, special chars, extra whitespace)
  → TF-IDF vectorization  (max_features=20000, min_df=5, max_df=0.9)
  → Trained classifier    (best of 4 models, selected by F1 score)
  → Prediction + confidence
```

## Model Comparison

Four models were trained and evaluated on the same TF-IDF features to identify the best performer:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **XGBoost** | 0.9879 | 0.9929 | 0.9826 | **0.9877** |
| LinearSVC | 0.9859 | 0.9855 | 0.9861 | 0.9858 |
| Logistic Regression | 0.9786 | 0.9801 | 0.9767 | 0.9784 |
| Multinomial Naive Bayes | 0.9218 | 0.9284 | 0.9129 | 0.9206 |

**XGBoost** was selected as the production model based on highest F1 score, and is the model saved/deployed in this repo. Full results are also saved to `models/model_comparison.csv` after each training run.

## Tech Stack

- **Preprocessing:** NLTK (stopwords), regex-based cleaning
- **Feature extraction:** scikit-learn TF-IDF
- **Models:** scikit-learn (Naive Bayes, Logistic Regression, LinearSVC), XGBoost
- **Serving:** joblib (model/vectorizer persistence), Streamlit
- **Environment:** uv

## Setup

```bash
uv venv
uv add pandas numpy scikit-learn nltk xgboost joblib streamlit
uv run python -c "import nltk; nltk.download('stopwords')"
```

**Dataset:** place `dataset.csv` (columns: `text`, `label` — where `label` is `1` for real, `0` for fake) into a `data/` folder at the project root. Not included in this repo due to file size (~120MB).

## Running

**Train (retrains all 4 models, saves the best one):**
```bash
uv run python -m src.train
```

**CLI prediction:**
```bash
uv run python -m src.predict
```

**Streamlit demo:**
```bash
uv run streamlit run app.py
```

## Project Structure

```
├── data/                     # dataset (gitignored — see Setup)
├── notebooks/
│   └── fake_vs_real_news.ipynb   # original EDA + experimentation notebook
├── src/
│   ├── preprocessing.py     # clean_text() — shared by training and inference
│   ├── train.py              # trains + compares 4 models, saves the best
│   └── predict.py            # loads saved model, predicts on new text
├── models/
│   ├── best_model.pkl        # trained XGBoost model
│   ├── tfidf_vectorizer.pkl  # fitted vectorizer (required for inference)
│   └── model_comparison.csv  # full metrics for all 4 models
├── app.py                    # Streamlit UI
└── requirements handled via pyproject.toml / uv
```

## Future Work

The current model is purely stylistic — it can be fooled by real news written in a sensational tone, and can miss fabricated news written in a flat, formal style. Planned enhancement: a **hybrid verification pipeline** that combines this model's stylistic signal with real-time web search (extracting the article's core claim, searching for corroborating/contradicting coverage from reputable sources, and weighing evidence strength against the stylistic verdict) rather than relying on writing style alone.
