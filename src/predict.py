import joblib
from src.preprocessing import clean_text

MODEL_PATH = "models/best_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict(text: str, model=None, vectorizer=None) -> dict:
    if model is None or vectorizer is None:
        model, vectorizer = load_artifacts()

    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    proba = model.predict_proba(vector)[0]

    return {
        "label": "Real" if prediction == 1 else "Fake",
        "confidence": float(max(proba)),
    }


if __name__ == "__main__":
    sample_text = input("Paste a news headline/article: ")
    result = predict(sample_text)
    print(f"\nPrediction: {result['label']} (confidence: {result['confidence']:.2%})")