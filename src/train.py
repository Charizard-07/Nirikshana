import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.preprocessing import clean_text

DATA_PATH = "data/dataset.csv"
MODEL_PATH = "models/best_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"


def get_models():
    return {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(),
        "LinearSVC": LinearSVC(),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric='logloss'
        ),
    }


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    print("Cleaning text...")
    df['text'] = df['text'].apply(clean_text)

    print("Splitting data...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )

    print("Vectorizing (TF-IDF)...")
    tfidf = TfidfVectorizer(max_features=20000, min_df=5, max_df=0.9)
    X_train = tfidf.fit_transform(X_train_text)
    X_test = tfidf.transform(X_test_text)

    results = []
    best_model = None
    best_f1 = -1
    best_name = None

    for name, model in get_models().items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1})
        print(f"  Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name

    print("\n=== Comparison Table ===")
    results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
    print(results_df.to_string(index=False))

    print(f"\nBest model: {best_name} (F1: {best_f1:.4f})")
    print("Saving best model + vectorizer...")
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(tfidf, VECTORIZER_PATH)
    print(f"Saved to {MODEL_PATH} and {VECTORIZER_PATH}")

    results_df.to_csv("models/model_comparison.csv", index=False)
    print("Comparison table saved to models/model_comparison.csv")


if __name__ == "__main__":
    main()