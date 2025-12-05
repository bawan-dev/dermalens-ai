import os
import sys

# Make it possible to import `src.*` even when run as a module
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from src.preprocessing import join_ingredients_for_model


# --------------------------
# PATHS
# --------------------------
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "ingredients_multilabel.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "tfidf_multiclass_model.joblib")


# --------------------------
# LOAD + CLEAN DATA
# --------------------------
def load_data():
    print(f"📥 Loading dataset from: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Drop incomplete rows
    df = df.dropna(subset=["ingredients", "label"])
    df["ingredients"] = df["ingredients"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    # Preprocess each ingredient list for modelling
    df["text_for_model"] = df["ingredients"].apply(join_ingredients_for_model)

    print(f"✅ Loaded {len(df)} rows.")
    print("Classes:", df["label"].unique())
    return df


# --------------------------
# BUILD PIPELINE
# --------------------------
def build_model():
    """
    TF-IDF + Logistic Regression for 10 classes.
    """
    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=20000,
                    min_df=2,
                    max_df=0.95,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=600,
                    multi_class="multinomial",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return pipeline


# --------------------------
# TRAIN + EVALUATE MODEL
# --------------------------
def train_and_evaluate():
    df = load_data()

    X = df["text_for_model"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("🧠 Building TF-IDF + Logistic Regression pipeline...")
    model = build_model()

    print("🚀 Training model...")
    model.fit(X_train, y_train)

    print("📊 Evaluating model...\n")
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\n✅ Model saved to: {MODEL_PATH}")


# --------------------------
# MAIN
# --------------------------
if __name__ == "__main__":
    train_and_evaluate()
