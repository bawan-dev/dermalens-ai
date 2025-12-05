import os
import joblib

from src.preprocessing import join_ingredients_for_model
from src.safety_score import calculate_safety_score

MODEL_PATH = os.path.join("models", "tfidf_logreg_model.joblib")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train it first.")
    return joblib.load(MODEL_PATH)


def predict_safety(ingredients_text: str):
    model = load_model()
    clean_text = join_ingredients_for_model(ingredients_text)

    pred_label = model.predict([clean_text])[0]
    pred_proba = model.predict_proba([clean_text])[0]

    classes = model.classes_
    proba_dict = {cls: float(p) for cls, p in zip(classes, pred_proba)}

    # calculate fungal acne score
    score = calculate_safety_score(ingredients_text)

    return pred_label, proba_dict, score


if __name__ == "__main__":
    text = input("Paste ingredients:\n> ")
    label, probs, score = predict_safety(text)

    print("\n🔍 Prediction")
    print("Label:", label)
    print("Probabilities:")
    for cls, p in probs.items():
        print(f"  {cls}: {p:.3f}")

    print(f"\n🧪 Fungal Acne Safety Score: {score}/10")
