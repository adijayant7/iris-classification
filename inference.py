"""
Iris Species Classifier – Inference Script
==========================================
Usage:
    python inference.py                       # runs built-in demo
    python inference.py 5.1 3.5 1.4 0.2      # sepal_l sepal_w petal_l petal_w
"""
import sys, joblib
import pandas as pd

MODEL_PATH   = "iris_best_model.joblib"
FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

def load_model(path: str = MODEL_PATH):
    bundle = joblib.load(path)
    return bundle["model"], bundle["scaler"], bundle["label_encoder"]

def predict(sepal_length: float, sepal_width: float,
            petal_length: float, petal_width: float,
            model_path: str = MODEL_PATH) -> dict:
    clf, scaler, le = load_model(model_path)
    sample = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=FEATURE_NAMES
    )
    scaled = scaler.transform(sample)
    label  = clf.predict(scaled)[0]
    proba  = clf.predict_proba(scaled)[0]
    return {
        "predicted_species": le.inverse_transform([label])[0],
        "probabilities": {cls: round(float(p), 4)
                          for cls, p in zip(le.classes_, proba)}
    }

if __name__ == "__main__":
    if len(sys.argv) == 5:
        vals = list(map(float, sys.argv[1:]))
    else:
        vals = [5.1, 3.5, 1.4, 0.2]   # default demo (should → Iris-setosa)
        print("No arguments given – running demo sample:", vals)

    result = predict(*vals)
    print(f"\n🌸 Predicted Species : {result['predicted_species']}")
    print("   Class Probabilities:")
    for cls, p in result["probabilities"].items():
        bar = "█" * int(p * 30)
        print(f"     {cls:<22} {p:.4f}  {bar}")
