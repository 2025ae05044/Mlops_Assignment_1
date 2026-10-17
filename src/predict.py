
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT / "models" / "best_model_pipeline.joblib"

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]


def predict_patient(patient):
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_FILE}"
        )

    missing = set(FEATURES) - set(patient)
    extra = set(patient) - set(FEATURES)

    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unexpected features: {sorted(extra)}")

    model = joblib.load(MODEL_FILE)
    X = pd.DataFrame([{feature: patient[feature] for feature in FEATURES}])

    prediction = int(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]

    return {
        "prediction": prediction,
        "label": "disease_present" if prediction == 1 else "no_disease",
        "confidence": round(float(probabilities[prediction]), 4)
    }


if __name__ == "__main__":
    sample_patient = {
        "age": 54,
        "sex": 1,
        "cp": 2,
        "trestbps": 130,
        "chol": 246,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 0,
        "thal": 3
    }

    print(predict_patient(sample_patient))
