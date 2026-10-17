
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predict import predict_patient

VALID_INPUT = {
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
    "thal": 3,
}


def test_model_file_exists():
    assert (ROOT / "models" / "best_model_pipeline.joblib").exists()


def test_valid_input_returns_result():
    result = predict_patient(VALID_INPUT)
    assert isinstance(result, dict)


def test_prediction_is_binary():
    result = predict_patient(VALID_INPUT)
    assert result["prediction"] in (0, 1)


def test_prediction_label_is_valid():
    result = predict_patient(VALID_INPUT)
    assert result["label"] in ("no_disease", "disease")


def test_confidence_is_between_zero_and_one():
    result = predict_patient(VALID_INPUT)
    assert 0 <= result["confidence"] <= 1


def test_missing_feature_is_rejected():
    invalid_input = VALID_INPUT.copy()
    invalid_input.pop("age")

    with pytest.raises((ValueError, KeyError, TypeError)):
        predict_patient(invalid_input)


def test_unexpected_feature_is_rejected():
    invalid_input = VALID_INPUT.copy()
    invalid_input["unexpected_feature"] = 123

    with pytest.raises((ValueError, KeyError, TypeError)):
        predict_patient(invalid_input)


def test_all_thirteen_features_are_present():
    assert len(VALID_INPUT) == 13
