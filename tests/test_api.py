from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

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


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["health"] == "/health"
    assert response.json()["docs"] == "/docs"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_endpoint_returns_prediction():
    response = client.post("/predict", json=VALID_INPUT)

    assert response.status_code == 200

    result = response.json()
    assert result["prediction"] in (0, 1)
    assert result["label"] in ("no_disease", "disease_present")
    assert 0 <= result["confidence"] <= 1


def test_predict_rejects_missing_feature():
    invalid_input = VALID_INPUT.copy()
    invalid_input.pop("age")

    response = client.post("/predict", json=invalid_input)

    assert response.status_code == 422


def test_predict_rejects_unexpected_feature():
    invalid_input = VALID_INPUT.copy()
    invalid_input["unexpected_feature"] = 123

    response = client.post("/predict", json=invalid_input)

    assert response.status_code == 422


def test_predict_rejects_non_numeric_feature():
    invalid_input = VALID_INPUT.copy()
    invalid_input["age"] = "not-a-number"

    response = client.post("/predict", json=invalid_input)

    assert response.status_code == 422
