from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from src.predict import predict_patient

app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "MLOps assignment API for heart disease classification. "
        "Predictions are for educational use, not medical advice."
    ),
    version="1.0.0",
)


class PatientInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


@app.get("/")
def home():
    return {
        "message": "Heart Disease Prediction API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(patient: PatientInput):
    try:
        return predict_patient(patient.model_dump())
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail="Trained model file is missing.",
        ) from exc
