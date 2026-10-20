
import logging
import time

from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, ConfigDict

from src.predict import predict_patient

# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger("heart_disease_api")


# Prometheus metrics
REQUEST_COUNT = Counter(
    "heart_disease_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

REQUEST_DURATION = Histogram(
    "heart_disease_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)

# Create the FastAPI application
app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "MLOps assignment API for heart disease classification. "
        "Predictions are for educational use, not medical advice."
    ),
    version="1.0.0",
)

# Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# Log request method, endpoint, status, and response duration.
# Do not log patient input values.

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    status_code = 500
    endpoint = request.url.path

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response

    except Exception:
        logger.exception(
            "Request failed: method=%s path=%s",
            request.method,
            endpoint,
        )
        raise

    finally:
        duration_seconds = time.perf_counter() - start_time

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status=str(status_code),
        ).inc()

        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=endpoint,
        ).observe(duration_seconds)

        logger.info(
            "method=%s path=%s status=%s duration_ms=%.2f",
            request.method,
            endpoint,
            status_code,
            duration_seconds * 1000,
        )



# Validate the 13 required input features.
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


# Root endpoint
@app.get("/")
def home():
    return {
        "message": "Heart Disease Prediction API",
        "docs": "/docs",
        "health": "/health",
    }


# Health check endpoint
@app.get("/health")
def health():
    return {"status": "healthy"}


# Prediction endpoint
@app.post("/predict")
def predict(patient: PatientInput):
    try:
        result = predict_patient(patient.model_dump())
        return result

    except ValueError as exc:
        logger.warning("Prediction input rejected: %s", str(exc))
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:
        logger.exception("Trained model file is missing")
        raise HTTPException(
            status_code=500,
            detail="Trained model file is missing.",
        ) from exc
