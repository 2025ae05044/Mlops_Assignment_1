
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

METRICS_FILE = ROOT / "report" / "model_metrics.json"
COMPARISON_FILE = ROOT / "report" / "model_comparison.csv"
MODEL_DIR = ROOT / "models"
PLOT_DIR = ROOT / "report" / "eda_plots"


TRACKING_DIR = ROOT / "mlruns"
TRACKING_DIR.mkdir(parents=True, exist_ok=True)

# Use SQLite for MLflow tracking.
DB_PATH = ROOT / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
mlflow.set_experiment("Heart Disease Classification")


with open(METRICS_FILE, encoding="utf-8") as file:
    summary = json.load(file)

comparison = pd.read_csv(COMPARISON_FILE)

for result in summary["results"]:
    name = result["model"]

    if name == "LogisticRegression":
        model_path = MODEL_DIR / "logisticregression_pipeline.joblib"
    elif name == "RandomForest":
        model_path = MODEL_DIR / "randomforest_pipeline.joblib"
    else:
        raise ValueError(f"Unexpected model: {name}")

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    with mlflow.start_run(run_name=name):
        # Log the actual parameters and evaluation results.
        mlflow.log_params(result["best_parameters"])
        mlflow.log_param("model_name", name)
        mlflow.log_param("selection_metric", "mean_cv_roc_auc")
        mlflow.log_param("cv_folds", summary["cv_folds"])
        mlflow.log_param("test_size", summary["test_size"])
        mlflow.log_param("random_state", summary["random_state"])

        for metric_name in [
            "best_cv_roc_auc",
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_roc_auc",
        ]:
            mlflow.log_metric(
                metric_name,
                float(result[metric_name])
            )

        # Preserve the original training comparison as an artifact.
        mlflow.log_artifact(str(COMPARISON_FILE))

        # Attach EDA plots when they exist.
        if PLOT_DIR.exists():
            for plot_file in PLOT_DIR.glob("*.png"):
                mlflow.log_artifact(str(plot_file), artifact_path="eda")

        # Log the saved preprocessing + estimator pipeline.
        mlflow.log_artifact(
            str(model_path),
            artifact_path="saved_pipeline"
        )

        mlflow.set_tag("tracking_type", "recorded_training_results")

        print(f"Logged recorded results for {name}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")

print("\nExperiment tracking records created.")
print(f"MLflow tracking directory: {TRACKING_DIR}")
print("Note: these runs log previously completed training results.")
