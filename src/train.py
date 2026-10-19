import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "heart_clean.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "report"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# MLflow experiment tracking
TRACKING_DB = ROOT / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB}")
mlflow.set_experiment("Heart Disease Classification")

RANDOM_STATE = 42

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]

CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def build_preprocessor():
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def main():
    df = pd.read_csv(DATA_FILE)

    expected_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing_columns = set(expected_columns + ["target"]) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    X = df[expected_columns].copy()
    y = df["target"].astype(int)

    if y.nunique() != 2:
        raise ValueError("Expected a binary classification target.")

    # Hold out the test set before tuning or cross-validation.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    models = {
        "LogisticRegression": (
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
            {
                "model__C": [0.1, 1.0, 10.0],
            },
        ),
        "RandomForest": (
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_leaf": [1, 2],
            },
        ),
    }

    results = []
    fitted_models = {}


    for name, (estimator, parameters) in models.items():
        with mlflow.start_run(run_name=name):
            print(f"\n{'=' * 55}")
            print(f"Training and tuning: {name}")
            print("=" * 55)

            pipeline = Pipeline(
                [
                    ("preprocessor", build_preprocessor()),
                    ("model", estimator),
                ]
            )

            search = GridSearchCV(
                estimator=pipeline,
                param_grid=parameters,
                scoring="roc_auc",
                cv=cv,
                n_jobs=-1,
                refit=True,
                error_score="raise",
            )

            # Log experiment configuration before training.
            mlflow.log_param("model_name", name)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_param("test_size", 0.20)
            mlflow.log_param("scoring", "roc_auc")
            mlflow.log_params(
                {f"grid_{key}": str(value) for key, value in parameters.items()}
            )

            search.fit(X_train, y_train)
            best_model = search.best_estimator_

            predictions = best_model.predict(X_test)
            probabilities = best_model.predict_proba(X_test)[:, 1]

            metrics = {
                "model": name,
                "best_parameters": search.best_params_,
                "best_cv_roc_auc": float(search.best_score_),
                "test_accuracy": float(accuracy_score(y_test, predictions)),
                "test_precision": float(
                    precision_score(y_test, predictions, zero_division=0)
                ),
                "test_recall": float(
                    recall_score(y_test, predictions, zero_division=0)
                ),
                "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
            }

            # Log the best hyperparameters and evaluation metrics.
            mlflow.log_params(
                {f"best_{key}": value for key, value in search.best_params_.items()}
            )

            mlflow.log_metrics(
                {
                    key: value
                    for key, value in metrics.items()
                    if key.startswith(("best_cv_roc_auc", "test_"))
                }
            )

            results.append(metrics)
            fitted_models[name] = best_model

            print("Best parameters:", search.best_params_)
            print(f"Cross-validation ROC-AUC: {search.best_score_:.4f}")
            print("\nHeld-out test metrics:")
            for key in [
                "test_accuracy",
                "test_precision",
                "test_recall",
                "test_roc_auc",
            ]:
                print(f"{key}: {metrics[key]:.4f}")

            print("\nConfusion matrix:")
            print(confusion_matrix(y_test, predictions))

            print("\nClassification report:")
            print(classification_report(y_test, predictions, zero_division=0))

            # Save the model pipeline and log it as an artifact.
            model_path = MODEL_DIR / f"{name.lower()}_pipeline.joblib"
            joblib.dump(best_model, model_path)
            mlflow.log_artifact(
                str(model_path),
                artifact_path="saved_pipeline",
            )

            # Save evaluation details with this MLflow run.
            mlflow.log_dict(
                metrics,
                "evaluation_metrics.json",
            )

            mlflow.set_tag(
                "tracking_type",
                "training_and_evaluation",
            )

            print("MLflow run ID:", mlflow.active_run().info.run_id)

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False,
    )

    # Select by cross-validation ROC-AUC, not test-set performance.
    winner = max(results, key=lambda row: row["best_cv_roc_auc"])
    final_model = fitted_models[winner["model"]]

    joblib.dump(
        final_model,
        MODEL_DIR / "best_model_pipeline.joblib",
    )

    summary = {
        "selection_metric": "mean cross-validation ROC-AUC",
        "selected_model": winner["model"],
        "random_state": RANDOM_STATE,
        "test_size": 0.20,
        "cv_folds": 5,
        "results": results,
    }

    with open(
        REPORT_DIR / "model_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print("\n" + "=" * 55)
    print("MODEL COMPARISON")
    print("=" * 55)
    print(results_df.to_string(index=False))
    print("\nSelected model:", winner["model"])
    print("Saved reusable pipeline:", MODEL_DIR / "best_model_pipeline.joblib")
    print("Saved metrics:", REPORT_DIR / "model_metrics.json")
    print("Saved comparison:", REPORT_DIR / "model_comparison.csv")


if __name__ == "__main__":
    main()
