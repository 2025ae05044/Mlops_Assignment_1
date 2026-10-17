# Task 4: Model Packaging and Reproducibility

## Model Serialization

The selected Logistic Regression pipeline and its preprocessing transformations were saved as `models/best_model_pipeline.joblib`. Separate pipelines were also saved for Logistic Regression and Random Forest.

The preprocessing pipeline includes numeric median imputation, feature scaling, categorical most-frequent imputation, and one-hot encoding.

## Inference

The reusable inference script is `src/predict.py`. Its `predict_patient()` function accepts a patient record containing the 13 required input features and returns a predicted binary class, label, and confidence score.

## Dependencies

The project uses `requirements.txt` to list its direct Python dependencies for data processing, model training and inference, visualization, experiment tracking, and automated testing.

## Verification

- The saved model pipeline was loaded successfully by the inference script.
- A sample prediction was generated successfully.
- The prediction returned a binary class, a label, and a confidence score.
- Automated prediction tests passed.
- Clean-environment installation and Docker verification remain to be completed.

## Reproducibility Considerations

Training and inference use a saved preprocessing pipeline to keep data transformations consistent. Model artifacts, data files, source scripts, reports, and configuration files are stored inside the persistent project workspace.

Dependency versions will be pinned after compatibility verification. A clean-environment test and container build will be performed as part of the subsequent MLOps tasks.
