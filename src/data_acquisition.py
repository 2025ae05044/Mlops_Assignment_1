
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

# Project root directory
ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# UCI Heart Disease - Cleveland processed dataset
DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/heart-disease/"
    "processed.cleveland.data"
)

RAW_FILE = RAW_DIR / "processed.cleveland.data"
CLEAN_FILE = PROCESSED_DIR / "heart_clean.csv"

COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


def download_dataset():
    """Download the dataset from the UCI repository."""
    if not RAW_FILE.exists():
        print("Downloading Heart Disease dataset...")

        try:
            urlretrieve(DATA_URL, RAW_FILE)
        except Exception as exc:
            raise RuntimeError(
                f"Dataset download failed: {exc}"
            ) from exc

    if RAW_FILE.stat().st_size == 0:
        raise ValueError("Downloaded dataset is empty.")

    print(f"Raw dataset saved at: {RAW_FILE}")


def clean_dataset():
    """Clean data and convert the original target to binary."""
    df = pd.read_csv(
        RAW_FILE,
        header=None,
        names=COLUMNS,
        na_values=["?"],
        skipinitialspace=True,
    )

    if df.shape[1] != 14:
        raise ValueError(
            f"Expected 14 columns, received {df.shape[1]}."
        )

    # Convert columns to numeric values.
    for column in COLUMNS:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    # Remove rows without a valid target.
    df = df.dropna(subset=["target"]).copy()

    # Original target: 0 = no disease, 1-4 = disease present.
    if not df["target"].isin([0, 1, 2, 3, 4]).all():
        raise ValueError("Unexpected original target values.")

    df["target"] = (df["target"] > 0).astype(int)

    # Keep missing feature values for later pipeline imputation.
    df.to_csv(CLEAN_FILE, index=False)

    print("\nDataset cleaning completed.")
    print(f"Number of records: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\nTarget distribution:")
    print(df["target"].value_counts().sort_index())

    print(f"\nCleaned dataset saved at: {CLEAN_FILE}")


if __name__ == "__main__":
    download_dataset()
    clean_dataset()
