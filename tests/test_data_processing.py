
import csv
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import data_acquisition as da


def write_sample_dataset(path, rows):
    """Write sample records using the original UCI column format."""
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)


@pytest.fixture
def sample_dataset(tmp_path, monkeypatch):
    raw_file = tmp_path / "sample.data"
    clean_file = tmp_path / "heart_clean.csv"

    rows = [
        [54, 1, 2, 130, 246, 0, 1, 150, 0, 1.0, 2, 0, 3, 0],
        [60, 0, 3, 140, 250, 0, 0, 140, 1, 2.0, 2, "?", 3, 2],
        [45, 1, 1, 120, 220, 0, 1, 160, 0, 0.5, 1, 0, 3, "?"],
    ]

    write_sample_dataset(raw_file, rows)

    monkeypatch.setattr(da, "RAW_FILE", raw_file)
    monkeypatch.setattr(da, "CLEAN_FILE", clean_file)

    return raw_file, clean_file


def test_clean_dataset_creates_csv(sample_dataset):
    _, clean_file = sample_dataset

    da.clean_dataset()

    assert clean_file.exists()


def test_clean_dataset_has_expected_columns(sample_dataset):
    _, clean_file = sample_dataset

    da.clean_dataset()
    df = pd.read_csv(clean_file)

    assert list(df.columns) == da.COLUMNS
    assert len(df.columns) == 14


def test_clean_dataset_converts_target_to_binary(sample_dataset):
    _, clean_file = sample_dataset

    da.clean_dataset()
    df = pd.read_csv(clean_file)

    assert set(df["target"].unique()) == {0, 1}
    assert df["target"].tolist() == [0, 1]


def test_clean_dataset_drops_missing_targets(sample_dataset):
    _, clean_file = sample_dataset

    da.clean_dataset()
    df = pd.read_csv(clean_file)

    assert len(df) == 2
    assert df["target"].notna().all()


def test_clean_dataset_preserves_missing_features(sample_dataset):
    _, clean_file = sample_dataset

    da.clean_dataset()
    df = pd.read_csv(clean_file)

    assert pd.isna(df.loc[1, "ca"])


def test_download_dataset_downloads_when_missing(tmp_path, monkeypatch):
    raw_file = tmp_path / "download.data"
    monkeypatch.setattr(da, "RAW_FILE", raw_file)

    def fake_urlretrieve(url, destination):
        Path(destination).write_text("sample dataset")

    monkeypatch.setattr(da, "urlretrieve", fake_urlretrieve)

    da.download_dataset()

    assert raw_file.exists()
    assert raw_file.read_text() == "sample dataset"


def test_download_dataset_rejects_empty_file(tmp_path, monkeypatch):
    raw_file = tmp_path / "empty.data"
    raw_file.write_text("")

    monkeypatch.setattr(da, "RAW_FILE", raw_file)

    with pytest.raises(ValueError, match="empty"):
        da.download_dataset()
