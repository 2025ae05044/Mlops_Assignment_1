
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data" / "processed" / "heart_clean.csv"
PLOT_DIR = ROOT / "report" / "eda_plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

# Load cleaned dataset
df = pd.read_csv(DATA_FILE)

print("=" * 50)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 50)

print("\nDataset shape:", df.shape)
print("\nData types:")
print(df.dtypes)

print("\nDescriptive statistics:")
print(df.describe().T.round(2))

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# 1. Feature histograms
features = ["age", "trestbps", "chol", "thalach", "oldpeak"]

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()

for ax, feature in zip(axes, features):
    sns.histplot(
        data=df,
        x=feature,
        bins=20,
        kde=True,
        ax=ax
    )
    ax.set_title(f"Distribution of {feature}")
    ax.set_xlabel(feature)

axes[-1].set_visible(False)

fig.suptitle("Heart Disease Dataset: Feature Distributions")
fig.tight_layout()

fig.savefig(
    PLOT_DIR / "feature_histograms.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close(fig)

# 2. Correlation heatmap
correlation = df.corr(numeric_only=True)

fig, ax = plt.subplots(figsize=(13, 10))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    ax=ax
)

ax.set_title("Heart Disease Feature Correlation Heatmap")
fig.tight_layout()

fig.savefig(
    PLOT_DIR / "correlation_heatmap.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close(fig)

# 3. Target class balance
counts = df["target"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    ["No disease (0)", "Disease present (1)"],
    [counts.get(0, 0), counts.get(1, 0)]
)

ax.set_title("Heart Disease Target Class Balance")
ax.set_xlabel("Target class")
ax.set_ylabel("Number of records")

ax.bar_label(bars)

fig.tight_layout()

fig.savefig(
    PLOT_DIR / "class_balance.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close(fig)

print("\nEDA visualizations generated successfully.")
print("Output directory:", PLOT_DIR)

for image_name in [
    "feature_histograms.png",
    "correlation_heatmap.png",
    "class_balance.png"
]:
    print("-", image_name)

print("\nEDA completed.")
