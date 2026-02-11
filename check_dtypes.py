"""Quick script to inspect column dtypes in UNSW-NB15 training data."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
TRAIN_CSV = "UNSW_NB15_training-set.csv"


def main():
    train = pd.read_csv(TRAIN_CSV)
    X = train.drop(columns=["label", "id"], errors="ignore")

    print("Column dtypes:")
    for col in X.columns:
        dt = str(X[col].dtype)
        sample = repr(X[col].iloc[0])
        print(f"{col:20} {dt:15} Sample: {sample}")
        if X[col].dtype == "object":
            print("  → IS CATEGORICAL")


if __name__ == "__main__":
    main()
