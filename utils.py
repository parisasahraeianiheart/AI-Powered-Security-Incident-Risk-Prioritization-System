"""
Shared data and evaluation utilities for UNSW-NB15 ML pipeline.
"""

from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    brier_score_loss,
)

# -----------------------------
# Config (single source of truth)
# -----------------------------
ROOT = Path(__file__).resolve().parent
TRAIN_CSV = "UNSW_NB15_training-set.csv"
TEST_CSV = "UNSW_NB15_testing-set.csv"

RANDOM_STATE = 42
C_FP = 1.0   # false alarm investigation cost
C_FN = 20.0  # missed incident cost


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Split dataframe into features X and label y; drop id and attack_cat."""
    if "label" not in df.columns:
        raise ValueError("Could not find 'label' column.")
    y = df["label"].astype(int).values
    X = df.drop(columns=["label"], errors="ignore")
    drop_cols = [c for c in ["id", "attack_cat"] if c in X.columns]
    if drop_cols:
        X = X.drop(columns=drop_cols)
    return X, y


def expected_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    c_fp: float = C_FP,
    c_fn: float = C_FN,
) -> float:
    """Expected cost at a given decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return c_fp * fp + c_fn * fn


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    c_fp: float = C_FP,
    c_fn: float = C_FN,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Find threshold minimizing expected cost; return best_t, costs, thresholds."""
    thresholds = np.linspace(0, 1, 501)
    costs = np.array([expected_cost(y_true, y_prob, t, c_fp, c_fn) for t in thresholds])
    best_idx = int(np.argmin(costs))
    return float(thresholds[best_idx]), costs, thresholds


def eval_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Precision, recall, FPR at a given threshold (2x2 confusion matrix)."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    fpr = fp / (fp + tn + 1e-12)
    return {
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
    }


def metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """PR-AUC, ROC-AUC, Brier score."""
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }
