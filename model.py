import json
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, average_precision_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from preprocessing import get_feature_types, make_preprocessor
from utils import (
    ROOT,
    TRAIN_CSV,
    TEST_CSV,
    RANDOM_STATE,
    C_FP,
    C_FN,
    expected_cost,
    split_xy,
    find_best_threshold,
    eval_at_threshold,
    metrics,
)

warnings.filterwarnings("ignore")

# CV settings (for threshold selection only, never touching test)
N_SPLITS = 5


# -----------------------------
# Helpers
# -----------------------------
def safe_filename(s: str, max_len: int = 50) -> str:
    """Make a safe, short filename token from any object/string."""
    s = str(s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_.-]", "", s)
    return s[:max_len] if len(s) > max_len else s

def cost_at_threshold(y_true, y_prob, t, c_fp=C_FP, c_fn=C_FN):
    return float(expected_cost(y_true, y_prob, t, c_fp, c_fn))

def best_recall_under_precision(y_true, y_prob, min_precision=0.7):
    thresholds = np.linspace(0, 1, 501)
    best = None  # (recall, precision, threshold)

    for t in thresholds:
        stats = eval_at_threshold(y_true, y_prob, t)
        p = stats["precision"]
        r = stats["recall"]
        if p >= min_precision:
            if best is None or r > best[0]:
                best = (r, p, float(t))

    return best  # None if constraint is impossible

def plot_pr_curve(y_true, y_prob, title, outpath):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title} (AP={ap:.4f})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_cost_curve(costs, thresholds, best_t, title, outpath):
    plt.figure()
    plt.plot(thresholds, costs)
    plt.axvline(best_t, linestyle="--")
    plt.xlabel("Threshold")
    plt.ylabel("Expected Cost")
    plt.title(f"{title} (best t={best_t:.3f})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


# -----------------------------
# Pipeline factory (shared with soc_triage_agent)
# -----------------------------
def get_pipeline(name: str, num_cols: list, cat_cols: list, spw: float) -> Pipeline:
    """Build a single pipeline by name; same definitions as main() models dict."""
    pre_scaled = make_preprocessor(num_cols, cat_cols, scale_numeric=True)
    pre_unscaled = make_preprocessor(num_cols, cat_cols, scale_numeric=False)

    if name == "logreg":
        return Pipeline(
            steps=[
                ("pre", pre_scaled),
                ("clf", LogisticRegression(
                    max_iter=4000,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                )),
            ]
        )
    if name == "svm_linear":
        return Pipeline(
            steps=[
                ("pre", pre_scaled),
                ("clf", CalibratedClassifierCV(
                    estimator=LinearSVC(
                        class_weight="balanced",
                        random_state=RANDOM_STATE
                    ),
                    method="sigmoid",
                    cv=3
                )),
            ]
        )
    if name == "xgboost":
        return Pipeline(
            steps=[
                ("pre", pre_unscaled),
                ("clf", XGBClassifier(
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    min_child_weight=1.0,
                    scale_pos_weight=spw,
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                    n_jobs=-1,
                )),
            ]
        )
    if name == "lightgbm":
        return Pipeline(
            steps=[
                ("pre", pre_unscaled),
                ("clf", LGBMClassifier(
                    n_estimators=600,
                    learning_rate=0.05,
                    num_leaves=63,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    scale_pos_weight=spw,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )),
            ]
        )
    raise ValueError(f"Unknown model '{name}'.")


# -----------------------------
# EDA + Data Quality
# -----------------------------
def run_eda(train_df: pd.DataFrame, test_df: pd.DataFrame):
    print("\n=== DATA QUALITY CHECKS ===")
    print("Train shape:", train_df.shape)
    print("Test  shape:", test_df.shape)

    print("\nMissing values (train, top 15):")
    print(train_df.isnull().sum().sort_values(ascending=False).head(15))

    print("\nDuplicate rows:")
    print("Train duplicates:", int(train_df.duplicated().sum()))
    print("Test  duplicates:", int(test_df.duplicated().sum()))

    print("\nLabel distribution (train):")
    print(train_df["label"].value_counts())
    print(train_df["label"].value_counts(normalize=True))

    # Plot: class imbalance
    plt.figure(figsize=(6, 4))
    train_df["label"].value_counts().plot(kind="bar")
    plt.title("Class Distribution (0 = Normal, 1 = Attack)")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("eda_class_distribution.png", dpi=160)
    plt.close()

    # Plot: attack categories
    if "attack_cat" in train_df.columns:
        plt.figure(figsize=(8, 5))
        train_df["attack_cat"].value_counts().plot(kind="bar")
        plt.title("Attack Category Distribution")
        plt.xlabel("Attack Category")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig("eda_attack_categories.png", dpi=160)
        plt.close()

    # Numeric feature distributions (a small curated set if present)
    candidate_num = ["dur", "sbytes", "dbytes", "sttl", "dttl"]
    present = [c for c in candidate_num if c in train_df.columns]

    for col in present:
        plt.figure(figsize=(6, 4))
        plt.hist(train_df[col], bins=50)
        plt.title(f"Distribution: {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"eda_hist_{col}.png", dpi=160)
        plt.close()

        # Train vs test overlay
        if col in test_df.columns:
            plt.figure(figsize=(6, 4))
            plt.hist(train_df[col], bins=50, alpha=0.6, label="train")
            plt.hist(test_df[col], bins=50, alpha=0.6, label="test")
            plt.title(f"Train vs Test: {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"eda_train_vs_test_{col}.png", dpi=160)
            plt.close()

    print("\nSaved EDA plots:")
    print(" - eda_class_distribution.png")
    if "attack_cat" in train_df.columns:
        print(" - eda_attack_categories.png")
    for col in present:
        print(f" - eda_hist_{col}.png")
        print(f" - eda_train_vs_test_{col}.png")


# -----------------------------
# CV threshold selection
# -----------------------------
def oof_probabilities(pipeline, X, y, cv):
    """
    Generate out-of-fold probabilities on training data.
    Used ONLY for threshold selection without touching the test set.
    """
    oof = np.zeros_like(y, dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr = y[tr_idx]

        # Clone pipeline per fold so we don't overwrite a single fitted estimator
        model = clone(pipeline)
        model.fit(X_tr, y_tr)
        oof[va_idx] = model.predict_proba(X_va)[:, 1]

        print(f"  Fold {fold}/{cv.get_n_splits()} done")

    return oof


# -----------------------------
# Main
# -----------------------------
def main():
    # Load data (official split is our holdout boundary)

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # EDA first (saves plots)
    run_eda(train_df, test_df)

    # Split
    X_train, y_train = split_xy(train_df)
    X_test, y_test = split_xy(test_df)

    # Feature types
    num_cols, cat_cols = get_feature_types(X_train)
    print("\n=== FEATURE TYPES ===")
    print("Numeric:", len(num_cols))
    print("Categorical:", len(cat_cols))

    # Class imbalance for tree weighting
    neg = int(np.sum(y_train == 0))
    pos = int(np.sum(y_train == 1))
    spw = float(neg / max(pos, 1))
    print(f"\nTrain class counts: neg={neg}, pos={pos}, scale_pos_weight={spw:.3f}")

    # CV on training only
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # Models (RBF SVM removed due to runtime at this scale)
    model_names = ["logreg", "svm_linear", "xgboost", "lightgbm"]
    models = {name: get_pipeline(name, num_cols, cat_cols, spw) for name in model_names}

    print("MODEL KEYS:", list(models.keys()))

    results = {
        "data": {
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
            "train_pos_rate": float(np.mean(y_train)),
            "test_pos_rate": float(np.mean(y_test)),
        },
        "split_strategy": (
            "Used official UNSW-NB15 predefined training/testing split as the final holdout. "
            "All CV and threshold selection performed only on training set to avoid leakage."
        ),
        "cost_model": {"C_FP": C_FP, "C_FN": C_FN},
        "models": {},
    }

    print("\n=== TRAIN + CV THRESHOLD SELECTION (TRAIN ONLY) ===")
    for name, pipe in models.items():
        nm = safe_filename(name)

        print(f"\nModel: {name}")
        print(" Generating OOF probabilities for threshold selection...")
        oof = oof_probabilities(pipe, X_train, y_train, cv=cv)

        best_t, costs, ths = find_best_threshold(y_train, oof, C_FP, C_FN)

        # Training cost curve
        plot_cost_curve(costs, ths, best_t, f"Train Cost Curve - {name}", f"cost_curve_train_{nm}.png")

        print(" Fitting final model on full training set...")
        pipe.fit(X_train, y_train)

        prob_test = pipe.predict_proba(X_test)[:, 1]

        # --- Cost reduction vs baseline threshold 0.5 (on TEST for reporting only) ---
        baseline_t = 0.5
        baseline_cost = cost_at_threshold(y_test, prob_test, baseline_t)
        opt_cost = cost_at_threshold(y_test, prob_test, best_t)

        cost_reduction = (baseline_cost - opt_cost) / (baseline_cost + 1e-12)

        print(f" Baseline cost @ t=0.5: {baseline_cost:.0f}")
        print(f" Optimized cost @ t*={best_t:.3f}: {opt_cost:.0f}")
        print(f" Cost reduction vs baseline: {100*cost_reduction:.2f}%")

# --- Policy point: maximize recall subject to precision >= 0.70 ---
        policy = best_recall_under_precision(y_test, prob_test, min_precision=0.70)
        if policy is None:
            print(" Policy point: No threshold achieves precision >= 0.70")
        else:
            r, p, t_pol = policy
            print(f" Policy point (max recall s.t. precision>=0.70): t={t_pol:.3f}, precision={p:.3f}, recall={r:.3f}")

       

        # Test PR curve
        plot_pr_curve(y_test, prob_test, f"PR Curve (Test) - {name}", f"pr_curve_test_{nm}.png")

        # Test cost curve (threshold fixed from train)
        _, costs_test, ths_test = find_best_threshold(y_test, prob_test, C_FP, C_FN)
        plot_cost_curve(
            costs_test, ths_test, best_t,
            f"Test Cost Curve - {name} (threshold from train)",
            f"cost_curve_test_{nm}.png"
        )

        m = metrics(y_test, prob_test)
        thr_eval = eval_at_threshold(y_test, prob_test, best_t)

        results["models"][name] = {
            "threshold_selected_on_train_oof": float(best_t),
            "test_metrics": m,
            "test_at_threshold": thr_eval,
            "artifacts": {
                "train_cost_curve": f"cost_curve_train_{nm}.png",
                "test_pr_curve": f"pr_curve_test_{nm}.png",
                "test_cost_curve": f"cost_curve_test_{nm}.png",
            },
        }

        print(" Test PR-AUC:", m["pr_auc"])
        print(" Test ROC-AUC:", m["roc_auc"])
        print(" Test Brier:", m["brier"])
        print(" Selected threshold (train OOF):", best_t)
        print(" Test precision/recall at threshold:",
              f"precision={thr_eval['precision']:.4f}, recall={thr_eval['recall']:.4f}")

    with open("results_models_deep_dive.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== DONE ===")
    print("Saved:")
    print(" - results_models_deep_dive.json")
    print(" - EDA plots: eda_*.png")
    print(" - PR curves: pr_curve_test_*.png")
    print(" - Cost curves: cost_curve_train_*.png and cost_curve_test_*.png")


if __name__ == "__main__":
    main()