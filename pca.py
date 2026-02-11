import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, brier_score_loss

from preprocessing import get_feature_types, make_preprocessor
from utils import (
    TRAIN_CSV,
    TEST_CSV,
    RANDOM_STATE,
    C_FP,
    C_FN,
    split_xy,
    find_best_threshold,
    expected_cost,
)

# PCA + SVM settings
N_COMPONENTS = 30          # try 20, 30, 50
C = 2.0                    # try 1.0, 2.0, 5.0
GAMMA = "scale"            # or try "auto"

# Optional: subsample for exploration speed (set None to use full train)
MAX_TRAIN_ROWS = 30000     # e.g., 20000-50000; set to None for full train


def main():

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    X_train, y_train = split_xy(train_df)
    X_test, y_test = split_xy(test_df)

    # Optional subsample (exploration)
    if MAX_TRAIN_ROWS is not None and len(X_train) > MAX_TRAIN_ROWS:
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(len(X_train), size=MAX_TRAIN_ROWS, replace=False)
        X_train = X_train.iloc[idx]
        y_train = y_train[idx]
        print(f"Subsampled train to {MAX_TRAIN_ROWS} rows for faster PCA+RBF SVM exploration.")

    num_cols, cat_cols = get_feature_types(X_train)
    print("Numeric cols:", len(num_cols), "Categorical cols:", len(cat_cols))

    # Preprocess: impute + onehot + scale
    pre = make_preprocessor(num_cols, cat_cols, scale_numeric=True)

    # Full pipeline: preprocess -> PCA -> RBF SVM (probabilities needed for thresholding)
    pipe = Pipeline([
        ("pre", pre),
        ("pca", PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)),
        ("svm", SVC(
            kernel="rbf",
            C=C,
            gamma=GAMMA,
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE
        ))
    ])

    print("Fitting PCA + RBF SVM...")
    pipe.fit(X_train, y_train)

    print("Evaluating...")
    prob_test = pipe.predict_proba(X_test)[:, 1]

    pr_auc = float(average_precision_score(y_test, prob_test))
    roc_auc = float(roc_auc_score(y_test, prob_test))
    brier = float(brier_score_loss(y_test, prob_test))

    best_t, costs_test, _ = find_best_threshold(y_test, prob_test, C_FP, C_FN)
    best_cost = expected_cost(y_test, prob_test, best_t, C_FP, C_FN)

    print(f"PR-AUC:  {pr_auc:.6f}")
    print(f"ROC-AUC: {roc_auc:.6f}")
    print(f"Brier:   {brier:.6f}")
    print(f"Best threshold (on TEST, exploratory): {best_t:.3f}  Cost: {best_cost:.1f}")

    # PR curve plot
    precision, recall, _ = precision_recall_curve(y_test, prob_test)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PCA({N_COMPONENTS}) + RBF SVM PR Curve (AP={pr_auc:.4f})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("explore_pr_curve_pca_rbf_svm.png", dpi=160)
    plt.close()

    # Save results
    out = {
        "train_rows_used": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "pca_components": int(N_COMPONENTS),
        "svm": {"kernel": "rbf", "C": C, "gamma": GAMMA},
        "metrics": {"pr_auc": pr_auc, "roc_auc": roc_auc, "brier": brier},
        "exploratory_best_threshold_on_test": best_t,
        "exploratory_best_cost_on_test": best_cost,
    }
    with open("explore_pca_rbf_svm_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print("Saved:")
    print(" - explore_pr_curve_pca_rbf_svm.png")
    print(" - explore_pca_rbf_svm_results.json")


if __name__ == "__main__":
    main()