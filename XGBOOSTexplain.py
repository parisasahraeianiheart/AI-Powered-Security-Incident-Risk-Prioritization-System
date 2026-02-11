# XGBOOSTexplain.py
# Robust SHAP explainability for XGBoost + preprocessing pipelines on macOS.
# Avoids SHAP TreeExplainer/XGBoost base_score parsing issue by using shap.Explainer with a predict_proba function.
# Uses scale_numeric=False to match model.py's XGBoost pipeline (unscaled trees).

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from preprocessing import get_feature_types, make_preprocessor
from utils import TRAIN_CSV, TEST_CSV, RANDOM_STATE, split_xy

# For SHAP, start small (fast + stable). Increase later if you want.
BACKGROUND_ROWS = 200     # background distribution for SHAP masker
EXPLAIN_ROWS = 300        # how many test rows to explain (increase to 1000 if desired)


def main():
    # -------------------------
    # Load data
    # -------------------------

    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    X_train, y_train = split_xy(train)
    X_test, y_test = split_xy(test)

    # Feature typing using shared utility
    num_cols, cat_cols = get_feature_types(X_train)

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Numeric cols: {len(num_cols)}, Categorical cols: {len(cat_cols)}")

    # -------------------------
    # Preprocess (unscaled to match model.py XGBoost pipeline)
    # -------------------------
    pre = make_preprocessor(num_cols, cat_cols, scale_numeric=False)

    # Class imbalance weight (same idea as your main modeling)
    neg = int(np.sum(y_train == 0))
    pos = int(np.sum(y_train == 1))
    spw = float(neg / max(pos, 1))

    clf = XGBClassifier(
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
        base_score=0.5,   # keep explicit (doesn't fix SHAP TreeExplainer bug, but fine here)
    )

    model = Pipeline(steps=[("pre", pre), ("clf", clf)])

    print("Fitting XGBoost pipeline...")
    model.fit(X_train, y_train)
    print("Done fitting.")

    # -------------------------
    # Transform data for SHAP
    # -------------------------
    print("Transforming data for SHAP...")
    X_train_trans = model.named_steps["pre"].transform(X_train)
    X_test_trans = model.named_steps["pre"].transform(X_test)

    # Feature names after one-hot
    feature_names = model.named_steps["pre"].get_feature_names_out()

    # Convert to dense for SHAP plotting stability (use small samples to keep memory low)
    if hasattr(X_train_trans, "toarray"):
        X_train_trans = X_train_trans.toarray()
    if hasattr(X_test_trans, "toarray"):
        X_test_trans = X_test_trans.toarray()

    rng = np.random.RandomState(RANDOM_STATE)

    bg_n = min(BACKGROUND_ROWS, X_train_trans.shape[0])
    ex_n = min(EXPLAIN_ROWS, X_test_trans.shape[0])

    bg_idx = rng.choice(X_train_trans.shape[0], size=bg_n, replace=False)
    ex_idx = rng.choice(X_test_trans.shape[0], size=ex_n, replace=False)

    X_bg = X_train_trans[bg_idx]
    X_explain = X_test_trans[ex_idx]

    print(f"SHAP background rows: {X_bg.shape[0]}")
    print(f"SHAP explain rows:    {X_explain.shape[0]}")

    # -------------------------
    # SHAP (model-agnostic interface using predict_proba)
    # -------------------------
    # We explain the classifier directly in transformed feature space.
    xgb = model.named_steps["clf"]

    masker = shap.maskers.Independent(X_bg)

    # Explain probability of positive class
    def predict_proba_pos(X):
        return xgb.predict_proba(X)[:, 1]

    print("Computing SHAP values (robust explainer)...")
    explainer = shap.Explainer(predict_proba_pos, masker, feature_names=feature_names)
    shap_values = explainer(X_explain)

    # -------------------------
    # Global plots
    # -------------------------
    print("Saving SHAP bar plot...")
    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig("shap_bar_xgb.png", dpi=160)
    plt.close()

    print("Saving SHAP beeswarm summary plot...")
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary_xgb.png", dpi=160)
    plt.close()

    # -------------------------
    # Local plot (single example)
    # -------------------------
    print("Saving SHAP waterfall for one example...")
    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig("shap_waterfall_example0.png", dpi=160)
    plt.close()

    print("\nSaved:")
    print(" - shap_bar_xgb.png")
    print(" - shap_summary_xgb.png")
    print(" - shap_waterfall_example0.png")


if __name__ == "__main__":
    main()