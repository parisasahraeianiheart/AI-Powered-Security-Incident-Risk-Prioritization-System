"""
Shared preprocessing utilities for UNSW-NB15 ML pipeline.
Consolidates feature type detection and ColumnTransformer building.
"""

from typing import List, Tuple
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Non-predictive columns to exclude from feature type detection (label-like / metadata)
EXCLUDE_FROM_FEATURES = {"attack_cat"}


def get_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify numeric and categorical columns, excluding label-like fields.
    
    Args:
        X: Feature DataFrame
        
    Returns:
        Tuple of (numeric_cols, categorical_cols) lists
    """
    exclude = EXCLUDE_FROM_FEATURES & set(X.columns)
    num_cols = [c for c in X.columns if c not in exclude and is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in exclude and c not in num_cols]
    return num_cols, cat_cols


def make_preprocessor(
    num_cols: List[str], 
    cat_cols: List[str], 
    scale_numeric: bool = True
) -> ColumnTransformer:
    """
    Build sklearn ColumnTransformer for numeric + categorical features.
    
    Numeric pipeline: median imputation + optional scaling
    Categorical pipeline: most_frequent imputation + one-hot encoding
    
    Args:
        num_cols: List of numeric column names
        cat_cols: List of categorical column names
        scale_numeric: If True, apply StandardScaler to numeric features
        
    Returns:
        ColumnTransformer configured for mixed data types
    """
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric = Pipeline(steps=num_steps)

    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
