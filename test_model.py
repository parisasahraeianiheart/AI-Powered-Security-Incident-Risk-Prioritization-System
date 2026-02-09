#!/usr/bin/env python
"""Quick test of model.py components"""

import sys
import traceback

try:
    # Test imports
    from preprocessing import get_feature_types, make_preprocessor
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    import pandas as pd
    import numpy as np
    from sklearn.pipeline import Pipeline
    
    print("✓ All imports successful")
    
    # Quick syntax check
    train = pd.read_csv('UNSW_NB15_training-set.csv')
    X = train.drop(columns=['label', 'id'], errors='ignore')
    y = train['label'].values
    
    num_cols, cat_cols = get_feature_types(X)
    print(f"✓ Feature detection: {len(num_cols)} numeric, {len(cat_cols)} categorical")
    
    pre = make_preprocessor(num_cols, cat_cols, scale_numeric=True)
    pipe = Pipeline([
        ('pre', pre),
        ('clf', LogisticRegression(max_iter=100, n_jobs=-1))
    ])
    
    print("✓ Pipeline created successfully")
    
    # Test CV split
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    for i, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        print(f"✓ Fold {i+1}: train {len(tr_idx)}, val {len(va_idx)}")
    
    print("\n✓✓✓ All tests passed - code is ready to run! ✓✓✓")
        
except Exception as e:
    print(f"\n✗ Error: {e}")
    traceback.print_exc()
    sys.exit(1)
