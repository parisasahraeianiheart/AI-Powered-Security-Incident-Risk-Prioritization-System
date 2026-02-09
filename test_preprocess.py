#!/usr/bin/env python
"""Test fixed preprocessing pipeline"""

from preprocessing import get_feature_types, make_preprocessor
import pandas as pd

train = pd.read_csv('UNSW_NB15_training-set.csv')

# Simulate what split_xy does
X = train.drop(columns=['label'], errors='ignore')
drop_cols = [c for c in ['id', 'attack_cat'] if c in X.columns]
if drop_cols:
    X = X.drop(columns=drop_cols)

print(f'✓ X shape after dropping label, id, attack_cat: {X.shape}')

# Now test get_feature_types
num_cols, cat_cols = get_feature_types(X)
print(f'✓ Numeric features: {len(num_cols)}')
print(f'✓ Categorical features: {len(cat_cols)}')
print(f'✓ Categorical columns: {cat_cols}')

# Test make_preprocessor
pre = make_preprocessor(num_cols, cat_cols, scale_numeric=True)
print(f'✓ Preprocessor created successfully')

# Test fit
pre.fit(X.iloc[:10], train['label'].iloc[:10])
print(f'✓ Preprocessor fit successful')

# Test transform
X_transformed = pre.transform(X.iloc[:10])
print(f'✓ Transform successful, shape: {X_transformed.shape}')

print("\n✓✓✓ All preprocessing tests passed! ✓✓✓")
