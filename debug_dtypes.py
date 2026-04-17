"""Debug feature dtype issues"""
from features.multi_tf import build_multi_tf_features
import pandas as pd
df = build_multi_tf_features('BTCUSDT')
bad_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"Object dtype columns ({len(bad_cols)}): {bad_cols[:20]}")
print(f"\nTotal cols: {df.shape[1]}")
print(f"NaN cols: {df.isnull().all().sum()}")
print(f"\nSample dtypes:")
print(df.dtypes.value_counts())
