from features.multi_tf import build_multi_tf_features
df = build_multi_tf_features('BTCUSDT')
cols_6h    = [c for c in df.columns if c.startswith('6h_')]
cols_cross = [c for c in df.columns if c.startswith('cross_')]
cols_1h    = [c for c in df.columns if c.startswith('1h_')]
cols_1d    = [c for c in df.columns if c.startswith('1d_')]
print(f"Total rows:        {len(df):,}")
print(f"Total features:    {df.shape[1]}")
print(f"5m features:       {df.shape[1] - len(cols_6h) - len(cols_cross) - len(cols_1h) - len(cols_1d) - 2}")
print(f"1h features:       {len(cols_1h)}")
print(f"6h features:       {len(cols_6h)}")
print(f"1d features:       {len(cols_1d)}")
print(f"Cross-tf features: {len(cols_cross)}")
print(f"\nSample 6h cols:    {cols_6h[:4]}")
print(f"Sample cross cols: {cols_cross[:4]}")
print("\nFeature build OK!")
