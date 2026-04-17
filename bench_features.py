import time, sys
from features.multi_tf import build_multi_tf_features
t = time.time()
print("Building BTC feature matrix...")
df = build_multi_tf_features('BTCUSDT')
elapsed = time.time() - t
print(f"Built {len(df):,} rows x {df.shape[1]} cols in {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"6h features:   {len([c for c in df.columns if c.startswith('6h_')])}")
print(f"Cross features:{len([c for c in df.columns if c.startswith('cross_')])}")
