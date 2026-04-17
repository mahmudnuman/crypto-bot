"""
status_check.py - Full system status: data + models
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import POPULAR_COINS, QUOTE_ASSET, MODEL_DIR, CACHE_DIR
from data.store import load

symbols  = [f"{c}{QUOTE_ASSET}" for c in POPULAR_COINS]
feat_dir = CACHE_DIR / "features"
feat_dir.mkdir(exist_ok=True)

print(f"\n{'='*65}")
print(f"  CryptoPredictBot — System Status")
print(f"{'='*65}\n")

ok_data, ok_models, ok_6h = 0, 0, 0
missing_data, missing_models, missing_6h = [], [], []

print(f"  {'Coin':<10} {'5m':>8} {'1h':>7} {'6h':>7} {'1d':>6}  {'Model':>8}  {'Cache':>6}")
print(f"  {'-'*60}")

for sym in symbols:
    coin = sym.replace("USDT","")
    try:
        n5  = len(load(sym, "5m"))
        n1h = len(load(sym, "1h"))
        n6h = len(load(sym, "6h"))
        n1d = len(load(sym, "1d"))
    except Exception as e:
        n5, n1h, n6h, n1d = 0, 0, 0, 0

    has_data  = n5 > 50000
    has_6h    = n6h > 100
    has_model = (MODEL_DIR / f"{sym}_ensemble.pkl").exists()
    has_cache = (feat_dir / f"{sym}_features.parquet").exists()

    data_s  = f"{n5//1000}k" if n5 > 0 else "MISS"
    h1_s    = f"{n1h//1000}k" if n1h > 0 else "MISS"
    h6_s    = f"{n6h}" if n6h > 0 else "MISS"
    d1_s    = f"{n1d}" if n1d > 0 else "MISS"
    model_s = "OK" if has_model else "---"
    cache_s = "OK" if has_cache else "---"

    flag = ""
    if not has_data: flag += " [!DATA]"
    if not has_6h:   flag += " [!6H]"

    print(f"  {coin:<10} {data_s:>8} {h1_s:>7} {h6_s:>7} {d1_s:>6}  {model_s:>8}  {cache_s:>6}{flag}")

    if has_data:  ok_data  += 1
    else: missing_data.append(sym)

    if has_6h:    ok_6h    += 1
    else: missing_6h.append(sym)

    if has_model: ok_models += 1
    else: missing_models.append(sym)

print(f"\n  {'='*60}")
print(f"  Data:   {ok_data}/25 coins have 5m data")
print(f"  6h:     {ok_6h}/25 coins have 6h data")
print(f"  Models: {ok_models}/25 coins trained")
print(f"\n  Missing data:   {[s.replace('USDT','') for s in missing_data] or 'None'}")
print(f"  Missing 6h:     {[s.replace('USDT','') for s in missing_6h] or 'None'}")
print(f"  Missing models: {[s.replace('USDT','') for s in missing_models] or 'None'}")
print(f"{'='*65}\n")
