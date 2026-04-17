"""
redownload_missing.py - Re-download only the coins with missing/partial data.
"""
import sys
import os
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from data.downloader import download_symbol, fetch_and_store_rest, fetch_and_store_5m_bulk
from data.store import load

# Coins that need downloading (0 or partial rows)
MISSING = [
    "NEARUSDT", "ARBUSDT", "OPUSDT", "APTUSDT", "SUIUSDT",
    "INJUSDT", "FILUSDT", "XLMUSDT", "ALGOUSDT", "TIAUSDT",
]

# Newer coins that may not have 5m bulk ZIPs at all — REST only
# These launched after 2021, so bulk ZIPs may 404
REST_ONLY_5M = {
    "ARBUSDT":  "2022-09-01",   # launched Sep 2022
    "OPUSDT":   "2022-06-01",   # launched Jun 2022
    "APTUSDT":  "2022-10-01",   # launched Oct 2022
    "SUIUSDT":  "2023-05-01",   # launched May 2023
    "INJUSDT":  "2021-10-01",
    "TIAUSDT":  "2023-10-01",   # launched Oct 2023
}

print(f"\n{'='*60}")
print(f"  Re-downloading {len(MISSING)} coins with missing data")
print(f"{'='*60}\n")

for idx, symbol in enumerate(MISSING, 1):
    print(f"\n[{idx:02d}/{len(MISSING)}] {symbol}")
    t0 = time.time()
    try:
        # Try bulk ZIPs first (will gracefully 404 for new coins)
        fetch_and_store_5m_bulk(symbol)
        # Always fill via REST (gets everything bulk missed + recent months)
        fetch_and_store_rest(symbol, "5m")
        fetch_and_store_rest(symbol, "1h")
        fetch_and_store_rest(symbol, "1d")

        # Verify
        r5  = len(load(symbol, "5m"))
        r1h = len(load(symbol, "1h"))
        r1d = len(load(symbol, "1d"))
        elapsed = time.time() - t0
        print(f"  [OK] 5m={r5:,}  1h={r1h:,}  1d={r1d:,}  ({elapsed:.0f}s)")
    except Exception as e:
        print(f"  [ERROR] {e}")
        logger.error(f"{symbol} failed: {e}")
    time.sleep(2)

print(f"\n{'='*60}")
print("  Done! Running final verification...")
print(f"{'='*60}\n")

# Final check
from config import POPULAR_COINS, QUOTE_ASSET
ok = 0
for coin in POPULAR_COINS:
    sym = f"{coin}{QUOTE_ASSET}"
    try:
        r5  = len(load(sym, "5m"))
        r1h = len(load(sym, "1h"))
        status = "OK" if r5 >= 50000 and r1h >= 5000 else "LOW"
        if status == "OK": ok += 1
        print(f"  {status}  {coin:<10} 5m={r5:>8,}  1h={r1h:>6,}")
    except:
        print(f"  MISS {coin}")

print(f"\n  TOTAL: {ok}/25 coins ready for training")
