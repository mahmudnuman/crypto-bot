"""
download_6h.py - Download 6h candles for all 25 coins.
Runs fast (~2-3 min total via REST API).
"""
import sys, time
from pathlib import Path
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
from config import POPULAR_COINS, QUOTE_ASSET
from data.downloader import fetch_and_store_rest
from data.store import load, last_timestamp

symbols = [f"{c}{QUOTE_ASSET}" for c in POPULAR_COINS]

print(f"\n{'='*60}")
print(f"  Downloading 6h candles for {len(symbols)} coins")
print(f"{'='*60}\n")

ok, fail = 0, 0
for i, sym in enumerate(symbols, 1):
    try:
        fetch_and_store_rest(sym, "6h")
        n = len(load(sym, "6h"))
        print(f"  [{i:02d}/25] {sym:<14} 6h={n:>7,} rows  OK")
        ok += 1
    except Exception as e:
        print(f"  [{i:02d}/25] {sym:<14} ERROR: {e}")
        fail += 1
    time.sleep(0.5)

print(f"\n{'='*60}")
print(f"  Done: {ok}/25 coins  |  {fail} failed")
print(f"  NEXT: python models/trainer.py --all")
print(f"{'='*60}\n")
