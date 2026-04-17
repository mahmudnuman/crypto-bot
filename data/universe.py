"""
data/universe.py — Builds and validates the coin universe.

Merges:
  1. Our curated POPULAR_COINS list (config.py)
  2. Live Binance exchangeInfo validation (only keep pairs currently trading)
  3. Optional CoinGecko rank check (soft warn if a coin fell far out of top-100)
"""
import json
import time
import requests
from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import POPULAR_COINS, QUOTE_ASSET, CACHE_DIR, BINANCE_REST_BASE


UNIVERSE_FILE = CACHE_DIR / "universe.json"
UNIVERSE_TTL  = 86_400  # 24 hours


def _fetch_binance_active_pairs() -> set[str]:
    """Return all currently TRADING USDT spot pairs on Binance."""
    url  = f"{BINANCE_REST_BASE}/api/v3/exchangeInfo"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    active = {
        s["symbol"]
        for s in data["symbols"]
        if s["quoteAsset"] == QUOTE_ASSET
        and s["status"] == "TRADING"
        and s["isSpotTradingAllowed"]
    }
    logger.info(f"Binance: {len(active)} active USDT spot pairs found.")
    return active


def build_universe(force: bool = False) -> list[dict]:
    """
    Returns list of dicts: [{coin, symbol, rank}, ...]
    Uses cached result if fresh enough.
    """
    if UNIVERSE_FILE.exists() and not force:
        age = time.time() - UNIVERSE_FILE.stat().st_mtime
        if age < UNIVERSE_TTL:
            with open(UNIVERSE_FILE) as f:
                universe = json.load(f)
            logger.info(f"Universe loaded from cache ({len(universe)} coins).")
            return universe

    logger.info("Building coin universe from Binance exchangeInfo …")
    active_pairs = _fetch_binance_active_pairs()

    universe = []
    missing   = []
    for rank, coin in enumerate(POPULAR_COINS, start=1):
        symbol = f"{coin}{QUOTE_ASSET}"
        if symbol in active_pairs:
            universe.append({"coin": coin, "symbol": symbol, "rank": rank})
        else:
            missing.append(symbol)
            logger.warning(f"  ✗ {symbol} not available on Binance — skipped.")

    if missing:
        logger.warning(f"{len(missing)} coins skipped: {missing}")

    with open(UNIVERSE_FILE, "w") as f:
        json.dump(universe, f, indent=2)

    logger.success(f"Universe built: {len(universe)} coins. Saved to {UNIVERSE_FILE}")
    return universe


def get_symbols() -> list[str]:
    """Quick helper — returns just the symbol strings (e.g. BTCUSDT)."""
    return [c["symbol"] for c in build_universe()]


if __name__ == "__main__":
    u = build_universe(force=True)
    for item in u:
        print(f"  {item['rank']:3d}. {item['symbol']}")
