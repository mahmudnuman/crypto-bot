"""
download_all_coins.py - Overnight batch downloader for all 25 coins.

Runs sequentially, handles errors per-coin, auto-retries failures,
and prints a full verification report at the end.

Usage:
    python download_all_coins.py
"""
import time
import sys
import os
from datetime import datetime
from pathlib import Path

# Force UTF-8 on Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from config import POPULAR_COINS, QUOTE_ASSET, CACHE_DIR
from data.downloader import download_symbol
from data.store import load, last_timestamp
import progress as prog

# Setup file logging
LOG_FILE = CACHE_DIR / "download_log.txt"
logger.add(LOG_FILE, rotation="50 MB", level="INFO",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

TIMEFRAMES = ["5m", "1h", "1d"]

# Minimum expected rows per timeframe (4+ years of history)
MIN_ROWS = {
    "5m":  300_000,
    "1h":  30_000,
    "1d":  1_200,
}

SEP = "=" * 65


def verify_symbol(symbol: str) -> dict:
    result = {"symbol": symbol, "ok": True, "issues": [], "rows": {}}
    for tf in TIMEFRAMES:
        try:
            df = load(symbol, tf)
            n  = len(df)
            result["rows"][tf] = n
            if n < MIN_ROWS[tf]:
                result["ok"] = False
                result["issues"].append(f"{tf}: only {n:,} rows (need >={MIN_ROWS[tf]:,})")
        except Exception as e:
            result["ok"] = False
            result["issues"].append(f"{tf}: error -> {e}")
    return result


def print_report(results: list) -> None:
    ok     = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    print(f"\n{SEP}")
    print(f"  DOWNLOAD COMPLETE -- {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(SEP)
    print(f"\n  [OK] {len(ok)}/{len(results)} coins fully downloaded\n")

    if ok:
        print(f"  {'Coin':<10} {'5m rows':>12} {'1h rows':>10} {'1d rows':>8}")
        print("  " + "-"*44)
        for r in ok:
            rows = r["rows"]
            sym  = r["symbol"].replace("USDT", "")
            print(f"  {sym:<10} {rows.get('5m',0):>12,} {rows.get('1h',0):>10,} {rows.get('1d',0):>8,}")

    if failed:
        print(f"\n  [FAIL] {len(failed)} coin(s) had issues:")
        for r in failed:
            print(f"    - {r['symbol']}: {'; '.join(r['issues'])}")

    print(f"\n{SEP}")
    print("  NEXT STEP: python models/trainer.py --all")
    print(f"{SEP}\n")


def main():
    symbols = [f"{c}{QUOTE_ASSET}" for c in POPULAR_COINS]
    total   = len(symbols)

    print(f"\n{SEP}")
    print(f"  CryptoPredictBot -- Overnight Data Download")
    print(f"  Coins: {total}  |  Timeframes: {TIMEFRAMES}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Log: {LOG_FILE}")
    print(f"{SEP}\n")

    logger.info(f"Starting overnight download for {total} symbols")
    prog.start_downloader(symbols, TIMEFRAMES)

    failed_symbols = []

    for idx, symbol in enumerate(symbols, 1):
        print(f"\n[{idx:02d}/{total}] -- {symbol} --")
        logger.info(f"[{idx}/{total}] Starting {symbol}")
        t0 = time.time()

        try:
            prog.update("downloader",
                current_symbol=symbol,
                current_task=f"{symbol} ({idx}/{total})",
                symbols_pending=symbols[idx:],
            )
            download_symbol(symbol, TIMEFRAMES)

            # Update progress state
            state = prog.read()
            done  = state["downloader"]["symbols_done"]
            done.append(symbol)
            prog.update("downloader",
                symbols_done=done,
                completed_tasks=state["downloader"]["completed_tasks"] + len(TIMEFRAMES),
            )
            elapsed = time.time() - t0
            print(f"  [OK] Done in {elapsed:.0f}s")
            logger.success(f"[{symbol}] completed in {elapsed:.0f}s")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [ERROR] {symbol} failed: {e}")
            logger.error(f"[{symbol}] failed after {elapsed:.0f}s: {e}")
            failed_symbols.append(symbol)

            state = prog.read()
            errs  = state["downloader"]["errors"]
            errs.append(f"{symbol}: {e}")
            prog.update("downloader", errors=errs)

        # Polite pause between coins
        time.sleep(2)

    prog.finish_downloader()

    # Retry any failures once
    if failed_symbols:
        print(f"\n[RETRY] Retrying {len(failed_symbols)} failed coin(s)...\n")
        logger.info(f"Retrying failed coins: {failed_symbols}")
        time.sleep(15)
        for symbol in failed_symbols[:]:
            print(f"  Retrying {symbol}...")
            try:
                download_symbol(symbol, TIMEFRAMES)
                failed_symbols.remove(symbol)
                print(f"  [OK] {symbol} retry succeeded")
                logger.success(f"Retry succeeded: {symbol}")
            except Exception as e:
                print(f"  [FAIL] {symbol} retry failed: {e}")
                logger.error(f"Retry failed {symbol}: {e}")

    # Verify all data
    print(f"\n[VERIFY] Checking downloaded data...\n")
    results = []
    for symbol in symbols:
        r = verify_symbol(symbol)
        results.append(r)
        status = "[OK]  " if r["ok"] else "[FAIL]"
        rows5m = r["rows"].get("5m", 0)
        rows1h = r["rows"].get("1h", 0)
        rows1d = r["rows"].get("1d", 0)
        issues = "  ".join(r["issues"]) if r["issues"] else "OK"
        print(f"  {status} {symbol:<14}  5m={rows5m:>8,}  1h={rows1h:>6,}  1d={rows1d:>4,}  {issues}")

    print_report(results)
    logger.info("Overnight download and verification complete.")

    ok_count = sum(1 for r in results if r["ok"])
    sys.exit(0 if ok_count >= 20 else 1)


if __name__ == "__main__":
    main()
