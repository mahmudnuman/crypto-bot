"""
overnight.py - Full overnight orchestrator.
Waits for any running training to finish, then trains all remaining coins.
Run once; handles everything automatically.

Usage: python overnight.py
"""
import sys, os, time, json, subprocess
from pathlib import Path
from datetime import datetime

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from config import POPULAR_COINS, QUOTE_ASSET, MODEL_DIR, CACHE_DIR
from data.store import load
import progress as prog

LOG_FILE = CACHE_DIR / "overnight_train_log.txt"
logger.add(LOG_FILE, rotation="100 MB", level="DEBUG",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

SEP = "=" * 65

def already_trained(symbol: str) -> bool:
    """Check if a saved model already exists for this symbol."""
    model_path = MODEL_DIR / f"{symbol}_ensemble.pkl"
    return model_path.exists()

def train_one(symbol: str) -> dict:
    """Train a single symbol by calling trainer.py in a subprocess."""
    logger.info(f"Training {symbol}...")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "models/trainer.py", "--symbol", symbol],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(Path(__file__).parent),
    )
    elapsed = time.time() - t0
    ok = result.returncode == 0 or "Result:" in result.stdout

    # Extract accuracy from output
    acc = 0.0
    for line in result.stdout.splitlines():
        if "Result:" in line and "accuracy=" in line:
            try:
                acc = float(line.split("accuracy=")[-1].strip())
            except:
                pass
        if "best_accuracy" in line or "test_acc" in line:
            pass

    status = "ok" if ok else "error"
    return {
        "symbol":  symbol,
        "status":  status,
        "elapsed": elapsed,
        "acc":     acc,
        "stdout":  result.stdout[-2000:],
        "stderr":  result.stderr[-500:] if result.stderr else "",
    }

def verify_all_data() -> list:
    """Return list of symbols that have insufficient data."""
    missing = []
    symbols = [f"{c}{QUOTE_ASSET}" for c in POPULAR_COINS]
    for sym in symbols:
        try:
            n = len(load(sym, "5m"))
            if n < 50000:
                missing.append(sym)
        except:
            missing.append(sym)
    return missing

def fix_missing_data(missing: list):
    """Re-download any coins with insufficient data."""
    if not missing:
        return
    print(f"\n[DATA] Re-downloading {len(missing)} coin(s) with missing data...")
    from data.downloader import download_symbol
    for sym in missing:
        try:
            print(f"  Downloading {sym}...")
            download_symbol(sym, ["5m", "1h", "6h", "1d"])
            n = len(load(sym, "5m"))
            print(f"  [OK] {sym}: {n:,} rows")
        except Exception as e:
            print(f"  [FAIL] {sym}: {e}")
            logger.error(f"Data fix failed for {sym}: {e}")

def main():
    symbols = [f"{c}{QUOTE_ASSET}" for c in POPULAR_COINS]

    print(f"\n{SEP}")
    print("  CryptoPredictBot -- Overnight Training Orchestrator")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Coins: {len(symbols)}  |  Features: 5m + 1h + 6h + 1d + cross-tf")
    print(f"  Log: {LOG_FILE}")
    print(f"{SEP}\n")

    # Step 1: Verify all data is present
    print("[STEP 1] Verifying all data...")
    missing_data = verify_all_data()
    if missing_data:
        print(f"  Missing data for: {missing_data}")
        fix_missing_data(missing_data)
    else:
        print(f"  All 25 coins have data. OK.\n")

    # Step 2: Determine what needs training
    to_train = []
    skip     = []
    for sym in symbols:
        if already_trained(sym):
            skip.append(sym)
        else:
            to_train.append(sym)

    if skip:
        print(f"[STEP 2] Already trained ({len(skip)}): {', '.join(s.replace('USDT','') for s in skip)}")
    print(f"[STEP 2] Coins to train: {len(to_train)}\n")

    if not to_train:
        print("All coins already trained! Run predictions with the dashboard.")
        return

    # Step 3: Train all remaining coins
    print(f"[STEP 3] Starting training ({len(to_train)} coins)...\n")
    prog.start_trainer(to_train)

    results   = []
    failed    = []
    retries   = {}

    for idx, symbol in enumerate(to_train, 1):
        print(f"[{idx:02d}/{len(to_train)}] Training {symbol}...")
        logger.info(f"[{idx}/{len(to_train)}] Training {symbol}")

        prog.update("trainer",
            current_symbol=symbol,
            completed_symbols=idx - 1,
            total_symbols=len(to_train),
        )

        result = train_one(symbol)
        results.append(result)

        # Log last few lines of output
        for line in result["stdout"].splitlines()[-5:]:
            if any(k in line for k in ["Result:", "Fold", "accuracy", "ERROR", "skipped"]):
                print(f"    {line.strip()}")

        if result["status"] == "ok":
            print(f"  [OK] {symbol} -- acc={result['acc']:.3f}  time={result['elapsed']/60:.1f}min")
            logger.success(f"{symbol} trained: acc={result['acc']:.3f}")

            state = prog.read()
            done  = state["trainer"]["symbols_done"]
            done.append({"symbol": symbol, "status": "ok", "best_acc": result["acc"]})
            prog.update("trainer", symbols_done=done, completed_symbols=idx)
        else:
            print(f"  [FAIL] {symbol} -- will retry once")
            logger.error(f"{symbol} failed: {result['stderr'][:200]}")
            failed.append(symbol)

    # Step 4: Retry failed symbols once
    if failed:
        print(f"\n[STEP 4] Retrying {len(failed)} failed coin(s)...\n")
        time.sleep(30)
        retry_results = []
        for symbol in failed[:]:
            print(f"  Retrying {symbol}...")
            result = train_one(symbol)
            retry_results.append(result)
            if result["status"] == "ok":
                failed.remove(symbol)
                print(f"  [OK] Retry succeeded: {symbol}  acc={result['acc']:.3f}")
            else:
                print(f"  [FAIL] Retry failed: {symbol}")
        results.extend(retry_results)

    prog.finish_trainer()

    # Step 5: Print final report
    ok_results   = [r for r in results if r["status"] == "ok"]
    fail_results = [r for r in results if r["status"] != "ok"]

    print(f"\n{SEP}")
    print(f"  TRAINING COMPLETE -- {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{SEP}")
    print(f"\n  [OK] {len(ok_results) + len(skip)}/25 coins trained\n")

    if ok_results:
        print(f"  {'Coin':<10} {'Accuracy':>10} {'Time':>8}")
        print("  " + "-"*30)
        for r in sorted(ok_results, key=lambda x: x["acc"], reverse=True):
            sym = r["symbol"].replace("USDT","")
            flag = " <-- HIGH" if r["acc"] >= 0.70 else ""
            print(f"  {sym:<10} {r['acc']:>10.3f} {r['elapsed']/60:>6.1f}m{flag}")

    if skip:
        print(f"\n  Pre-existing models: {', '.join(s.replace('USDT','') for s in skip)}")

    if fail_results:
        print(f"\n  [FAIL] {len(fail_results)} coin(s) could not be trained:")
        for r in fail_results:
            print(f"    - {r['symbol']}: {r['stderr'][:100]}")

    total_min = sum(r["elapsed"] for r in ok_results) / 60
    avg_acc   = sum(r["acc"] for r in ok_results) / max(len(ok_results), 1)
    print(f"\n  Avg accuracy (trained now): {avg_acc:.3f}")
    print(f"  Total training time: {total_min:.0f} min")
    print(f"\n{SEP}")
    print("  NEXT: python -m streamlit run dashboard/app.py")
    print(f"{SEP}\n")

    logger.info(f"Overnight training complete. {len(ok_results)} OK, {len(fail_results)} failed.")
    sys.exit(0 if len(fail_results) == 0 else 1)


if __name__ == "__main__":
    main()
