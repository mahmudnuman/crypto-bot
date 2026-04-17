"""
run_all.py - Complete pipeline with FOLD-LEVEL CHECKPOINT/RESUME.

On restart: skips fully completed coins, resumes mid-coin from the
last completed fold — no work is repeated.

Usage: python run_all.py
"""
import sys, os, time, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime, timedelta

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from config import POPULAR_COINS, QUOTE_ASSET, MODEL_DIR, CACHE_DIR
from data.store import load

FEAT_DIR    = CACHE_DIR / "features"
CKPT_DIR    = CACHE_DIR / "checkpoints"   # one JSON per coin
LOG_FILE    = CACHE_DIR / "run_all_log.txt"
REPORT      = CACHE_DIR / "training_report.json"

for d in [FEAT_DIR, CKPT_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logger.add(LOG_FILE, rotation="50 MB", level="INFO",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", enqueue=True)

SEP = "=" * 65

# ── Checkpoint helpers ────────────────────────────────────────────────
def ckpt_path(symbol): return CKPT_DIR / f"{symbol}.json"

def ckpt_load(symbol):
    p = ckpt_path(symbol)
    if p.exists():
        try: return json.loads(p.read_text())
        except: pass
    return {"status": "pending", "folds": [], "best_acc": 0.0, "best_fold": -1}

def ckpt_save(symbol, data):
    ckpt_path(symbol).write_text(json.dumps(data, indent=2))

def ckpt_is_done(symbol):
    c = ckpt_load(symbol)
    return c.get("status") == "done"


# ── 1. Data verification ──────────────────────────────────────────────
def verify_and_fix_data(symbols):
    print("\n[STEP 1/3] Verifying data completeness...")
    problems = []
    for sym in symbols:
        try:
            n5  = len(load(sym, "5m"))
            n1h = len(load(sym, "1h"))
            n6h = len(load(sym, "6h"))
            n1d = len(load(sym, "1d"))
            if n5 < 50000 or n1h < 1000 or n6h < 100 or n1d < 100:
                problems.append((sym, n5, n1h, n6h, n1d))
        except Exception:
            problems.append((sym, 0, 0, 0, 0))

    if not problems:
        print("  All 25 coins fully verified.\n")
        return

    from data.downloader import fetch_and_store_rest, fetch_and_store_5m_bulk
    print(f"  Fixing {len(problems)} coins with missing data...")
    for sym, n5, n1h, n6h, n1d in problems:
        try:
            if n5 < 50000:
                fetch_and_store_5m_bulk(sym)
                fetch_and_store_rest(sym, "5m")
            if n1h < 1000: fetch_and_store_rest(sym, "1h")
            if n6h < 100:  fetch_and_store_rest(sym, "6h")
            if n1d < 100:  fetch_and_store_rest(sym, "1d")
        except Exception as e:
            print(f"    WARNING {sym}: {e}")
    print()


# ── 2. Feature pre-build ──────────────────────────────────────────────
def prebuild_features(symbols):
    print("[STEP 2/3] Pre-building feature matrices (cached)...")
    import pandas as pd
    from features.multi_tf import build_multi_tf_features

    results = {}
    for i, sym in enumerate(symbols, 1):
        cache_path = FEAT_DIR / f"{sym}_features.parquet"

        if cache_path.exists():
            try:
                age_h = (pd.Timestamp.now() - pd.Timestamp(
                    cache_path.stat().st_mtime, unit="s")).total_seconds() / 3600
                if age_h < 48:
                    df = pd.read_parquet(cache_path)
                    if df.shape[1] > 300:
                        done_tag = " ✓ DONE" if ckpt_is_done(sym) else ""
                        print(f"  [{i:02d}/25] {sym:<14} cache OK ({df.shape[0]:,}r x {df.shape[1]}c){done_tag}")
                        results[sym] = df.shape
                        continue
                    cache_path.unlink()
            except Exception:
                pass

        t0 = time.time()
        print(f"  [{i:02d}/25] {sym:<14} building...", end="", flush=True)
        try:
            from features.multi_tf import build_multi_tf_features
            df = build_multi_tf_features(sym)
            if df.empty or len(df) < 5000:
                print(f" WARN: only {len(df)} rows")
                results[sym] = None
                continue
            df.to_parquet(cache_path)
            print(f" {df.shape[0]:,}r x {df.shape[1]}c  ({time.time()-t0:.0f}s)")
            results[sym] = df.shape
        except Exception as e:
            print(f" ERROR: {e}")
            results[sym] = None

    ok = sum(1 for v in results.values() if v is not None)
    print(f"\n  Features ready: {ok}/{len(symbols)}\n")
    return results


# ── 3. Train all coins with checkpoint/resume ─────────────────────────
def train_all_inprocess(symbols):
    import pandas as pd
    import numpy as np
    import joblib
    from features.pipeline import get_feature_cols
    from models.ensemble import DualHeadEnsemble
    from models.validator import analyse_fold, build_report
    from config import (WFV_INITIAL_TRAIN_YEARS, WFV_GAP_DAYS,
                        WFV_TEST_DAYS, WFV_STEP_DAYS)

    # Count how many are already done
    n_done_before = sum(1 for s in symbols if ckpt_is_done(s))
    print(f"[STEP 3/3] Training {len(symbols)} coins  "
          f"({n_done_before} already done, resuming rest)...\n")

    all_results = []

    for coin_idx, symbol in enumerate(symbols, 1):
        ckpt = ckpt_load(symbol)

        # ── Already fully done? Skip ──────────────────────────────────
        if ckpt["status"] == "done":
            acc = ckpt.get("best_acc", 0)
            print(f"  [{coin_idx:02d}/{len(symbols)}] {symbol:<14} ✓ DONE (acc={acc:.3f})")
            all_results.append({
                "symbol": symbol, "status": "ok",
                "acc": acc, "elapsed": ckpt.get("elapsed", 0)
            })
            continue

        t0 = time.time()
        logger.info(f"[{coin_idx}/{len(symbols)}] Training {symbol}")

        # ── How many folds already done? ──────────────────────────────
        completed_folds = ckpt.get("folds", [])
        n_folds_done    = len(completed_folds)
        best_acc        = ckpt.get("best_acc", 0.0)
        resume_msg      = f" (resuming from fold {n_folds_done})" if n_folds_done > 0 else ""

        print(f"  [{coin_idx:02d}/{len(symbols)}] {symbol:<14} loading features{resume_msg}...")

        # ── Load features ─────────────────────────────────────────────
        cache_path = FEAT_DIR / f"{symbol}_features.parquet"
        try:
            df = pd.read_parquet(cache_path)
        except Exception:
            print(f"  [{coin_idx:02d}] CACHE MISS — rebuilding...")
            from features.multi_tf import build_multi_tf_features
            df = build_multi_tf_features(symbol)
            if not df.empty:
                df.to_parquet(cache_path)

        if df.empty or len(df) < 5000:
            print(f"      SKIP — insufficient data ({len(df)} rows)")
            all_results.append({"symbol": symbol, "status": "skip", "acc": 0.0, "elapsed": 0.0})
            continue

        feature_cols = get_feature_cols(df)
        df = df.sort_index()
        print(f"      {len(df):,} rows | {len(feature_cols)} features | {n_folds_done} folds already done")

        # ── Walk-forward setup ────────────────────────────────────────
        t_start       = df.index.min()
        t_end         = df.index.max()
        cur_train_end = t_start + timedelta(days=WFV_INITIAL_TRAIN_YEARS * 365)

        # Enumerate all fold windows
        all_windows = []
        scan = cur_train_end
        while True:
            ts = scan + timedelta(days=WFV_GAP_DAYS)
            te = ts   + timedelta(days=WFV_TEST_DAYS)
            if te > t_end: break
            all_windows.append((scan, ts, te))
            scan += timedelta(days=WFV_STEP_DAYS)

        total_folds  = len(all_windows)
        fold_results = list(completed_folds)   # restore previous fold results
        best_model   = None
        fold_errors  = 0

        # Re-load best model if we're resuming
        best_fold_idx = ckpt.get("best_fold", -1)
        if n_folds_done > 0 and best_fold_idx >= 0:
            try:
                ckpt_model_path = CKPT_DIR / f"{symbol}_fold{best_fold_idx}.pkl"
                if ckpt_model_path.exists():
                    best_model = joblib.load(ckpt_model_path)
                    print(f"      Loaded best model from fold {best_fold_idx}")
            except Exception as e:
                print(f"      Could not reload mid-run model: {e}")

        # ── Walk-Forward Loop (skip already completed folds) ──────────
        for fold_idx, (train_end, test_start, test_end) in enumerate(all_windows):

            # Skip folds already done in previous run
            if fold_idx < n_folds_done:
                continue

            df_train = df[df.index < train_end]
            df_test  = df[(df.index >= test_start) & (df.index < test_end)]

            if len(df_train) < 1000 or len(df_test) < 50:
                continue

            def _prep(df_split):
                clean   = df_split.dropna(subset=["target_dir", "target_price"])
                X       = clean[feature_cols].replace([np.inf, -np.inf], np.nan)
                y_dir   = clean["target_dir"].astype(int)
                y_price = clean["target_price"].astype(float)
                return X, y_dir, y_price

            X_train, y_dir_train, y_price_train = _prep(df_train)
            X_test,  y_dir_test,  y_price_test  = _prep(df_test)

            if len(X_train) < 500 or len(X_test) < 20:
                continue

            val_split   = int(len(X_train) * 0.85)
            X_tr        = X_train.iloc[:val_split]
            y_tr_dir    = y_dir_train.iloc[:val_split]
            y_tr_price  = y_price_train.iloc[:val_split]
            X_val       = X_train.iloc[val_split:]
            y_val_dir   = y_dir_train.iloc[val_split:]
            y_val_price = y_price_train.iloc[val_split:]

            model = DualHeadEnsemble(symbol)
            try:
                model.fit(X_tr, y_tr_dir, y_tr_price,
                          X_val_raw=X_val, y_val_dir=y_val_dir, y_val_price=y_val_price)

                tr_m   = model.evaluate(X_tr,    y_tr_dir,    y_tr_price)
                test_m = model.evaluate(X_test,  y_dir_test,  y_price_test)

                fold_data = {
                    "fold":      fold_idx,
                    "train_acc": round(tr_m["accuracy"], 4),
                    "test_acc":  round(test_m["accuracy"], 4),
                    "mape":      round(float(test_m["mape"]) if str(test_m["mape"]) != "nan" else 0, 4),
                    "n_train":   len(X_tr),
                    "n_test":    len(X_test),
                }
                fold_results.append(fold_data)

                logger.info(
                    f"  [{symbol}] Fold {fold_idx:02d}/{total_folds-1} "
                    f"train={tr_m['accuracy']:.3f} test={test_m['accuracy']:.3f} "
                    f"mape={test_m['mape']:.3f}"
                )
                print(f"      Fold {fold_idx:02d}/{total_folds-1} | "
                      f"train={tr_m['accuracy']:.3f} test={test_m['accuracy']:.3f}", flush=True)

                # Track best model
                if test_m["accuracy"] > best_acc:
                    best_acc = test_m["accuracy"]
                    best_model = model
                    # Save this fold's model as the best checkpoint
                    joblib.dump(model, CKPT_DIR / f"{symbol}_fold{fold_idx}.pkl")

                # ── SAVE CHECKPOINT after every fold ─────────────────
                ckpt_save(symbol, {
                    "status":     "in_progress",
                    "folds":      fold_results,
                    "best_acc":   round(best_acc, 4),
                    "best_fold":  fold_idx if test_m["accuracy"] >= best_acc else ckpt.get("best_fold", -1),
                    "elapsed":    round(time.time() - t0, 1),
                    "updated_at": datetime.now().isoformat(),
                })

            except Exception as e:
                fold_errors += 1
                logger.error(f"  [{symbol}] Fold {fold_idx} failed: {e}")
                print(f"      Fold {fold_idx} ERROR: {str(e)[:80]}")
                if fold_errors > 5:
                    logger.error(f"  [{symbol}] Too many errors, aborting.")
                    break

        # ── Save final model ──────────────────────────────────────────
        elapsed = time.time() - t0
        if best_model is not None and fold_results:
            best_model.save("ensemble")   # → saved_models/{symbol}_ensemble.pkl
            logger.info(f"{symbol} DONE: acc={best_acc:.3f} in {elapsed/60:.1f}min, {len(fold_results)} folds")
            print(f"      ✓ SAVED  acc={best_acc:.3f}  {elapsed/60:.1f}min  ({len(fold_results)} folds)\n")
            ckpt_save(symbol, {
                "status":     "done",
                "folds":      fold_results,
                "best_acc":   round(best_acc, 4),
                "elapsed":    round(elapsed, 1),
                "done_at":    datetime.now().isoformat(),
            })
            # Clean up intermediate checkpoint model files
            for f in CKPT_DIR.glob(f"{symbol}_fold*.pkl"): f.unlink(missing_ok=True)
            all_results.append({"symbol": symbol, "status": "ok", "acc": best_acc, "elapsed": elapsed})
        else:
            logger.warning(f"{symbol}: no successful folds")
            print(f"      ✗ FAILED (0 successful folds)\n")
            all_results.append({"symbol": symbol, "status": "failed", "acc": 0.0, "elapsed": elapsed})

    return all_results


# ── 4. Final report ───────────────────────────────────────────────────
def print_report(results, t_start):
    ok      = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skip"]
    failed  = [r for r in results if r["status"] == "failed"]
    total_m = (time.time() - t_start) / 60

    print(f"\n{SEP}")
    print(f"  TRAINING COMPLETE  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{SEP}")
    print(f"  OK: {len(ok)}  |  Skipped: {len(skipped)}  |  Failed: {len(failed)}")
    print(f"  Total time: {total_m:.0f} min\n")

    if ok:
        print(f"  {'Coin':<12} {'Accuracy':>10} {'Time':>8} {'Grade':>6}")
        print(f"  {'-'*38}")
        for r in sorted(ok, key=lambda x: x["acc"], reverse=True):
            coin  = r["symbol"].replace("USDT","")
            grade = "A★★★" if r["acc"]>=0.60 else "B★★" if r["acc"]>=0.55 else "C★" if r["acc"]>=0.52 else "D"
            print(f"  {coin:<12} {r['acc']:>10.3f} {r['elapsed']/60:>6.1f}m  {grade}")
        avg = sum(r["acc"] for r in ok) / len(ok)
        print(f"\n  Avg best-fold accuracy: {avg:.3f}")

    if failed:
        print(f"\n  FAILED: {[r['symbol'] for r in failed]}")

    REPORT.write_text(json.dumps({
        "completed_at": datetime.now().isoformat(),
        "total_min": round(total_m, 1),
        "ok": len(ok), "skipped": len(skipped), "failed": len(failed),
        "coins": results,
    }, indent=2), encoding="utf-8")

    print(f"\n{SEP}")
    print(f"  NEXT: python -m streamlit run dashboard\\app.py")
    print(f"{SEP}\n")


# ── Main ──────────────────────────────────────────────────────────────
def main():
    symbols = [f"{c}{QUOTE_ASSET}" for c in POPULAR_COINS]
    t_start = time.time()

    # Show resume status
    n_done = sum(1 for s in symbols if ckpt_is_done(s))
    n_prog = sum(1 for s in symbols
                 if not ckpt_is_done(s) and len(ckpt_load(s).get("folds",[])) > 0)

    print(f"\n{SEP}")
    print(f"  CryptoPredictBot — Training Pipeline  (checkpoint/resume)")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {len(symbols)} coins  |  {n_done} done  |  {n_prog} in-progress  |  {len(symbols)-n_done-n_prog} pending")
    print(f"{SEP}\n")
    logger.info("run_all.py started")

    verify_and_fix_data(symbols)
    prebuild_features(symbols)
    results = train_all_inprocess(symbols)
    print_report(results, t_start)
    logger.info("run_all.py complete")

if __name__ == "__main__":
    main()
