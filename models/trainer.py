"""
models/trainer.py — Walk-Forward Validation training loop.

Walk-forward schedule (default):
  ┌─────────────────────────────────────────────────────────────────┐
  │  TRAIN (3.5 yrs) │ GAP (14d) │ TEST (14d) │→ roll 14d forward  │
  └─────────────────────────────────────────────────────────────────┘

For each fold:
  1. Split data into train / gap / test
  2. Train DualHeadEnsemble on train split
  3. Evaluate on test split
  4. Record FoldResult
  5. Check for overfitting/underfitting via validator
  6. Roll forward by step_days

After all folds:
  - Build ValidationReport
  - Save best-performing model (by mean test accuracy)

Usage:
  python models/trainer.py --symbol BTCUSDT
  python models/trainer.py --all
"""
import argparse
import numpy as np
import pandas as pd
from datetime import timedelta
from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    WFV_INITIAL_TRAIN_YEARS,
    WFV_GAP_DAYS, WFV_TEST_DAYS, WFV_STEP_DAYS,
    HISTORY_YEARS, CACHE_DIR,
)
from data.universe import get_symbols
from features.multi_tf import build_multi_tf_features
from features.pipeline import (
    get_feature_cols, prepare_Xy_classifier, prepare_Xy_regressor
)
from models.ensemble import DualHeadEnsemble
from models.validator import analyse_fold, build_report
import progress as prog

FEATURE_CACHE_DIR = CACHE_DIR / "features"
FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _load_or_build_features(symbol: str) -> pd.DataFrame:
    """
    Load pre-built feature matrix from cache if fresh (< 12h old),
    otherwise rebuild and save. Avoids 30-min rebuild on every run.
    """
    cache_path = FEATURE_CACHE_DIR / f"{symbol}_features.parquet"

    if cache_path.exists():
        age_hours = (pd.Timestamp.now() - pd.Timestamp(cache_path.stat().st_mtime, unit="s")).total_seconds() / 3600
        if age_hours < 12:
            logger.info(f"[{symbol}] Loading cached features ({age_hours:.1f}h old)...")
            try:
                df = pd.read_parquet(cache_path)
                logger.success(f"[{symbol}] Cache hit: {df.shape[0]:,} rows x {df.shape[1]} cols")
                return df
            except Exception as e:
                logger.warning(f"[{symbol}] Cache load failed ({e}), rebuilding...")

    logger.info(f"[{symbol}] Building feature matrix (will cache)...")
    df = build_multi_tf_features(symbol)

    if not df.empty:
        try:
            df.to_parquet(cache_path)
            logger.debug(f"[{symbol}] Features cached to {cache_path.name}")
        except Exception as e:
            logger.warning(f"[{symbol}] Cache save failed: {e}")

    return df


# ── Walk-Forward Loop ─────────────────────────────────────────────────

def train_symbol(symbol: str, verbose: bool = True) -> dict:
    """
    Run full walk-forward training for one symbol.
    Returns dict with: report, best_accuracy, model_path.
    """
    logger.info(f"{'='*60}")
    logger.info(f"Walk-Forward Training: {symbol}")
    logger.info(f"{'='*60}")

    # ── Load feature matrix (cached) ───────────────────────────────────────
    df = _load_or_build_features(symbol)
    if df.empty or len(df) < 5000:
        logger.warning(f"[{symbol}] Insufficient data ({len(df)} rows). Skipping.")
        prog.update("trainer", current_symbol=f"{symbol} -- skipped (no data)")
        return {"symbol": symbol, "status": "skipped", "reason": "insufficient_data"}

    feature_cols = get_feature_cols(df)
    df = df.sort_index()

    # ── Define time boundaries ────────────────────────────────────────
    t_start  = df.index.min()
    t_end    = df.index.max()
    cur_train_end = t_start + timedelta(days=WFV_INITIAL_TRAIN_YEARS * 365)

    # Count total expected folds upfront for progress
    _test_count = 0
    _scan = cur_train_end
    while True:
        _ts = _scan + timedelta(days=WFV_GAP_DAYS)
        _te = _ts   + timedelta(days=WFV_TEST_DAYS)
        if _te > t_end:
            break
        _test_count += 1
        _scan += timedelta(days=WFV_STEP_DAYS)

    prog.update("trainer",
        current_symbol=symbol,
        current_fold=0,
        total_folds=_test_count,
        fold_results=[],
    )

    fold_results = []
    best_acc     = 0.0
    best_model   = None
    fold_idx     = 0

    # ── Walk-Forward Loop ─────────────────────────────────────────────
    while True:
        test_start = cur_train_end + timedelta(days=WFV_GAP_DAYS)
        test_end   = test_start    + timedelta(days=WFV_TEST_DAYS)

        if test_end > t_end:
            break  # No more test data

        # Split
        df_train = df[df.index < cur_train_end]
        df_test  = df[(df.index >= test_start) & (df.index < test_end)]

        if len(df_train) < 1000 or len(df_test) < 50:
            logger.warning(f"  Fold {fold_idx}: too few rows — skipping.")
            cur_train_end += timedelta(days=WFV_STEP_DAYS)
            continue

        # ── Prepare aligned X, y_dir, y_price ────────────────────────
        # Drop rows where EITHER target is NaN so all three arrays have same index
        def _prep_split(df_split):
            import numpy as np
            clean = df_split.dropna(subset=["target_dir", "target_price"])
            X       = clean[feature_cols].replace([np.inf, -np.inf], np.nan)
            y_dir   = clean["target_dir"].astype(int)
            y_price = clean["target_price"].astype(float)
            return X, y_dir, y_price

        X_train, y_dir_train, y_price_train = _prep_split(df_train)
        X_test,  y_dir_test,  y_price_test  = _prep_split(df_test)

        if len(X_train) < 500 or len(X_test) < 20:
            logger.warning(f"  Fold {fold_idx}: too few clean rows after target alignment — skipping.")
            cur_train_end += timedelta(days=WFV_STEP_DAYS)
            continue

        # Validation split from end of training (15%)
        val_split   = int(len(X_train) * 0.85)
        X_tr        = X_train.iloc[:val_split]
        y_tr_dir    = y_dir_train.iloc[:val_split]
        y_tr_price  = y_price_train.iloc[:val_split]
        X_val       = X_train.iloc[val_split:]
        y_val_dir   = y_dir_train.iloc[val_split:]
        y_val_price = y_price_train.iloc[val_split:]


        # Train
        model = DualHeadEnsemble(symbol)
        try:
            model.fit(
                X_tr, y_tr_dir, y_tr_price,
                X_val_raw=X_val,
                y_val_dir=y_val_dir,
                y_val_price=y_val_price,
            )
        except Exception as e:
            logger.error(f"  Fold {fold_idx} training failed: {e}")
            cur_train_end += timedelta(days=WFV_STEP_DAYS)
            continue

        # Evaluate on TRAIN set (for overfitting check)
        tr_metrics   = model.evaluate(X_tr, y_tr_dir, y_tr_price)
        # Evaluate on TEST set
        test_metrics = model.evaluate(X_test, y_dir_test, y_price_test)

        # Feature importances from LightGBM
        try:
            feature_names = list(X_tr.columns)
            importances   = dict(zip(
                feature_names,
                model.clf_lgbm.feature_importances_,
            ))
        except Exception:
            importances = {}

        fold_res = analyse_fold(
            fold_idx   = fold_idx,
            train_acc  = tr_metrics["accuracy"],
            test_acc   = test_metrics["accuracy"],
            train_f1   = tr_metrics["f1"],
            test_f1    = test_metrics["f1"],
            test_mape  = test_metrics["mape"],
            n_train    = len(X_tr),
            n_test     = len(X_test),
            feature_importances=importances,
        )
        fold_results.append(fold_res)

        # ── Update progress ───────────────────────────────────────────
        state       = prog.read()
        prev_folds  = state["trainer"].get("fold_results", [])
        prev_folds.append({
            "fold":      fold_idx,
            "train_acc": round(tr_metrics["accuracy"], 4),
            "test_acc":  round(test_metrics["accuracy"], 4),
            "mape":      round(float(test_metrics["mape"]) if not __import__('math').isnan(float(test_metrics["mape"])) else 0, 4),
        })
        prog.update("trainer",
            current_fold=fold_idx + 1,
            fold_results=prev_folds,
        )

        log_sym = f"[{symbol}] Fold {fold_idx:02d}"
        logger.info(
            f"{log_sym} | train={tr_metrics['accuracy']:.3f} "
            f"test={test_metrics['accuracy']:.3f} "
            f"mape={test_metrics['mape']:.3f} "
            f"conf={test_metrics['avg_conf']:.3f}"
        )

        # Keep best model
        if test_metrics["accuracy"] > best_acc:
            best_acc   = test_metrics["accuracy"]
            best_model = model

        # Roll forward
        fold_idx      += 1
        cur_train_end += timedelta(days=WFV_STEP_DAYS)

    # ── Post-training ─────────────────────────────────────────────────
    report = build_report(symbol, fold_results)

    if best_model is not None:
        model_path = best_model.save("latest")
        logger.success(f"[{symbol}] Best fold accuracy: {best_acc:.3f}")
    else:
        model_path = None
        logger.warning(f"[{symbol}] No model was trained successfully.")

    return {
        "symbol":        symbol,
        "status":        "ok",
        "best_accuracy": best_acc,
        "mean_accuracy": report.mean_test_acc,
        "model_path":    str(model_path),
        "report":        report,
    }


def train_all(max_workers: int = 1) -> list[dict]:
    """Train all universe symbols sequentially (or in parallel if >1 workers)."""
    symbols = get_symbols()
    prog.start_trainer(symbols)
    results = []
    for idx, sym in enumerate(symbols):
        prog.update("trainer",
            current_symbol=sym,
            completed_symbols=idx,
        )
        result = train_symbol(sym)
        results.append(result)
        # Record symbol completion
        state = prog.read()
        done  = state["trainer"]["symbols_done"]
        done.append({
            "symbol":   sym,
            "status":   result.get("status","ok"),
            "best_acc": round(result.get("best_accuracy", 0), 4),
        })
        prog.update("trainer",
            symbols_done=done,
            completed_symbols=idx + 1,
        )
    prog.finish_trainer()
    return results


# ── CLI ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-Forward Trainer")
    parser.add_argument("--symbol", type=str, help="Single symbol e.g. BTCUSDT")
    parser.add_argument("--all",    action="store_true", help="Train all symbols")
    args = parser.parse_args()

    if args.symbol:
        prog.start_trainer([args.symbol])
        res = train_symbol(args.symbol)
        prog.finish_trainer()
        print(f"\nResult: {res['status']} | accuracy={res.get('best_accuracy', 0):.3f}")
    elif args.all:
        results = train_all()
        for r in results:
            print(f"  {r['symbol']:12s} {r['status']} acc={r.get('best_accuracy', 0):.3f}")
    else:
        parser.print_help()
