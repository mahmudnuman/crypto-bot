"""
predict/predictor.py — Real-time prediction engine.

For each symbol:
  1. Load latest candles (incremental fetch from Binance)
  2. Build feature matrix (multi-timeframe)
  3. Load the saved DualHeadEnsemble model
  4. Run Head A (direction) + Head B (price)
  5. Apply confidence gates via confidence.py
  6. Optionally log result to error log (when actual outcome is known)

Usage:
  python predict/predictor.py --symbol BTCUSDT
  python predict/predictor.py --all
"""
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.downloader import download_symbol
from data.store import load, last_timestamp
from features.multi_tf import build_multi_tf_features
from features.pipeline import get_feature_cols, prepare_Xy_classifier
from models.ensemble import DualHeadEnsemble
from models.online import log_prediction_result
from predict.confidence import compute_signal, Signal
from data.universe import get_symbols


def predict_symbol(symbol: str, update_data: bool = True) -> Signal | None:
    """
    Generate a prediction signal for the given symbol.
    Returns Signal (if conditions met) or None (SILENT).
    """

    # ── Step 1: Update candles ────────────────────────────────────────
    if update_data:
        try:
            download_symbol(symbol, ["5m", "1h", "1d"])
        except Exception as e:
            logger.warning(f"[{symbol}] Data update failed: {e}. Using cached data.")

    # ── Step 2: Build feature matrix ─────────────────────────────────
    # Use only last 1000 candles for speed (we only need the latest row)
    df = build_multi_tf_features(symbol)
    if df.empty:
        logger.warning(f"[{symbol}] No feature data.")
        return None

    feature_cols = get_feature_cols(df)
    latest_row   = df.iloc[[-1]]  # Last candle (most recent)

    if latest_row.empty:
        return None

    # Extract ADX for gating
    adx_col = "adx14" if "adx14" in df.columns else None
    adx     = float(latest_row["adx14"].iloc[0]) if adx_col else 0.0
    atr_pct = float(latest_row["atr14_pct"].iloc[0]) if "atr14_pct" in df.columns else 0.02
    current_close = load(symbol, "5m").iloc[-1]["close"] if not load(symbol, "5m").empty else 0.0

    # ── Step 3: Load model ────────────────────────────────────────────
    try:
        model = DualHeadEnsemble.load(symbol, "latest")
    except FileNotFoundError:
        logger.warning(f"[{symbol}] No trained model found. Run trainer.py first.")
        return None

    # ── Step 4: Head A — Direction ────────────────────────────────────
    X = latest_row[feature_cols].replace([float("inf"), float("-inf")], float("nan"))
    try:
        _, proba_arr      = model.predict_direction(X)
        proba_up          = float(proba_arr[0])
        base_probas       = model.base_model_probas(X)[0]   # shape (3,)
    except Exception as e:
        logger.error(f"[{symbol}] Head A prediction failed: {e}")
        return None

    # ── Step 5: Head B — Price ────────────────────────────────────────
    try:
        predicted_price = float(model.predict_price(X)[0])
    except Exception as e:
        logger.warning(f"[{symbol}] Head B prediction failed: {e}. Using current price.")
        predicted_price = float(current_close)

    # ── Step 6: Confidence gate ───────────────────────────────────────
    signal = compute_signal(
        symbol=symbol,
        proba_up=proba_up,
        base_probas=base_probas,
        predicted_price=predicted_price,
        current_price=float(current_close),
        adx=adx,
        atr_pct=atr_pct,
    )
    return signal


def predict_all(update_data: bool = True) -> list[Signal | None]:
    """Run predictions for all universe symbols."""
    symbols = get_symbols()
    results = []
    for sym in symbols:
        sig = predict_symbol(sym, update_data=update_data)
        results.append(sig)
    return results


def backfill_error_log(symbol: str) -> None:
    """
    Walk through stored 5m candles, generate predictions, and record
    outcomes (for historical accuracy benchmarking after the model is trained).
    Note: This uses the CURRENT model on historical data, not walk-forward.
    Use only for live-simulation testing, not training.
    """
    df = build_multi_tf_features(symbol)
    if df.empty:
        return
    try:
        model = DualHeadEnsemble.load(symbol, "latest")
    except FileNotFoundError:
        logger.warning(f"[{symbol}] No model; cannot backfill error log.")
        return

    feature_cols = get_feature_cols(df)
    X = df[feature_cols].replace([float("inf"), float("-inf")], float("nan"))
    _, proba_arr    = model.predict_direction(X)
    preds_price     = model.predict_price(X)
    y_dir           = df["target_dir"].values
    y_price         = df["target_price"].values if "target_price" in df.columns else np.full(len(df), float("nan"))
    base_all        = model.base_model_probas(X)

    for i, (ts, row) in enumerate(df.iterrows()):
        adx_val = float(row.get("adx14", 0.0))
        conf    = max(float(proba_arr[i]), 1 - float(proba_arr[i]))
        log_prediction_result(
            symbol          = symbol,
            open_time       = ts,
            predicted_dir   = int(proba_arr[i] >= 0.5),
            actual_dir      = int(y_dir[i]),
            confidence      = conf,
            predicted_price = float(preds_price[i]),
            actual_price    = float(y_price[i]),
            adx             = adx_val,
        )
    logger.success(f"[{symbol}] Error log backfilled with {len(df)} rows.")


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto Predictior")
    parser.add_argument("--symbol", type=str)
    parser.add_argument("--all",    action="store_true")
    parser.add_argument("--no-update", action="store_true",
                        help="Skip data download (use cached)")
    args = parser.parse_args()

    update = not args.no_update

    if args.symbol:
        sig = predict_symbol(args.symbol, update_data=update)
        if sig:
            print(f"\n{'='*50}")
            print(f"  Symbol:    {sig.symbol}")
            print(f"  Direction: {sig.direction}")
            print(f"  Confidence:{sig.confidence:.2%}")
            print(f"  Pred Price:{sig.predicted_price:.4f} ±{sig.price_band_pct:.2%}")
            print(f"  ADX:       {sig.adx:.1f} {'(STRONG)' if sig.is_strong_trend else ''}")
            print(f"{'='*50}")
        else:
            print(f"\n[{args.symbol}] No signal (market not trending or low confidence).")

    elif args.all:
        signals = predict_all(update_data=update)
        active  = [s for s in signals if s is not None]
        print(f"\n{'='*60}")
        print(f"  Active signals: {len(active)}/{len(signals)}")
        print(f"{'='*60}")
        for sig in active:
            band_str = f"±{sig.price_band_pct:.2%}"
            print(
                f"  {sig.symbol:12s} {sig.direction:4s} "
                f"conf={sig.confidence:.2%} price={sig.predicted_price:.4f}{band_str} "
                f"ADX={sig.adx:.1f}"
            )
    else:
        parser.print_help()
