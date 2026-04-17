"""
models/online.py — Continuous (online) re-learning after each prediction cycle.

Strategy:
  1. Every ONLINE_RETRAIN_INTERVAL_HOURS:
     a. Fetch new candles (incremental)
     b. Log prediction errors from the last cycle
     c. Compute 7-day rolling accuracy
     d. If accuracy < ONLINE_RETRAIN_TRIGGER_ACC → trigger sliding-window retrain

Error log schema (error_log.parquet):
  open_time, symbol, predicted_dir, actual_dir, correct,
  predicted_price, actual_price, confidence, adx

The sliding-window retrain uses only the most recent ONLINE_SLIDING_WINDOW_DAYS
of data, so the model doesn't drift back into outdated market regimes.
"""
import time
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from loguru import logger
from apscheduler.schedulers.background import BackgroundScheduler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ONLINE_RETRAIN_INTERVAL_HOURS,
    ONLINE_RETRAIN_TRIGGER_ACC,
    ONLINE_SLIDING_WINDOW_DAYS,
    MODEL_DIR, CACHE_DIR,
)
from data.downloader import download_symbol
from data.universe import get_symbols
from features.multi_tf import build_multi_tf_features
from features.pipeline import get_feature_cols, prepare_Xy_classifier, prepare_Xy_regressor
from models.ensemble import DualHeadEnsemble

ERROR_LOG_PATH = CACHE_DIR / "error_log.parquet"


# ── Error Logging ─────────────────────────────────────────────────────

def log_prediction_result(
    symbol:          str,
    open_time:       pd.Timestamp,
    predicted_dir:   int,
    actual_dir:      int,
    confidence:      float,
    predicted_price: float,
    actual_price:    float,
    adx:             float,
) -> None:
    """Append one prediction outcome to the error log."""
    record = pd.DataFrame([{
        "open_time":       open_time,
        "symbol":          symbol,
        "predicted_dir":   predicted_dir,
        "actual_dir":      actual_dir,
        "correct":         int(predicted_dir == actual_dir),
        "confidence":      confidence,
        "predicted_price": predicted_price,
        "actual_price":    actual_price,
        "adx":             adx,
        "logged_at":       datetime.now(timezone.utc),
    }])

    if ERROR_LOG_PATH.exists():
        existing = pd.read_parquet(ERROR_LOG_PATH)
        combined = pd.concat([existing, record], ignore_index=True)
    else:
        combined = record

    combined.to_parquet(ERROR_LOG_PATH, index=False, compression="snappy")


def get_rolling_accuracy(symbol: str, days: int = 7) -> float | None:
    """Return rolling accuracy for a symbol over the last `days` days."""
    if not ERROR_LOG_PATH.exists():
        return None
    df = pd.read_parquet(ERROR_LOG_PATH)
    df = df[df["symbol"] == symbol]
    if df.empty:
        return None

    df["logged_at"] = pd.to_datetime(df["logged_at"])
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    recent = df[df["logged_at"] >= cutoff]
    if len(recent) < 10:
        return None
    return float(recent["correct"].mean())


# ── Sliding-Window Retrain ────────────────────────────────────────────

def retrain_symbol_sliding(symbol: str) -> bool:
    """
    Retrain the model for `symbol` using only the most recent
    ONLINE_SLIDING_WINDOW_DAYS of data.
    Returns True on success.
    """
    logger.info(f"[{symbol}] Sliding-window retrain triggered …")

    # Download latest candles first
    try:
        download_symbol(symbol, ["5m", "1h", "1d"])
    except Exception as e:
        logger.error(f"[{symbol}] Data update failed: {e}")

    end   = pd.Timestamp.now()   # tz-naive to match stored datetime64[ns]
    start = end - pd.Timedelta(days=ONLINE_SLIDING_WINDOW_DAYS)

    df = build_multi_tf_features(symbol, start=start, end=end)
    if df.empty or len(df) < 500:
        logger.warning(f"[{symbol}] Not enough data for sliding retrain.")
        return False

    feature_cols = get_feature_cols(df)
    split = int(len(df) * 0.85)
    df_train = df.iloc[:split]
    df_val   = df.iloc[split:]

    X_train, y_dir_train = prepare_Xy_classifier(df_train, feature_cols)
    _, y_price_train     = prepare_Xy_regressor(df_train, feature_cols)
    X_val,   y_val_dir   = prepare_Xy_classifier(df_val,  feature_cols)
    _, y_val_price       = prepare_Xy_regressor(df_val, feature_cols)

    model = DualHeadEnsemble(symbol)
    try:
        model.fit(
            X_train, y_dir_train, y_price_train,
            X_val_raw=X_val, y_val_dir=y_val_dir, y_val_price=y_val_price,
        )
        model.save("latest")
        logger.success(f"[{symbol}] Sliding-window retrain complete.")
        return True
    except Exception as e:
        logger.error(f"[{symbol}] Retrain failed: {e}")
        return False


# ── Scheduled Check ───────────────────────────────────────────────────

def scheduled_check():
    """Called by APScheduler every ONLINE_RETRAIN_INTERVAL_HOURS hours."""
    logger.info("Scheduled online learning check …")
    symbols = get_symbols()
    for sym in symbols:
        acc = get_rolling_accuracy(sym, days=7)
        if acc is None:
            logger.debug(f"[{sym}] No error log yet; skipping.")
            continue
        logger.info(f"[{sym}] 7d rolling accuracy: {acc:.3f}")
        if acc < ONLINE_RETRAIN_TRIGGER_ACC:
            logger.warning(f"[{sym}] Accuracy {acc:.3f} < {ONLINE_RETRAIN_TRIGGER_ACC} → retraining …")
            retrain_symbol_sliding(sym)


def start_scheduler() -> BackgroundScheduler:
    """Start the background scheduler. Call this from the dashboard or main."""
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        scheduled_check,
        "interval",
        hours=ONLINE_RETRAIN_INTERVAL_HOURS,
        id="online_check",
        replace_existing=True,
    )
    scheduler.start()
    logger.info(f"Online scheduler started (every {ONLINE_RETRAIN_INTERVAL_HOURS}h).")
    return scheduler


if __name__ == "__main__":
    # Manual trigger for testing
    scheduled_check()
