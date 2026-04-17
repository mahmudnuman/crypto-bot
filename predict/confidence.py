"""
predict/confidence.py — ADX Trend Gate + Ensemble Confidence Scoring.

Pipeline:
  1. Check ADX (trend filter) — if ADX < 25 → SILENT (no signal)
  2. Check ensemble confidence — if max_proba < MIN_CONFIDENCE → SILENT
  3. Check ensemble agreement — if < 2/3 models agree → SILENT
  4. Boost confidence for strong trends (ADX > 35)
  5. Return structured Signal or None

Signal fields:
  symbol          : e.g. "BTCUSDT"
  direction       : "UP" | "DOWN"
  confidence      : 0.0 – 1.0 (calibrated probability)
  predicted_price : float (next 1h close estimate)
  price_band_pct  : ±% band around predicted price
  adx             : float (trend strength)
  is_strong_trend : bool
  timestamp       : UTC
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ADX_TREND_THRESHOLD, ADX_STRONG_TREND,
    MIN_CONFIDENCE, MIN_ENSEMBLE_AGREEMENT,
)


@dataclass
class Signal:
    symbol:          str
    direction:       str          # "UP" or "DOWN"
    confidence:      float        # 0.0 – 1.0
    predicted_price: float        # Head B output
    price_band_pct:  float        # e.g. 0.015 = ±1.5%
    adx:             float
    is_strong_trend: bool
    timestamp:       datetime


def compute_signal(
    symbol:        str,
    proba_up:      float,               # ensemble meta-classifier P(UP)
    base_probas:   np.ndarray,          # shape (3,) — per-base-model P(UP)
    predicted_price: float,
    current_price:   float,
    adx:             float,
    atr_pct:         float,             # ATR as % of price (for band width)
) -> Optional[Signal]:
    """
    Apply all confidence gates and return Signal or None.

    Parameters
    ----------
    proba_up       : meta-ensemble probability of UP
    base_probas    : array of [lgbm_p, xgb_p, catboost_p] UP probabilities
    predicted_price: Head B regression output
    current_price  : latest close price
    adx            : ADX(14) value
    atr_pct        : ATR(14) / close (volatility normalised)
    """

    # ── Gate 1: Trend filter ──────────────────────────────────────────
    if adx < ADX_TREND_THRESHOLD:
        logger.debug(f"[{symbol}] SILENT — ADX={adx:.1f} < {ADX_TREND_THRESHOLD} (no trend).")
        return None

    # ── Gate 2: Ensemble confidence ───────────────────────────────────
    confidence = max(proba_up, 1 - proba_up)  # distance from 0.5, mapped to [0.5, 1.0]
    if confidence < MIN_CONFIDENCE:
        logger.debug(f"[{symbol}] SILENT — confidence={confidence:.3f} < {MIN_CONFIDENCE}.")
        return None

    # ── Gate 3: Ensemble agreement ────────────────────────────────────
    direction     = "UP" if proba_up >= 0.5 else "DOWN"
    direction_int = 1 if direction == "UP" else 0
    base_votes    = int((base_probas >= 0.5).sum())   # number of models voting UP
    agreement     = base_votes if direction_int == 1 else (3 - base_votes)
    if agreement < MIN_ENSEMBLE_AGREEMENT:
        logger.debug(
            f"[{symbol}] SILENT — only {agreement}/3 models agree on {direction}."
        )
        return None

    # ── Gate 4: Boost for strong trend ───────────────────────────────
    is_strong = adx > ADX_STRONG_TREND
    if is_strong:
        # Slightly amplify confidence signal for strong trends
        confidence = min(1.0, confidence * 1.05)

    # ── Price band: use ATR-based uncertainty ─────────────────────────
    price_band_pct = atr_pct * 1.5  # 1.5× ATR as ±band

    signal = Signal(
        symbol=symbol,
        direction=direction,
        confidence=round(confidence, 4),
        predicted_price=round(predicted_price, 6),
        price_band_pct=round(price_band_pct, 4),
        adx=round(adx, 2),
        is_strong_trend=is_strong,
        timestamp=datetime.now(timezone.utc),
    )

    logger.info(
        f"[{symbol}] SIGNAL {direction} | conf={confidence:.2%} "
        f"| pred_price={predicted_price:.4f} ±{price_band_pct:.2%} "
        f"| ADX={adx:.1f} {'🔥' if is_strong else ''}"
    )
    return signal


def signals_to_dataframe(signals: list[Optional[Signal]]) -> pd.DataFrame:
    """Convert a list of Signal objects to a DataFrame (None entries dropped)."""
    rows = []
    for s in signals:
        if s is not None:
            rows.append({
                "symbol":          s.symbol,
                "direction":       s.direction,
                "confidence":      s.confidence,
                "predicted_price": s.predicted_price,
                "price_band_pct":  s.price_band_pct,
                "adx":             s.adx,
                "is_strong_trend": s.is_strong_trend,
                "timestamp":       s.timestamp,
            })
    return pd.DataFrame(rows)
