"""
features/multi_tf.py — Aligns 1h, 6h, and 1d features onto 5m candle timestamps.

Process:
1. Load 5m, 1h, 6h, 1d DataFrames
2. Compute technical indicators on each timeframe independently
3. Merge higher-tf features onto 5m timeline via merge_asof (forward-fill, no look-ahead)
4. Add inter-timeframe relationship features (cross-tf signals)
5. Return combined DataFrame ready for model training

No look-ahead bias: we always use the candle that CLOSED before the 5m candle opened.
"""
import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.store import load
from features.technical import add_all_features


def _tz_naive(ts):
    """Strip timezone from any Timestamp."""
    if ts is None:
        return None
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts


def _merge_tf_onto_5m(df_5m: pd.DataFrame, df_higher: pd.DataFrame,
                      prefix: str) -> pd.DataFrame:
    """
    Merge higher-timeframe features onto 5m index with no look-ahead.
    Uses searchsorted for memory-efficient O(n log n) merge — no extra allocations.
    Always returns df with open_time as the index.
    """
    # Ensure 5m has open_time as index
    if "open_time" in df_5m.columns:
        df_5m = df_5m.set_index("open_time")

    if df_higher.empty:
        return df_5m

    df_h = add_all_features(df_higher, tf_label=prefix)
    df_h = df_h.set_index("open_time").sort_index()

    # Shift by 1: use only CLOSED bars (no look-ahead bias)
    df_h_shifted = df_h.shift(1)
    cols = [c for c in df_h_shifted.columns if c.startswith(prefix)]
    if not cols:
        return df_5m

    # Efficient backward-fill merge using searchsorted:
    # For each 5m timestamp, find the last higher-tf bar that opened before it.
    h_index   = df_h_shifted.index.values  # numpy array for searchsorted
    s_index   = df_5m.sort_index().index.values
    df_5m     = df_5m.sort_index()

    # searchsorted: side='right'-1 → last bar at or before each 5m timestamp
    positions  = np.searchsorted(h_index, s_index, side="right") - 1
    valid_mask = positions >= 0
    valid_pos  = positions[valid_mask]

    # Assign each higher-tf column directly into df_5m (no full DataFrame copy)
    h_values = df_h_shifted[cols].values  # (n_higher_bars, n_cols) — small array
    n_5m     = len(df_5m)
    for j, col in enumerate(cols):
        col_data          = np.full(n_5m, np.nan, dtype=np.float64)
        col_data[valid_mask] = h_values[valid_pos, j].astype(np.float64)
        df_5m[col]        = col_data  # direct in-place assignment

    return df_5m




def _add_cross_tf_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add inter-timeframe relationship features that capture disagreements
    and alignments between timeframes — these are among the most predictive features.
    """
    p = df.columns.tolist()

    # RSI divergence between timeframes
    if "rsi14" in p and "1h_rsi14" in p:
        df["cross_rsi_5m_1h"] = df["rsi14"] - df["1h_rsi14"]
    if "rsi14" in p and "6h_rsi14" in p:
        df["cross_rsi_5m_6h"] = df["rsi14"] - df["6h_rsi14"]
    if "1h_rsi14" in p and "6h_rsi14" in p:
        df["cross_rsi_1h_6h"] = df["1h_rsi14"] - df["6h_rsi14"]
    if "6h_rsi14" in p and "1d_rsi14" in p:
        df["cross_rsi_6h_1d"] = df["6h_rsi14"] - df["1d_rsi14"]

    # Trend alignment: all timeframes trending in same direction?
    trending_cols = [c for c in ["is_trending", "1h_is_trending", "6h_is_trending", "1d_is_trending"] if c in p]
    if len(trending_cols) >= 2:
        df["cross_trend_alignment"] = df[trending_cols].sum(axis=1) / len(trending_cols)

    # ADX comparison (trend strength across timeframes)
    if "adx14" in p and "1h_adx14" in p:
        df["cross_adx_5m_vs_1h"] = df["adx14"] - df["1h_adx14"]
    if "1h_adx14" in p and "6h_adx14" in p:
        df["cross_adx_1h_vs_6h"] = df["1h_adx14"] - df["6h_adx14"]
    if "6h_adx14" in p and "1d_adx14" in p:
        df["cross_adx_6h_vs_1d"] = df["6h_adx14"] - df["1d_adx14"]

    # EMA distance across timeframes (price position)
    if "ema21_dist" in p and "1h_ema21_dist" in p:
        df["cross_ema21_5m_vs_1h"] = df["ema21_dist"] - df["1h_ema21_dist"]
    if "1h_ema21_dist" in p and "6h_ema21_dist" in p:
        df["cross_ema21_1h_vs_6h"] = df["1h_ema21_dist"] - df["6h_ema21_dist"]
    if "6h_ema21_dist" in p and "1d_ema21_dist" in p:
        df["cross_ema21_6h_vs_1d"] = df["6h_ema21_dist"] - df["1d_ema21_dist"]

    # MACD agreement
    if "macd_hist" in p and "1h_macd_hist" in p:
        df["cross_macd_agree_5m_1h"] = (
            np.sign(df["macd_hist"]) == np.sign(df["1h_macd_hist"])
        ).astype(int)
    if "1h_macd_hist" in p and "6h_macd_hist" in p:
        df["cross_macd_agree_1h_6h"] = (
            np.sign(df["1h_macd_hist"]) == np.sign(df["6h_macd_hist"])
        ).astype(int)
    if "6h_macd_hist" in p and "1d_macd_hist" in p:
        df["cross_macd_agree_6h_1d"] = (
            np.sign(df["6h_macd_hist"]) == np.sign(df["1d_macd_hist"])
        ).astype(int)

    # BB position cross-tf
    if "bb_pct" in p and "1h_bb_pct" in p:
        df["cross_bb_5m_vs_1h"] = df["bb_pct"] - df["1h_bb_pct"]
    if "1h_bb_pct" in p and "6h_bb_pct" in p:
        df["cross_bb_1h_vs_6h"] = df["1h_bb_pct"] - df["6h_bb_pct"]

    # Volume regime divergence
    if "vol_ratio" in p and "1h_vol_ratio" in p:
        df["cross_vol_5m_vs_1h"] = df["vol_ratio"] - df["1h_vol_ratio"]
    if "1h_vol_ratio" in p and "6h_vol_ratio" in p:
        df["cross_vol_1h_vs_6h"] = df["1h_vol_ratio"] - df["6h_vol_ratio"]

    return df


def build_multi_tf_features(
    symbol:     str,
    start:      pd.Timestamp | None = None,
    end:        pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Build full multi-timeframe feature matrix for a symbol.

    Returns a 5m-indexed DataFrame with:
    - Primary 5m OHLCV + all technical features (no prefix)
    - 1h features  (prefix '1h_')
    - 6h features  (prefix '6h_')  <-- NEW
    - 1d features  (prefix '1d_')
    - Cross-tf relationship features (prefix 'cross_')
    - Target columns:
        - target_dir   — 1 if next 5m close > current close, else 0
        - target_price — next 1h close price (regression head B)
    """
    logger.info(f"[{symbol}] Building multi-tf feature matrix (5m+1h+6h+1d)…")

    start = _tz_naive(start)
    end   = _tz_naive(end)

    # ── Load all four timeframes ─────────────────────────────────────
    df_5m = load(symbol, "5m", start, end)
    df_1h = load(symbol, "1h", start, end)
    df_6h = load(symbol, "6h", start, end)
    df_1d = load(symbol, "1d", start, end)

    if df_5m.empty:
        logger.warning(f"[{symbol}] No 5m data — skipping.")
        return pd.DataFrame()

    # ── Compute 5m technical features (primary timeframe) ────────────
    df_5m = add_all_features(df_5m, tf_label="")
    df_5m = df_5m.set_index("open_time")

    # ── Merge each higher timeframe → 5m (returns open_time as index) ─
    df_5m = _merge_tf_onto_5m(df_5m, df_1h, prefix="1h_")
    df_5m = _merge_tf_onto_5m(df_5m, df_6h, prefix="6h_")
    df_5m = _merge_tf_onto_5m(df_5m, df_1d, prefix="1d_")
    df_5m = df_5m.copy()   # defragment after many in-place column assignments

    # ── Add cross-timeframe relationship features ─────────────────────
    df_5m = _add_cross_tf_features(df_5m)


    # ── Target A: next 5m direction ─────────────────────────────────
    df_5m["target_dir"] = (df_5m["close"].shift(-1) > df_5m["close"]).astype(int)

    # ── Target B: next 1h close (regression) ────────────────────────
    if not df_1h.empty:
        df_1h_close = df_1h.set_index("open_time")["close"].rename("next1h_close")
        df_1h_next  = df_1h_close.shift(-1)
        df_5m = pd.merge_asof(
            df_5m.sort_index().reset_index(),
            df_1h_next.reset_index(),
            left_on="open_time",
            right_on="open_time",
            direction="backward",
        ).set_index("open_time")
        df_5m.rename(columns={"next1h_close": "target_price"}, inplace=True)
    else:
        df_5m["target_price"] = np.nan

    # ── Drop boundary rows with NaN targets ──────────────────────────
    df_5m.dropna(subset=["target_dir"], inplace=True)

    # ── Drop raw OHLCV columns (prevent data leakage) ────────────────
    drop_raw = ["open", "high", "low", "volume", "quote_volume", "num_trades",
                "taker_buy_base_vol", "taker_buy_quote_vol", "ignore",
                "close_time", "close"]
    df_5m.drop(columns=[c for c in drop_raw if c in df_5m.columns], inplace=True)

    n_cross = len([c for c in df_5m.columns if c.startswith("cross_")])
    n_6h    = len([c for c in df_5m.columns if c.startswith("6h_")])
    logger.success(
        f"[{symbol}] Feature matrix: {df_5m.shape[0]:,} rows x {df_5m.shape[1]} cols "
        f"(incl. {n_6h} 6h-features, {n_cross} cross-tf features)"
    )
    return df_5m
