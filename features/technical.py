"""
features/technical.py — Computes 60+ technical analysis features.

All features are look-ahead safe: only past data is used.
Returns a DataFrame with all indicator columns appended.
"""
import numpy as np
import pandas as pd
from loguru import logger


# ── Public API ────────────────────────────────────────────────────────

def add_all_features(df: pd.DataFrame, tf_label: str = "") -> pd.DataFrame:
    """
    Add all technical features to a OHLCV DataFrame.
    Columns expected: open, high, low, close, volume
    tf_label: '' for primary tf; '1h_' or '1d_' for cross-tf prefixes.
    """
    df = df.copy()
    p  = tf_label  # prefix

    # ── Price-derived basics ──────────────────────────────────────────
    df[f"{p}returns"]    = df["close"].pct_change()
    df[f"{p}log_ret"]    = np.log(df["close"] / df["close"].shift(1))
    df[f"{p}hl_ratio"]   = (df["high"] - df["low"]) / df["close"]
    df[f"{p}oc_ratio"]   = (df["close"] - df["open"]) / df["open"]
    df[f"{p}upper_wick"] = (df["high"] - df[["open","close"]].max(axis=1)) / df["close"]
    df[f"{p}lower_wick"] = (df[["open","close"]].min(axis=1) - df["low"]) / df["close"]
    df[f"{p}body_size"]  = abs(df["close"] - df["open"]) / df["close"]
    df[f"{p}is_bull"]    = (df["close"] > df["open"]).astype(int)

    # ── Moving Averages ───────────────────────────────────────────────
    for w in [8, 21, 55, 200]:
        df[f"{p}ema{w}"]      = df["close"].ewm(span=w, adjust=False).mean()
        df[f"{p}ema{w}_dist"] = (df["close"] - df[f"{p}ema{w}"]) / df[f"{p}ema{w}"]
    for w in [10, 20, 50]:
        df[f"{p}sma{w}"]      = df["close"].rolling(w).mean()
        df[f"{p}sma{w}_dist"] = (df["close"] - df[f"{p}sma{w}"]) / df[f"{p}sma{w}"]

    # ── MACD ──────────────────────────────────────────────────────────
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df[f"{p}macd"]        = ema12 - ema26
    df[f"{p}macd_signal"] = df[f"{p}macd"].ewm(span=9, adjust=False).mean()
    df[f"{p}macd_hist"]   = df[f"{p}macd"] - df[f"{p}macd_signal"]

    # ── RSI ───────────────────────────────────────────────────────────
    for period in [7, 14, 21]:
        df[f"{p}rsi{period}"] = _rsi(df["close"], period)

    # ── Stochastic ────────────────────────────────────────────────────
    df[f"{p}stoch_k"], df[f"{p}stoch_d"] = _stochastic(df)

    # ── Bollinger Bands ───────────────────────────────────────────────
    sma20  = df["close"].rolling(20).mean()
    std20  = df["close"].rolling(20).std()
    df[f"{p}bb_upper"] = sma20 + 2 * std20
    df[f"{p}bb_lower"] = sma20 - 2 * std20
    df[f"{p}bb_width"] = (df[f"{p}bb_upper"] - df[f"{p}bb_lower"]) / sma20
    df[f"{p}bb_pct"]   = (df["close"] - df[f"{p}bb_lower"]) / (
                          df[f"{p}bb_upper"] - df[f"{p}bb_lower"] + 1e-9)

    # ── ATR ───────────────────────────────────────────────────────────
    df[f"{p}atr14"] = _atr(df, 14)
    df[f"{p}atr7"]  = _atr(df, 7)
    df[f"{p}atr14_pct"] = df[f"{p}atr14"] / df["close"]

    # ── ADX (Trend Strength) ──────────────────────────────────────────
    df[f"{p}adx14"], df[f"{p}di_plus"], df[f"{p}di_minus"] = _adx(df, 14)
    df[f"{p}is_trending"] = (df[f"{p}adx14"] > 25).astype(int)
    df[f"{p}trend_strength"] = df[f"{p}adx14"] / 100.0  # normalised 0-1

    # ── CCI ───────────────────────────────────────────────────────────
    df[f"{p}cci14"] = _cci(df, 14)

    # ── Williams %R ───────────────────────────────────────────────────
    df[f"{p}willr14"] = _williams_r(df, 14)

    # ── Volume features ───────────────────────────────────────────────
    df[f"{p}vol_sma20"]   = df["volume"].rolling(20).mean()
    df[f"{p}vol_ratio"]   = df["volume"] / (df[f"{p}vol_sma20"] + 1e-9)
    df[f"{p}vol_returns"] = df["volume"].pct_change()
    df[f"{p}obv"]         = _obv(df)
    df[f"{p}obv_ema10"]   = df[f"{p}obv"].ewm(span=10, adjust=False).mean()
    df[f"{p}obv_dist"]    = (df[f"{p}obv"] - df[f"{p}obv_ema10"]) / (abs(df[f"{p}obv_ema10"]) + 1e-9)

    # ── VWAP (rolling daily proxy) ────────────────────────────────────
    tp = (df["high"] + df["low"] + df["close"]) / 3
    df[f"{p}vwap20"]      = (tp * df["volume"]).rolling(20).sum() / (df["volume"].rolling(20).sum() + 1e-9)
    df[f"{p}vwap_dist"]   = (df["close"] - df[f"{p}vwap20"]) / (df[f"{p}vwap20"] + 1e-9)

    # ── Rolling return statistics ─────────────────────────────────────
    for w in [5, 12, 24, 72]:
        df[f"{p}ret_mean{w}"] = df[f"{p}returns"].rolling(w).mean()
        df[f"{p}ret_std{w}"]  = df[f"{p}returns"].rolling(w).std()
        df[f"{p}ret_skew{w}"] = df[f"{p}returns"].rolling(w).skew()

    # ── Lag features ──────────────────────────────────────────────────
    for lag in range(1, 11):
        df[f"{p}lag_ret{lag}"]  = df[f"{p}returns"].shift(lag)
        df[f"{p}lag_vol{lag}"]  = df[f"{p}vol_ratio"].shift(lag)

    # ── Momentum ──────────────────────────────────────────────────────
    for w in [5, 10, 20]:
        df[f"{p}mom{w}"] = df["close"].pct_change(w)

    # ── Gap ───────────────────────────────────────────────────────────
    df[f"{p}gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

    logger.debug(f"Added {len([c for c in df.columns if c.startswith(p)])} features (prefix='{p}').")
    return df


# ── Internal calculators ──────────────────────────────────────────────

def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta  = series.diff()
    gain   = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss   = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs     = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _stochastic(df: pd.DataFrame, k=14, d=3):
    low_min  = df["low"].rolling(k).min()
    high_max = df["high"].rolling(k).max()
    stoch_k  = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-9)
    stoch_d  = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def _adx(df: pd.DataFrame, period: int):
    prev_high  = df["high"].shift(1)
    prev_low   = df["low"].shift(1)
    prev_close = df["close"].shift(1)

    up_move   = df["high"]  - prev_high
    down_move = prev_low    - df["low"]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_s  = pd.Series(plus_dm,  index=df.index).ewm(com=period-1, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm, index=df.index).ewm(com=period-1, adjust=False).mean()
    atr        = _atr(df, period)

    di_plus   = 100 * plus_dm_s  / (atr + 1e-9)
    di_minus  = 100 * minus_dm_s / (atr + 1e-9)
    dx        = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-9)
    adx       = dx.ewm(com=period-1, adjust=False).mean()
    return adx, di_plus, di_minus


def _cci(df: pd.DataFrame, period: int) -> pd.Series:
    tp       = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp   = tp.rolling(period).mean()
    mean_dev = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma_tp) / (0.015 * mean_dev + 1e-9)


def _williams_r(df: pd.DataFrame, period: int) -> pd.Series:
    high_max = df["high"].rolling(period).max()
    low_min  = df["low"].rolling(period).min()
    return -100 * (high_max - df["close"]) / (high_max - low_min + 1e-9)


def _obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"] - df["close"].shift(1))
    return (direction * df["volume"]).cumsum()
