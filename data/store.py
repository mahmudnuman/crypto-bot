"""
data/store.py — Parquet-based local data store.

Layout:
  cache/
    klines/
      BTCUSDT/
        5m/
          2021.parquet
          2022.parquet
          ...
        1h/
          all.parquet
        1d/
          all.parquet

API:
  save(symbol, tf, df)
  load(symbol, tf, start, end) → pd.DataFrame
  last_timestamp(symbol, tf) → pd.Timestamp | None
"""
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from loguru import logger
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CACHE_DIR

KLINES_DIR = CACHE_DIR / "klines"
KLINES_DIR.mkdir(parents=True, exist_ok=True)

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "num_trades",
    "taker_buy_base_vol", "taker_buy_quote_vol", "ignore",
]
# All columns that should be numeric (pd.to_numeric with errors='coerce')
NUMERIC_COLS = [
    "open", "high", "low", "close", "volume",
    "quote_volume", "num_trades",
    "taker_buy_base_vol", "taker_buy_quote_vol",
]


def _coin_dir(symbol: str, tf: str) -> Path:
    d = KLINES_DIR / symbol / tf
    d.mkdir(parents=True, exist_ok=True)
    return d


def _partition_key(symbol: str, tf: str) -> str:
    """5m uses yearly partitions; 1h and 1d use a single 'all' file."""
    return "year" if tf == "5m" else "all"


def save(symbol: str, tf: str, df: pd.DataFrame) -> None:
    """
    Upsert rows into the parquet store. Handles deduplication on open_time.
    """
    if df.empty:
        return

    df = _normalise(df)
    coin_dir = _coin_dir(symbol, tf)

    if _partition_key(symbol, tf) == "year":
        # Split by year for 5m data to keep files manageable
        for year, group in df.groupby(df["open_time"].dt.year):
            fpath = coin_dir / f"{year}.parquet"
            _upsert_parquet(fpath, group)
    else:
        fpath = coin_dir / "all.parquet"
        _upsert_parquet(fpath, df)


def _upsert_parquet(fpath: Path, new_df: pd.DataFrame) -> None:
    if fpath.exists():
        existing = pd.read_parquet(fpath)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.drop_duplicates(subset=["open_time"], keep="last", inplace=True)
        combined.sort_values("open_time", inplace=True)
        combined.reset_index(drop=True, inplace=True)
    else:
        combined = new_df
    combined.to_parquet(fpath, index=False, compression="snappy")
    logger.debug(f"Saved {len(combined)} rows → {fpath.name}")


def load(
    symbol: str,
    tf: str,
    start: Optional[pd.Timestamp] = None,
    end:   Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Load candles for a symbol+timeframe, optionally filtered by date range."""
    coin_dir = _coin_dir(symbol, tf)
    files    = sorted(coin_dir.glob("*.parquet"))
    if not files:
        logger.warning(f"No data found for {symbol}/{tf}")
        return pd.DataFrame(columns=KLINE_COLUMNS)

    parts = [pd.read_parquet(f) for f in files]
    df    = pd.concat(parts, ignore_index=True)
    df.sort_values("open_time", inplace=True)
    df.drop_duplicates(subset=["open_time"], keep="last", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if start is not None:
        s = pd.Timestamp(start)
        if s.tzinfo is not None:
            s = s.tz_convert(None)
        df = df[df["open_time"] >= s]
    if end is not None:
        e = pd.Timestamp(end)
        if e.tzinfo is not None:
            e = e.tz_convert(None)
        df = df[df["open_time"] <= e]

    return df.reset_index(drop=True)


def last_timestamp(symbol: str, tf: str) -> Optional[pd.Timestamp]:
    """Return the open_time of the most recent stored candle, or None."""
    df = load(symbol, tf)
    if df.empty:
        return None
    return df["open_time"].max()


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure standard column names, dtypes, and timezone-naive UTC timestamps."""
    if len(df.columns) == len(KLINE_COLUMNS):
        df.columns = KLINE_COLUMNS

    # Drop the useless 'ignore' column if present
    if "ignore" in df.columns:
        df = df.drop(columns=["ignore"])

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert open_time ms integer → tz-naive datetime64[ns]
    if not pd.api.types.is_datetime64_any_dtype(df["open_time"]):
        ot = pd.to_numeric(df["open_time"], errors="coerce")
        df["open_time"] = pd.to_datetime(ot, unit="ms", utc=True, errors="coerce").dt.tz_convert(None)
    elif df["open_time"].dt.tz is not None:
        df["open_time"] = df["open_time"].dt.tz_convert(None)

    # close_time: also convert to datetime (or drop — we don't use it for ML)
    if "close_time" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["close_time"]):
            ct = pd.to_numeric(df["close_time"], errors="coerce")
            df["close_time"] = pd.to_datetime(ct, unit="ms", utc=True, errors="coerce").dt.tz_convert(None)
        elif df["close_time"].dt.tz is not None:
            df["close_time"] = df["close_time"].dt.tz_convert(None)

    # Drop rows where open_time couldn't be parsed
    df = df.dropna(subset=["open_time"]).reset_index(drop=True)
    return df

