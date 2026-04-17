"""
data/downloader.py — Downloads historical OHLCV candles from Binance.

Strategy:
  - 5m  data: Binance bulk data.binance.vision monthly ZIP files (fast, reliable)
  - 1h/1d data: Binance REST API /api/v3/klines with pagination & backoff
  - All strategies are resume-safe (skips already-stored months/ranges)

Usage:
  python data/downloader.py --symbols BTCUSDT ETHUSDT --tfs 5m 1h 1d
  python data/downloader.py --all          # download full universe
"""
import io
import time
import zipfile
import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from loguru import logger
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    BINANCE_BULK_BASE, BINANCE_REST_BASE,
    HISTORY_YEARS, REST_MAX_RETRIES, REST_BACKOFF_BASE,
    REST_RATE_LIMIT_PAUSE,
)
from data.store import save, last_timestamp
from data.universe import get_symbols
import progress as prog

# ── Helpers ──────────────────────────────────────────────────────────

def _tz_naive(ts):
    """Strip timezone — works for both tz-aware and tz-naive Timestamps."""
    if ts is None:
        return None
    ts = pd.Timestamp(ts)          # normalise to pd.Timestamp first
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)   # tz-aware → tz-naive (keeps UTC wall time)
    return ts

def _binance_get(url: str, params: dict) -> list:
    """GET with exponential backoff on 429 / 5xx errors."""
    for attempt in range(REST_MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=20)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                wait = REST_BACKOFF_BASE ** (attempt + 1)
                logger.warning(f"Rate limited. Sleeping {wait:.1f}s …")
                time.sleep(wait)
            elif resp.status_code >= 500:
                wait = REST_BACKOFF_BASE ** attempt
                logger.warning(f"Server error {resp.status_code}. Retry in {wait:.1f}s …")
                time.sleep(wait)
            else:
                logger.error(f"Unexpected HTTP {resp.status_code} for {url}")
                return []
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error: {e}. Retry …")
            time.sleep(REST_BACKOFF_BASE ** attempt)
    logger.error(f"Failed after {REST_MAX_RETRIES} retries: {url}")
    return []


# ── REST-based downloader (1h, 1d) ───────────────────────────────────

def download_rest(symbol: str, tf: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Fetch all candles for symbol/tf between start_ms and end_ms (UTC ms).
    Handles Binance 1000-row limit by paginating.
    """
    url    = f"{BINANCE_REST_BASE}/api/v3/klines"
    all_rows = []
    cur_start = start_ms

    while cur_start < end_ms:
        params = {
            "symbol":    symbol,
            "interval":  tf,
            "startTime": cur_start,
            "endTime":   end_ms,
            "limit":     1000,
        }
        rows = _binance_get(url, params)
        if not rows:
            break
        all_rows.extend(rows)
        # Move start to just after the last returned candle
        cur_start = rows[-1][0] + 1
        if len(rows) < 1000:
            break
        time.sleep(REST_RATE_LIMIT_PAUSE)

    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows)


def fetch_and_store_rest(symbol: str, tf: str) -> None:
    """Incremental REST download: picks up from last stored candle."""
    history_start = datetime.now(timezone.utc) - timedelta(days=HISTORY_YEARS * 365)
    last_ts   = _tz_naive(last_timestamp(symbol, tf))

    if last_ts is not None:
        start_dt = last_ts + timedelta(minutes=1)
        logger.info(f"[{symbol}/{tf}] Resuming from {start_dt.date()} …")
    else:
        start_dt = history_start
        logger.info(f"[{symbol}/{tf}] Full download from {start_dt.date()} …")

    prog.update("downloader",
        current_symbol=symbol,
        current_tf=tf,
        current_task=f"{symbol}/{tf} — REST fetch from {start_dt.date()}",
    )

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)

    df = download_rest(symbol, tf, start_ms, end_ms)
    if df.empty:
        logger.warning(f"[{symbol}/{tf}] No data returned.")
    else:
        save(symbol, tf, df)
        logger.success(f"[{symbol}/{tf}] Stored {len(df):,} candles.")

    state = prog.read()
    prog.update("downloader",
        completed_tasks=state["downloader"]["completed_tasks"] + 1,
        current_task=f"{symbol}/{tf} — done ({len(df):,} candles)",
    )


# ── Bulk ZIP downloader (5m) ──────────────────────────────────────────

def _bulk_zip_url(symbol: str, year: int, month: int) -> str:
    return f"{BINANCE_BULK_BASE}/{symbol}/5m/{symbol}-5m-{year}-{month:02d}.zip"


def _parse_zip(content: bytes) -> pd.DataFrame | None:
    """Parse a Binance bulk ZIP bytes object into a clean OHLCV DataFrame."""
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            with zf.open(zf.namelist()[0]) as f:
                raw = pd.read_csv(f, header=None, dtype=str)

        # Drop any header rows (Binance sometimes includes column names as row 0)
        # Real open_time values are 13-digit ms integers like 1609459200000
        def _is_numeric_ts(val):
            try:
                int(str(val).strip())
                return True
            except (ValueError, TypeError):
                return False

        mask = raw[0].apply(_is_numeric_ts)
        raw  = raw[mask].copy()

        if raw.empty:
            return None

        # Cast columns to correct types
        raw.columns = range(len(raw.columns))
        df = raw.copy()

        # Validate timestamp range: pandas datetime64[ns] supports ~1678–2261
        # Filter rows outside this range before conversion
        ts_ms = pd.to_numeric(df[0], errors="coerce")
        valid = (ts_ms >= 0) & (ts_ms <= 9_214_646_400_000)  # up to ~2261
        df    = df[valid].copy()

        if df.empty:
            return None

        return df

    except Exception as e:
        logger.error(f"Error parsing ZIP: {e}")
        return None


def fetch_and_store_5m_bulk(symbol: str) -> None:
    """
    Download 5m candles via monthly bulk ZIPs.
    Resume-safe: skips years already fully stored.
    """
    now          = datetime.now(timezone.utc)
    now_naive    = datetime.now()               # tz-naive for comparisons
    start_year   = now.year - HISTORY_YEARS
    last_ts      = _tz_naive(last_timestamp(symbol, "5m"))

    months = []
    for year in range(start_year, now.year + 1):
        for month in range(1, 13):
            dt = datetime(year, month, 1)      # tz-naive
            if dt > now_naive:
                break
            if last_ts is not None:
                month_end = (dt + timedelta(days=32)).replace(day=1) - timedelta(seconds=1)
                # Both month_end and last_ts are tz-naive — safe comparison
                if pd.Timestamp(month_end) <= last_ts:
                    continue
            months.append((year, month))

    if not months:
        logger.info(f"[{symbol}/5m] Already up to date.")
        prog.update("downloader",
            completed_tasks=prog.read()["downloader"]["completed_tasks"] + 1,
            current_task=f"{symbol}/5m — already up to date",
        )
        return

    logger.info(f"[{symbol}/5m] Downloading {len(months)} monthly files via bulk ZIP …")
    total_bytes = 0
    for i, (year, month) in enumerate(tqdm(months, desc=f"{symbol}/5m", unit="month")):
        prog.update("downloader",
            current_symbol=symbol,
            current_tf="5m",
            current_month=f"{year}-{month:02d}",
            done_months=i,
            total_months=len(months),
            current_task=f"{symbol}/5m  {year}-{month:02d}  ({i+1}/{len(months)})",
        )
        resp = requests.get(_bulk_zip_url(symbol, year, month), timeout=60)
        if resp.status_code == 404:
            continue
        if resp.status_code != 200:
            logger.warning(f"Bulk HTTP {resp.status_code}: {_bulk_zip_url(symbol, year, month)}")
            continue
        total_bytes += len(resp.content)
        try:
            df = _parse_zip(resp.content)
            if df is not None and not df.empty:
                save(symbol, "5m", df)
        except Exception as e:
            logger.error(f"Error parsing bulk ZIP: {e}")
        time.sleep(0.1)

    state = prog.read()
    prog.update("downloader",
        completed_tasks=state["downloader"]["completed_tasks"] + 1,
        bytes_downloaded=state["downloader"]["bytes_downloaded"] + total_bytes,
        done_months=len(months),
    )


# ── Orchestrator ──────────────────────────────────────────────────────

def download_symbol(symbol: str, tfs: list[str]) -> None:
    for tf in tfs:
        if tf == "5m":
            fetch_and_store_5m_bulk(symbol)      # bulk ZIPs (fast, historical)
            fetch_and_store_rest(symbol, "5m")   # REST gap-fill (recent months)
        else:
            fetch_and_store_rest(symbol, tf)     # 1h, 6h, 1d all via REST


def download_all(tfs: list[str] | None = None, max_workers: int = 3) -> None:
    """Download all universe symbols in parallel (default 3 workers)."""
    tfs     = tfs or ["5m", "1h", "1d"]
    symbols = get_symbols()
    logger.info(f"Starting download for {len(symbols)} symbols, timeframes: {tfs}")

    prog.start_downloader(symbols, tfs)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_symbol, sym, tfs): sym
            for sym in symbols
        }
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                fut.result()
                state = prog.read()
                done  = state["downloader"]["symbols_done"]
                done.append(sym)
                pend  = [s for s in state["downloader"]["symbols_pending"] if s != sym]
                prog.update("downloader", symbols_done=done, symbols_pending=pend)
                logger.success(f"[{sym}] Done.")
            except Exception as e:
                errs = prog.read()["downloader"]["errors"]
                errs.append(f"{sym}: {e}")
                prog.update("downloader", errors=errs)
                logger.error(f"[{sym}] Error: {e}")

    prog.finish_downloader()


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto OHLCV Downloader")
    parser.add_argument("--all",     action="store_true",  help="Download full universe")
    parser.add_argument("--symbols", nargs="+", default=[], help="Specific symbols e.g. BTCUSDT")
    parser.add_argument("--tfs",     nargs="+", default=["5m", "1h", "1d"])
    parser.add_argument("--workers", type=int,  default=3)
    args = parser.parse_args()

    if args.all:
        download_all(tfs=args.tfs, max_workers=args.workers)
    elif args.symbols:
        syms = args.symbols
        prog.start_downloader(syms, args.tfs)
        for sym in syms:
            download_symbol(sym, args.tfs)
            state = prog.read()
            done  = state["downloader"]["symbols_done"]
            done.append(sym)
            prog.update("downloader", symbols_done=done)
        prog.finish_downloader()
    else:
        parser.print_help()
