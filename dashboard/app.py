"""
dashboard/app.py — Live Streamlit Dashboard for CryptoPredictBot.

Sections:
  1. Sidebar   — coin selector, settings
  2. Live Price Chart — real-time 5m candles (Plotly candlestick)
  3. Prediction Panel — Head A (direction) + Head B (price) with gauges
  4. Accuracy Tracker — rolling 7d/30d/all-time accuracy from error_log
  5. Model Health     — fold metrics, overfitting scores, last train date
  6. Error Log Viewer — recent mistakes with confidence context
  7. Controls         — Re-train button, data update button

Run:
  streamlit run dashboard/app.py
"""
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone, timedelta
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    POPULAR_COINS, CACHE_DIR, MODEL_DIR, BINANCE_REST_BASE,
    ADX_TREND_THRESHOLD, ADX_STRONG_TREND, DASHBOARD_DEFAULT_COIN,
)
from data.universe import build_universe
from models.online import get_rolling_accuracy, ERROR_LOG_PATH

# ── Page Config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="CryptoPredictBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* Dark background */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1529 50%, #0a1020 100%);
    color: #e2e8f0;
}

/* Hide default header/footer */
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1529 0%, #111827 100%) !important;
    border-right: 1px solid #1e2d4a;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #0d1f3c 0%, #132040 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(59, 130, 246, 0.2);
}
.metric-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.3rem; }
.metric-value { font-size: 2rem; font-weight: 700; color: #f1f5f9; }
.metric-sub   { font-size: 0.7rem; color: #475569; margin-top: 0.2rem; }

/* Signal cards */
.signal-up   { background: linear-gradient(135deg, #064e3b, #065f46); border: 1px solid #10b981; border-radius: 12px; padding: 1rem; }
.signal-down { background: linear-gradient(135deg, #7f1d1d, #991b1b); border: 1px solid #ef4444; border-radius: 12px; padding: 1rem; }
.signal-none { background: linear-gradient(135deg, #1c1c2e, #252545); border: 1px solid #4b5563; border-radius: 12px; padding: 1rem; }

/* Section headers */
.section-header {
    font-size: 1rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 1rem 0 0.6rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid #1e2d4a;
}

/* Confidence bar */
.conf-bar-container { background: #1e2d4a; border-radius: 99px; height: 10px; overflow: hidden; margin-top: 4px;}
.conf-bar { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%); transition: width 0.5s; }

/* Trend badge */
.badge-trend    { display:inline-block; padding:2px 10px; border-radius:99px; font-size:0.72rem; font-weight:600; background:#1d4ed8; color:#bfdbfe; }
.badge-nottrend { display:inline-block; padding:2px 10px; border-radius:99px; font-size:0.72rem; font-weight:600; background:#374151; color:#9ca3af; }
.badge-strong   { background:#7c3aed; color:#ddd6fe; }

/* Stmetric styling */
[data-testid="stMetric"] {
    background: #0d1f3c;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 0.8rem 1rem;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def fetch_live_candles(symbol: str, interval: str = "5m", limit: int = 200) -> pd.DataFrame:
    """Fetch latest candles from Binance REST (cached 60s)."""
    try:
        url    = f"{BINANCE_REST_BASE}/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        resp   = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        df = pd.DataFrame(resp.json(), columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_volume","num_trades",
            "taker_buy_base_vol","taker_buy_quote_vol","ignore",
        ])
        for col in ["open","high","low","close","volume"]:
            df[col] = pd.to_numeric(df[col])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        return df
    except Exception as e:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def fetch_24h_ticker(symbol: str) -> dict:
    try:
        url  = f"{BINANCE_REST_BASE}/api/v3/ticker/24hr"
        resp = requests.get(url, params={"symbol": symbol}, timeout=8)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


def load_error_log(symbol: str | None = None) -> pd.DataFrame:
    if not ERROR_LOG_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(ERROR_LOG_PATH)
    if symbol:
        df = df[df["symbol"] == symbol]
    df["logged_at"] = pd.to_datetime(df["logged_at"])
    return df.sort_values("logged_at", ascending=False).reset_index(drop=True)


def load_model_meta(symbol: str) -> dict:
    """Return model file metadata (size, modified time)."""
    p = MODEL_DIR / f"{symbol}_latest.pkl"
    if p.exists():
        mtime = datetime.fromtimestamp(p.stat().st_mtime)
        size  = p.stat().st_size / 1_048_576  # MB
        return {"exists": True, "last_trained": mtime, "size_mb": size}
    return {"exists": False, "last_trained": None, "size_mb": 0}


def run_prediction(symbol: str):
    """Run the live predictor (import lazily to avoid slow startup)."""
    from predict.predictor import predict_symbol
    return predict_symbol(symbol, update_data=False)


# ── Sidebar ───────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0 0.5rem;'>
        <div style='font-size:2.5rem;'>🤖</div>
        <div style='font-size:1.1rem; font-weight:700; color:#60a5fa;'>CryptoPredictBot</div>
        <div style='font-size:0.7rem; color:#475569; margin-top:2px;'>AI-Powered · Trend-Aware · Self-Learning</div>
    </div>
    <hr style='border-color:#1e2d4a; margin:0.8rem 0;'/>
    """, unsafe_allow_html=True)

    try:
        universe = build_universe()
        coin_opts = [c["coin"] for c in universe]
    except Exception:
        coin_opts = POPULAR_COINS

    default_idx = coin_opts.index(DASHBOARD_DEFAULT_COIN) if DASHBOARD_DEFAULT_COIN in coin_opts else 0
    selected_coin = st.selectbox("📊 Select Coin", coin_opts, index=default_idx)
    selected_symbol = f"{selected_coin}USDT"

    st.markdown("<div class='section-header'>Chart Settings</div>", unsafe_allow_html=True)
    chart_tf      = st.selectbox("Timeframe", ["5m", "15m", "1h", "4h"], index=0)
    chart_candles = st.slider("Candles to show", 50, 500, 150, step=50)

    st.markdown("<div class='section-header'>Auto-Refresh</div>", unsafe_allow_html=True)
    auto_refresh  = st.checkbox("Auto Refresh (30s)", value=False)

    st.markdown("<hr style='border-color:#1e2d4a;'/>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        predict_btn = st.button("🎯 Predict", use_container_width=True, type="primary")
    with col_b:
        retrain_btn = st.button("🔁 Retrain", use_container_width=True)

    st.markdown("<hr style='border-color:#1e2d4a;'/>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.65rem; color:#334155; text-align:center; line-height:1.5;'>
        Data: Binance Public API<br>
        Model: LightGBM+XGBoost+CatBoost<br>
        Validation: Walk-Forward (52 folds)<br>
        Signals: Trend-gated (ADX > 25)
    </div>
    """, unsafe_allow_html=True)


# ── Auto-refresh ──────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(30)
    st.rerun()


# ── Main Layout ───────────────────────────────────────────────────────

# Title bar
ticker_data  = fetch_24h_ticker(selected_symbol)
price_now    = float(ticker_data.get("lastPrice", 0))
price_change = float(ticker_data.get("priceChangePercent", 0))
price_color  = "#10b981" if price_change >= 0 else "#ef4444"
price_arrow  = "▲" if price_change >= 0 else "▼"

st.markdown(f"""
<div style='display:flex; align-items:center; gap:1.5rem; padding:0.5rem 0 1rem; border-bottom:1px solid #1e2d4a; margin-bottom:1rem;'>
    <div>
        <div style='font-size:1.8rem; font-weight:800; color:#f1f5f9;'>{selected_coin}/USDT</div>
        <div style='font-size:0.72rem; color:#64748b;'>via Binance · {datetime.now(timezone.utc).strftime("%H:%M UTC")}</div>
    </div>
    <div style='margin-left:auto; text-align:right;'>
        <div style='font-size:2rem; font-weight:700; color:#f1f5f9;'>${price_now:,.4f}</div>
        <div style='font-size:1rem; color:{price_color}; font-weight:600;'>{price_arrow} {abs(price_change):.2f}% (24h)</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Row 1: Stats ──────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

with c1:
    vol_24h = float(ticker_data.get("volume", 0))
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>24h Volume</div>
        <div class='metric-value'>{vol_24h:,.0f}</div>
        <div class='metric-sub'>{selected_coin}</div>
    </div>""", unsafe_allow_html=True)

with c2:
    high_24h = float(ticker_data.get("highPrice", 0))
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>24h High</div>
        <div class='metric-value'>${high_24h:,.4f}</div>
        <div class='metric-sub'>USD</div>
    </div>""", unsafe_allow_html=True)

with c3:
    low_24h = float(ticker_data.get("lowPrice", 0))
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>24h Low</div>
        <div class='metric-value'>${low_24h:,.4f}</div>
        <div class='metric-sub'>USD</div>
    </div>""", unsafe_allow_html=True)

with c4:
    acc_7d = get_rolling_accuracy(selected_symbol, days=7)
    acc_str = f"{acc_7d:.1%}" if acc_7d is not None else "N/A"
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>7d Accuracy</div>
        <div class='metric-value'>{acc_str}</div>
        <div class='metric-sub'>Rolling directional</div>
    </div>""", unsafe_allow_html=True)

st.write("")

# ── Row 2: Chart + Prediction Panel ──────────────────────────────────
chart_col, pred_col = st.columns([3, 1], gap="medium")

with chart_col:
    st.markdown("<div class='section-header'>📈 Live Price Chart</div>", unsafe_allow_html=True)
    df_candles = fetch_live_candles(selected_symbol, chart_tf, chart_candles)

    if not df_candles.empty:
        fig = go.Figure()

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df_candles["open_time"],
            open=df_candles["open"],
            high=df_candles["high"],
            low=df_candles["low"],
            close=df_candles["close"],
            name="OHLC",
            increasing_fillcolor="#10b981",
            increasing_line_color="#10b981",
            decreasing_fillcolor="#ef4444",
            decreasing_line_color="#ef4444",
        ))

        # EMA 21
        ema21 = df_candles["close"].ewm(span=21, adjust=False).mean()
        fig.add_trace(go.Scatter(
            x=df_candles["open_time"], y=ema21,
            mode="lines", name="EMA 21",
            line=dict(color="#3b82f6", width=1.5, dash="dot"),
        ))

        # Volume bars (secondary y)
        colors = ["#10b981" if c >= o else "#ef4444"
                  for c, o in zip(df_candles["close"], df_candles["open"])]
        fig.add_trace(go.Bar(
            x=df_candles["open_time"], y=df_candles["volume"],
            name="Volume", marker_color=colors,
            opacity=0.35, yaxis="y2",
        ))

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#94a3b8", size=11),
            margin=dict(l=0, r=0, t=10, b=0),
            height=420,
            showlegend=True,
            legend=dict(
                orientation="h", x=0, y=1.05,
                bgcolor="rgba(0,0,0,0)", font=dict(size=10),
            ),
            xaxis=dict(
                gridcolor="#1e2d4a", zeroline=False,
                showspikes=True, spikecolor="#3b82f6",
                rangeslider=dict(visible=False),
            ),
            yaxis=dict(gridcolor="#1e2d4a", zeroline=False, side="right"),
            yaxis2=dict(overlaying="y", side="left", showgrid=False,
                        range=[0, df_candles["volume"].max() * 5], visible=False),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Chart data unavailable (Binance API may be rate-limited).")


with pred_col:
    st.markdown("<div class='section-header'>🎯 Prediction</div>", unsafe_allow_html=True)

    signal = st.session_state.get(f"signal_{selected_symbol}")

    if predict_btn:
        with st.spinner("Generating signals …"):
            try:
                signal = run_prediction(selected_symbol)
                st.session_state[f"signal_{selected_symbol}"] = signal
            except Exception as e:
                st.error(f"Prediction error: {e}")
                signal = None

    if signal is not None:
        dir_class = "signal-up" if signal.direction == "UP" else "signal-down"
        dir_emoji = "🟢 ▲ UP"  if signal.direction == "UP" else "🔴 ▼ DOWN"
        conf_pct  = int(signal.confidence * 100)
        adx_badge = "badge-strong" if signal.is_strong_trend else "badge-trend"
        adx_label = "STRONG TREND" if signal.is_strong_trend else "TRENDING"

        st.markdown(f"""
        <div class='{dir_class}' style='margin-bottom:0.8rem;'>
            <div style='font-size:0.7rem; color:#9ca3af; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:4px;'>Next 5-min Candle</div>
            <div style='font-size:1.6rem; font-weight:800;'>{dir_emoji}</div>
            <div style='margin-top:8px;'>
                <div style='font-size:0.65rem; color:#9ca3af; margin-bottom:2px;'>Confidence</div>
                <div style='font-size:1.1rem; font-weight:700;'>{conf_pct}%</div>
                <div class='conf-bar-container'><div class='conf-bar' style='width:{conf_pct}%;'></div></div>
            </div>
        </div>
        <div style='background:#0d1f3c; border:1px solid #1e3a5f; border-radius:12px; padding:0.9rem; margin-bottom:0.7rem;'>
            <div style='font-size:0.65rem; color:#64748b; text-transform:uppercase; letter-spacing:0.07em;'>Next 1-hr Price Target</div>
            <div style='font-size:1.4rem; font-weight:700; color:#f1f5f9; margin-top:4px;'>${signal.predicted_price:,.4f}</div>
            <div style='font-size:0.7rem; color:#6b7280;'>±{signal.price_band_pct:.2%} band</div>
        </div>
        <div style='background:#0d1f3c; border:1px solid #1e3a5f; border-radius:12px; padding:0.9rem;'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <div>
                    <div style='font-size:0.65rem; color:#64748b; text-transform:uppercase; letter-spacing:0.07em;'>ADX (Trend)</div>
                    <div style='font-size:1.1rem; font-weight:700; color:#f1f5f9;'>{signal.adx:.1f}</div>
                </div>
                <span class='badge-trend {adx_badge}'>{adx_label}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif signal is False or signal is None and predict_btn:
        st.markdown("""
        <div class='signal-none' style='text-align:center; padding:1.5rem;'>
            <div style='font-size:2rem;'>🔇</div>
            <div style='font-size:0.9rem; font-weight:600; color:#9ca3af; margin-top:8px;'>SILENT</div>
            <div style='font-size:0.72rem; color:#6b7280; margin-top:4px;'>Market not trending or<br>confidence too low (ADX < 25)</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align:center; padding:2rem 1rem; color:#475569;'>
            <div style='font-size:2.5rem;'>🎯</div>
            <div style='margin-top:8px; font-size:0.85rem;'>Click <strong>Predict</strong> to generate a signal</div>
        </div>
        """, unsafe_allow_html=True)


# ── Re-train handler ──────────────────────────────────────────────────
if retrain_btn:
    with st.spinner(f"Retraining model for {selected_coin} (this may take a few minutes) …"):
        from models.online import retrain_symbol_sliding
        ok = retrain_symbol_sliding(selected_symbol)
        if ok:
            st.success(f"✅ Model for {selected_coin} retrained successfully!")
        else:
            st.error("Retrain failed. Check logs.")


# ── Row 3: Accuracy + Model Health ───────────────────────────────────
st.divider()
acc_col, health_col = st.columns([2, 1], gap="medium")

with acc_col:
    st.markdown("<div class='section-header'>📊 Accuracy Tracker</div>", unsafe_allow_html=True)
    err_df = load_error_log(selected_symbol)

    if not err_df.empty:
        # Rolling accuracy chart
        err_df_sorted = err_df.sort_values("logged_at")
        err_df_sorted["rolling_acc_100"] = err_df_sorted["correct"].rolling(100, min_periods=10).mean()
        err_df_sorted["is_trending"] = err_df_sorted["adx"] > ADX_TREND_THRESHOLD

        # All-time stats
        acc_all  = err_df["correct"].mean()
        acc_7d   = get_rolling_accuracy(selected_symbol, 7)  or 0
        acc_30d  = get_rolling_accuracy(selected_symbol, 30) or 0
        trend_df = err_df[err_df["adx"] > ADX_TREND_THRESHOLD]
        acc_trend= trend_df["correct"].mean() if not trend_df.empty else 0

        # Bar chart
        cats = ["All-time", "30-day", "7-day", "Trending only"]
        vals = [acc_all, acc_30d, acc_7d, acc_trend]
        bar_colors = ["#3b82f6" if v >= 0.7 else "#f59e0b" if v >= 0.55 else "#ef4444" for v in vals]
        fig2 = go.Figure(go.Bar(
            x=cats, y=[v * 100 for v in vals],
            marker_color=bar_colors,
            text=[f"{v:.1%}" for v in vals], textposition="outside",
        ))
        fig2.add_hline(y=80, line_color="#8b5cf6", line_dash="dot",
                       annotation_text="80% Target", annotation_position="right")
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#94a3b8", size=11),
            margin=dict(l=0, r=0, t=30, b=0), height=240,
            yaxis=dict(range=[0, 105], gridcolor="#1e2d4a", ticksuffix="%"),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Rolling line chart
        if len(err_df_sorted) > 20:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=err_df_sorted["logged_at"],
                y=err_df_sorted["rolling_acc_100"] * 100,
                mode="lines", name="Rolling Acc (100)",
                fill="tozeroy",
                line=dict(color="#3b82f6", width=2),
                fillcolor="rgba(59,130,246,0.1)",
            ))
            fig3.add_hline(y=80, line_color="#8b5cf6", line_dash="dot")
            fig3.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#94a3b8", size=11),
                margin=dict(l=0, r=0, t=10, b=0), height=180,
                yaxis=dict(range=[0, 105], gridcolor="#1e2d4a",
                           ticksuffix="%", title="Accuracy"),
                showlegend=False,
            )
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No prediction history yet. Run predictions to start tracking accuracy.")


with health_col:
    st.markdown("<div class='section-header'>🏥 Model Health</div>", unsafe_allow_html=True)
    meta = load_model_meta(selected_symbol)

    if meta["exists"]:
        last_trained = meta["last_trained"].strftime("%Y-%m-%d %H:%M")
        size_str     = f"{meta['size_mb']:.1f} MB"
        st.markdown(f"""
        <div style='background:#0d1f3c; border:1px solid #1e3a5f; border-radius:12px; padding:1rem;'>
            <div style='font-size:0.65rem; color:#64748b; text-transform:uppercase; letter-spacing:0.07em;'>Status</div>
            <div style='font-size:1rem; font-weight:700; color:#10b981; margin-bottom:0.8rem;'>✅ Model Loaded</div>

            <div style='font-size:0.65rem; color:#64748b; text-transform:uppercase; letter-spacing:0.07em;'>Last Trained</div>
            <div style='font-size:0.85rem; color:#e2e8f0; margin-bottom:0.8rem;'>{last_trained}</div>

            <div style='font-size:0.65rem; color:#64748b; text-transform:uppercase; letter-spacing:0.07em;'>Model Size</div>
            <div style='font-size:0.85rem; color:#e2e8f0; margin-bottom:0.8rem;'>{size_str}</div>

            <div style='font-size:0.65rem; color:#64748b; text-transform:uppercase; letter-spacing:0.07em;'>Architecture</div>
            <div style='font-size:0.75rem; color:#60a5fa;'>LightGBM + XGBoost<br>+ CatBoost + Meta-LR</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background:#1c1c2e; border:1px solid #4b5563; border-radius:12px; padding:1rem;'>
            <div style='font-size:1rem; font-weight:600; color:#f59e0b;'>⚠️ No Model</div>
            <div style='font-size:0.78rem; color:#6b7280; margin-top:6px;'>
                Train a model first:<br>
                <code style='background:#0d1f3c; padding:2px 6px; border-radius:4px; font-size:0.7rem;'>
                python models/trainer.py --symbol {selected_symbol}
                </code>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ADX Indicator
    adx_val = None
    if not (df_candles := fetch_live_candles(selected_symbol, "5m", 50)).empty:
        from features.technical import _adx
        adx_series, _, _ = _adx(df_candles, 14)
        adx_val = float(adx_series.iloc[-1]) if not adx_series.empty else None

    if adx_val is not None:
        trend_status = ("🔥 Strong" if adx_val > ADX_STRONG_TREND
                        else "📈 Trending" if adx_val > ADX_TREND_THRESHOLD
                        else "😴 Ranging")
        trend_color  = ("#7c3aed" if adx_val > ADX_STRONG_TREND
                        else "#10b981" if adx_val > ADX_TREND_THRESHOLD
                        else "#6b7280")
        st.markdown(f"""
        <div style='background:#0d1f3c; border:1px solid {trend_color}; border-radius:12px; padding:1rem; margin-top:0.8rem; text-align:center;'>
            <div style='font-size:0.65rem; color:#64748b; text-transform:uppercase; letter-spacing:0.07em;'>Current ADX (14)</div>
            <div style='font-size:2rem; font-weight:800; color:{trend_color};'>{adx_val:.1f}</div>
            <div style='font-size:0.85rem; color:{trend_color}; font-weight:600;'>{trend_status}</div>
            <div style='font-size:0.65rem; color:#475569; margin-top:4px;'>Signals active when ADX > {ADX_TREND_THRESHOLD}</div>
        </div>
        """, unsafe_allow_html=True)


# ── Row 4: Recent Predictions ─────────────────────────────────────────
st.divider()
st.markdown("<div class='section-header'>📋 Recent Prediction Log</div>", unsafe_allow_html=True)
err_df_all = load_error_log(selected_symbol)

if not err_df_all.empty:
    display_df = err_df_all.head(20).copy()
    display_df["Result"]     = display_df["correct"].map({1: "✅ Correct", 0: "❌ Wrong"})
    display_df["Direction"]  = display_df["predicted_dir"].map({1: "UP ▲", 0: "DOWN ▼"})
    display_df["Confidence"] = (display_df["confidence"] * 100).round(1).astype(str) + "%"
    display_df["ADX"]        = display_df["adx"].round(1)
    display_df["Time"]       = display_df["logged_at"].dt.strftime("%m-%d %H:%M")
    display_df["Pred Price"] = display_df["predicted_price"].round(4)
    display_df["Actual Price"] = display_df["actual_price"].round(4)

    st.dataframe(
        display_df[["Time","Direction","Result","Confidence","ADX","Pred Price","Actual Price"]],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No prediction history yet.")


# ── Footer ────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#1e2d4a; margin-top:2rem;'/>
<div style='text-align:center; font-size:0.65rem; color:#334155; padding:0.5rem;'>
    CryptoPredictBot · Data from Binance Public API · Not financial advice ·
    Walk-Forward Validated · ADX Trend-Gated Signals
</div>
""", unsafe_allow_html=True)
