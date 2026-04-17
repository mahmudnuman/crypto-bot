"""
config.py — Central configuration for the Crypto Prediction Bot.
All tuneable parameters live here. No magic numbers elsewhere.
"""
from pathlib import Path

# ── Project Paths ────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
CACHE_DIR  = BASE_DIR / "cache"
MODEL_DIR  = BASE_DIR / "saved_models"
LOG_DIR    = BASE_DIR / "logs"

for _d in [CACHE_DIR, MODEL_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Universe ─────────────────────────────────────────────────────────
# Top-25 most liquid coins with consistent Binance USDT pairs
POPULAR_COINS = [
    "BTC", "ETH", "BNB", "SOL", "XRP",
    "ADA", "AVAX", "DOGE", "MATIC", "DOT",
    "LINK", "LTC", "UNI", "ATOM", "ETC",
    "NEAR", "ARB", "OP", "APT", "SUI",
    "INJ", "FIL", "XLM", "ALGO", "TIA",
]
QUOTE_ASSET = "USDT"

# ── Timeframes ────────────────────────────────────────────────────────
TIMEFRAMES = {
    "5m":  {"interval": "5m",  "pandas_freq": "5min", "limit": 1000},
    "1h":  {"interval": "1h",  "pandas_freq": "1h",   "limit": 1000},
    "6h":  {"interval": "6h",  "pandas_freq": "6h",   "limit": 1000},
    "1d":  {"interval": "1d",  "pandas_freq": "1D",   "limit": 1000},
}

# ── History Window ────────────────────────────────────────────────────
HISTORY_YEARS = 5  # years back from today

# Binance Bulk Data (data.binance.vision) — only for 5m timeframe
BINANCE_BULK_BASE = "https://data.binance.vision/data/spot/monthly/klines"

# Binance REST API
BINANCE_REST_BASE = "https://api.binance.com"
REST_RATE_LIMIT_PAUSE = 0.25      # seconds between calls (safe side)
REST_MAX_RETRIES      = 5
REST_BACKOFF_BASE     = 2.0       # exponential backoff multiplier

# ── Feature Engineering ───────────────────────────────────────────────
LOOKBACK_PERIODS = [5, 12, 24, 48, 72, 144]   # for rolling stats
LAG_PERIODS      = list(range(1, 21))          # lag feature shifts
EMA_PERIODS      = [8, 21, 55, 200]

# ── Trend Filter ──────────────────────────────────────────────────────
ADX_PERIOD          = 14
ADX_TREND_THRESHOLD = 25   # ADX > 25 → trending market → predictions ON
ADX_STRONG_TREND    = 35   # ADX > 35 → strong trend → confidence +boost

# ── Prediction Targets ────────────────────────────────────────────────
# HEAD A: direction of next N 5-min candles
DIRECTION_LOOKAHEAD_5M = 1    # predict next single 5-min candle UP/DOWN

# HEAD B: 1-hr close price
PRICE_LOOKAHEAD_1H     = 1    # predict close of next 1-hr candle (regression)

# ── Walk-Forward Validation ───────────────────────────────────────────
WFV_INITIAL_TRAIN_YEARS = 3.5   # first train on 3.5 years
WFV_GAP_DAYS            = 14    # embargo gap to prevent leakage
WFV_TEST_DAYS           = 14    # each test window
WFV_STEP_DAYS           = 60    # step 60d → ~9 folds per coin (was 30d → ~18 folds)


# ── Model Hyperparameters ─────────────────────────────────────────────
# GPU: XGBoost + CatBoost use CUDA. LightGBM uses CPU (needs GPU-build).
LGBM_PARAMS = dict(
    n_estimators     = 200,      # reduced: LGBM is CPU-only, keep it fast
    learning_rate    = 0.05,
    max_depth        = 5,        # shallower = faster
    num_leaves       = 31,
    min_child_samples= 40,
    subsample        = 0.8,
    colsample_bytree = 0.7,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    random_state     = 42,
    n_jobs           = -1,       # all CPU cores
    verbose          = -1,
)

XGBM_PARAMS = dict(
    n_estimators     = 400,
    learning_rate    = 0.05,
    max_depth        = 5,
    min_child_weight = 5,
    subsample        = 0.8,
    colsample_bytree = 0.7,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    random_state     = 42,
    tree_method      = "hist",
    device           = "cuda",
    verbosity        = 0,
    eval_metric      = "logloss",
)

CATBOOST_PARAMS = dict(
    iterations        = 400,
    learning_rate     = 0.05,
    depth             = 6,
    l2_leaf_reg       = 3,
    border_count      = 128,
    random_seed       = 42,
    task_type         = "GPU",
    devices           = "0",
    verbose           = 0,
)

EARLY_STOPPING_ROUNDS = 50

# ── Overfitting / Underfitting Guards ─────────────────────────────────
OVERFIT_GAP_THRESHOLD    = 0.15   # if train_acc - test_acc > 15% → overfit
UNDERFIT_MIN_ACCURACY    = 0.52   # if test_acc < 52% → underfit
TOP_FEATURE_GINI_LIMIT   = 0.40   # if top feature importance > 40% → drop
WFV_STD_THRESHOLD        = 0.10   # std of fold accuracy > 10% → flag

# ── Confidence Gate ───────────────────────────────────────────────────
MIN_CONFIDENCE           = 0.65   # only emit signal if prob ≥ 65%
MIN_ENSEMBLE_AGREEMENT   = 2      # at least 2/3 base models must agree (direction)

# ── Online Re-learning ────────────────────────────────────────────────
ONLINE_RETRAIN_INTERVAL_HOURS  = 4    # check every 4 hours
ONLINE_RETRAIN_TRIGGER_ACC     = 0.60 # retrain if rolling 7d accuracy < 60%
ONLINE_SLIDING_WINDOW_DAYS     = 180  # training window for incremental retrain

# ── Dashboard ─────────────────────────────────────────────────────────
DASHBOARD_REFRESH_SECONDS = 30
DASHBOARD_DEFAULT_COIN    = "BTC"
