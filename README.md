# 🤖 CryptoPredictBot

An automated multi-timeframe cryptocurrency prediction system using walk-forward validated ensemble learning.

## Features
- **25 coins** tracked (BTC, ETH, BNB, SOL, XRP and 20 more)
- **4 timeframes** merged: 5m + 1h + 6h + 1d → 372 features
- **Dual-head ensemble**: LightGBM + XGBoost (GPU) + CatBoost (GPU)
- **Walk-forward validation** with fold-level checkpoint/resume
- **Live training monitor** dashboard
- **Streamlit dashboard** for predictions

## Structure
```
crypto_bot/
├── config.py              # All settings (coins, timeframes, model params)
├── run_all.py             # Main entry point — trains all 25 coins
├── monitor.py             # Live training progress dashboard
├── keepawake.py           # Prevents PC sleep during training
├── data/
│   ├── downloader.py      # Binance REST API downloader
│   ├── store.py           # Parquet data store
│   └── universe.py        # Coin universe builder
├── features/
│   ├── multi_tf.py        # Multi-timeframe feature engineering
│   ├── technical.py       # Technical indicators (RSI, MACD, BB, etc.)
│   └── pipeline.py        # sklearn preprocessing pipeline
├── models/
│   ├── ensemble.py        # Dual-head LightGBM+XGBoost+CatBoost ensemble
│   ├── trainer.py         # Walk-forward validation loop
│   └── validator.py       # Fold analysis & overfitting detection
├── dashboard/
│   └── app.py             # Streamlit prediction dashboard
└── cache/
    ├── checkpoints/       # Fold-level training checkpoints (resume support)
    └── training_report.json
```

## Quick Start
```bash
# 1. Download all data
python data/downloader.py --all

# 2. Train all 25 coins (resumes from checkpoint on restart)
python run_all.py

# 3. Watch training live (open in new terminal)
python monitor.py

# 4. View predictions dashboard
streamlit run dashboard/app.py
```

## Model Performance
Walk-forward test accuracy typically 51-56% across coins.
Random baseline = 50%, so models show consistent edge.

## Hardware
- Optimized for NVIDIA GPU (XGBoost CUDA + CatBoost GPU)
- Falls back to CPU automatically
- Tested on: i5-8265U + NVIDIA GeForce MX130 (2GB VRAM)
