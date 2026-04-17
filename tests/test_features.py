"""
tests/test_features.py — Verify no look-ahead bias and correct feature shapes.
"""
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_ohlcv(n: int = 500) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame."""
    np.random.seed(42)
    close  = 100 + np.cumsum(np.random.randn(n) * 0.5)
    opens  = close * (1 + np.random.randn(n) * 0.002)
    highs  = np.maximum(close, opens) * (1 + np.random.uniform(0, 0.01, n))
    lows   = np.minimum(close, opens) * (1 - np.random.uniform(0, 0.01, n))
    vols   = np.random.uniform(1000, 5000, n)
    ts     = pd.date_range("2024-01-01", periods=n, freq="5min")
    return pd.DataFrame({
        "open_time":  ts,
        "open":       opens,
        "high":       highs,
        "low":        lows,
        "close":      close,
        "volume":     vols,
        "close_time": ts + pd.Timedelta(minutes=4, seconds=59),
        "quote_volume":        vols * close,
        "num_trades":          np.random.randint(50, 500, n),
        "taker_buy_base_vol":  vols * 0.5,
        "taker_buy_quote_vol": vols * close * 0.5,
        "ignore":              ["0"] * n,
    })


class TestTechnicalFeatures(unittest.TestCase):

    def setUp(self):
        self.df = _make_ohlcv(500)

    def test_no_future_leakage_lag_features(self):
        """Lag features at row i must depend only on rows < i."""
        from features.technical import add_all_features
        df_feat = add_all_features(self.df.copy(), tf_label="")
        # lag_ret1 at row 5 should equal returns at row 4
        ret      = df_feat["returns"]
        lag_ret1 = df_feat["lag_ret1"]
        # First valid row is at index 2 (need 1 return + 1 lag)
        for idx in range(2, 10):
            self.assertAlmostEqual(lag_ret1.iloc[idx], ret.iloc[idx - 1], places=10,
                                   msg=f"Lag-1 feature at row {idx} has look-ahead bias!")

    def test_feature_count(self):
        """Must generate at least 60 feature columns."""
        from features.technical import add_all_features
        original_cols = set(self.df.columns)
        df_feat = add_all_features(self.df.copy(), tf_label="")
        new_cols = set(df_feat.columns) - original_cols
        self.assertGreaterEqual(len(new_cols), 60,
            f"Expected ≥60 new features, got {len(new_cols)}")

    def test_no_all_nan_columns(self):
        """Every feature should have at least 50% non-NaN values."""
        from features.technical import add_all_features
        df_feat = add_all_features(self.df.copy(), tf_label="")
        for col in df_feat.columns:
            nan_frac = df_feat[col].isna().mean()
            self.assertLess(nan_frac, 0.5,
                f"Column '{col}' has {nan_frac:.0%} NaN values (too many)")

    def test_adx_range(self):
        """ADX must be between 0 and 100."""
        from features.technical import add_all_features
        df_feat = add_all_features(self.df.copy(), tf_label="")
        adx = df_feat["adx14"].dropna()
        self.assertTrue((adx >= 0).all() and (adx <= 100).all(),
                        "ADX out of [0, 100] range")


class TestPipeline(unittest.TestCase):

    def test_pipeline_no_data_leak(self):
        """Pipeline must be fit on train data only, not test data."""
        from features.technical import add_all_features
        from features.pipeline import (
            build_classifier_pipeline, get_feature_cols,
            prepare_Xy_classifier, NON_FEATURE_COLS
        )

        df = _make_ohlcv(300)
        df = add_all_features(df, tf_label="")
        df.set_index("open_time", inplace=True)
        df["target_dir"] = (df["close"].shift(-1) > df["close"]).astype(int)
        df.drop(columns=["close","open","high","low","volume","close_time",
                          "quote_volume","num_trades","taker_buy_base_vol",
                          "taker_buy_quote_vol","ignore"], inplace=True, errors="ignore")

        train, test = df.iloc[:200], df.iloc[200:]
        feat_cols   = get_feature_cols(df)
        X_train, y_train = prepare_Xy_classifier(train, feat_cols)
        X_test,  y_test  = prepare_Xy_classifier(test,  feat_cols)

        pipe = build_classifier_pipeline()
        pipe.fit(X_train, y_train)        # fit only on train
        X_test_t = pipe.transform(X_test) # transform test (must not refit)
        self.assertGreater(X_test_t.shape[1], 0)


if __name__ == "__main__":
    unittest.main()
