"""
tests/test_predictor.py — End-to-end smoke test for the prediction pipeline.
Uses synthetic data so no real Binance connection needed.
"""
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from predict.confidence import compute_signal, signals_to_dataframe
from models.validator import analyse_fold, build_report


class TestConfidenceGating(unittest.TestCase):

    def _make_signal(self, adx=30, proba_up=0.80, base_probas=None):
        if base_probas is None:
            base_probas = np.array([0.75, 0.80, 0.78])
        return compute_signal(
            symbol="BTCUSDT",
            proba_up=proba_up,
            base_probas=base_probas,
            predicted_price=50000.0,
            current_price=49800.0,
            adx=adx,
            atr_pct=0.015,
        )

    def test_signal_emitted_when_all_gates_pass(self):
        sig = self._make_signal(adx=30, proba_up=0.82)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.direction, "UP")
        self.assertGreaterEqual(sig.confidence, 0.65)

    def test_silent_when_adx_below_threshold(self):
        sig = self._make_signal(adx=20)
        self.assertIsNone(sig, "Should be SILENT when ADX < 25")

    def test_silent_when_low_confidence(self):
        sig = self._make_signal(adx=30, proba_up=0.52)  # near 50/50
        self.assertIsNone(sig, "Should be SILENT when confidence < 0.65")

    def test_silent_when_models_disagree(self):
        sig = self._make_signal(
            adx=30,
            proba_up=0.80,
            base_probas=np.array([0.75, 0.40, 0.35]),  # only 1/3 agree UP
        )
        self.assertIsNone(sig, "Should be SILENT when <2 models agree")

    def test_strong_trend_boosts_confidence(self):
        sig_normal = self._make_signal(adx=28)
        sig_strong = self._make_signal(adx=40)
        if sig_normal and sig_strong:
            self.assertGreaterEqual(sig_strong.confidence, sig_normal.confidence)
        self.assertIsNotNone(sig_strong)
        if sig_strong:
            self.assertTrue(sig_strong.is_strong_trend)

    def test_signals_to_dataframe(self):
        sigs = [self._make_signal(adx=32), None, self._make_signal(adx=28)]
        df   = signals_to_dataframe(sigs)
        # Should have 2 rows (one None dropped)
        self.assertEqual(len(df), 2)
        self.assertIn("direction", df.columns)
        self.assertIn("confidence", df.columns)


class TestValidator(unittest.TestCase):

    def test_overfit_detection(self):
        from models.validator import FoldResult
        fold = analyse_fold(
            fold_idx=0,
            train_acc=0.92,
            test_acc=0.55,   # 37% gap → overfit
            train_f1=0.91,
            test_f1=0.53,
            test_mape=0.05,
            n_train=5000,
            n_test=500,
        )
        report = build_report("TEST", [fold])
        self.assertIn(0, report.overfit_folds)
        self.assertTrue(any("regularization" in a.lower() for a in report.recommended_actions))

    def test_underfit_detection(self):
        fold = analyse_fold(
            fold_idx=0,
            train_acc=0.50,
            test_acc=0.49,   # below 52% threshold
            train_f1=0.49,
            test_f1=0.48,
            test_mape=0.10,
            n_train=5000,
            n_test=500,
        )
        report = build_report("TEST", [fold])
        self.assertIn(0, report.underfit_folds)
        self.assertTrue(any("feature" in a.lower() for a in report.recommended_actions))

    def test_healthy_model_no_flags(self):
        fold = analyse_fold(
            fold_idx=0,
            train_acc=0.78,
            test_acc=0.73,   # 5% gap → healthy
            train_f1=0.77,
            test_f1=0.72,
            test_mape=0.02,
            n_train=5000,
            n_test=500,
        )
        report = build_report("TEST", [fold])
        self.assertEqual(report.overfit_folds, [])
        self.assertEqual(report.underfit_folds, [])


if __name__ == "__main__":
    unittest.main()
