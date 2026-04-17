"""
models/ensemble.py — Dual-head ensemble model (LightGBM + XGBoost + CatBoost).

HEAD A: Binary classification  — next 5m candle UP (1) or DOWN (0)
HEAD B: Regression             — next 1h close price

Each head is a stacked ensemble:
  Layer 1: LightGBM + XGBoost + CatBoost (base learners)
  Layer 2: Logistic Regression / Ridge meta-learner on OOF predictions

CoinModel wraps both heads for a single coin.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_percentage_error
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    LGBM_PARAMS, XGBM_PARAMS, CATBOOST_PARAMS,
    EARLY_STOPPING_ROUNDS, MODEL_DIR,
)


class DualHeadEnsemble:
    """
    Dual-head stacked ensemble for one coin.

    Attributes
    ----------
    symbol : str
    clf_*  : classifiers for Head A
    reg_*  : regressors  for Head B
    meta_clf, meta_reg : meta-learners
    preproc_clf, preproc_reg : fitted sklearn Pipelines (passed in from trainer)
    """

    def __init__(self, symbol: str):
        self.symbol       = symbol
        # ── Head A (classification) ──────────────────────────────────
        self.clf_lgbm = LGBMClassifier(**LGBM_PARAMS)
        self.clf_xgb  = XGBClassifier(**XGBM_PARAMS)
        self.clf_cat  = CatBoostClassifier(**CATBOOST_PARAMS)
        self.meta_clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        self.preproc_clf = None

        # ── Head B (regression) ───────────────────────────────────────
        reg_lgbm_p = {k: v for k, v in LGBM_PARAMS.items()
                      if k not in ["verbose"]}
        reg_xgb_p  = {k: v for k, v in XGBM_PARAMS.items()
                      if k not in ["eval_metric"]}
        self.reg_lgbm  = LGBMRegressor(**{**reg_lgbm_p, "verbose": -1})
        self.reg_xgb   = XGBRegressor(**{**reg_xgb_p, "eval_metric": "rmse"})
        self.reg_cat   = CatBoostRegressor(**CATBOOST_PARAMS)
        self.meta_reg  = Ridge(alpha=1.0)
        self.preproc_reg = None

        self._fitted = False

    # ── Training ──────────────────────────────────────────────────────

    def fit(
        self,
        X_train_raw: pd.DataFrame,
        y_dir:       pd.Series,
        y_price:     pd.Series,
        X_val_raw:   pd.DataFrame | None = None,
        y_val_dir:   pd.Series | None    = None,
        y_val_price: pd.Series | None    = None,
    ):
        """
        Train both heads with early stopping using a validation split.
        All preprocessing is fit here (training data only).
        """
        from features.pipeline import build_classifier_pipeline, build_regressor_pipeline

        # ── Preprocess ────────────────────────────────────────────────
        # Convert to numpy FIRST to prevent sklearn feature-name warnings
        # (SelectFromModel fitted with named DF, cross_val_predict gives array)
        self.preproc_clf = build_classifier_pipeline()
        self.preproc_reg = build_regressor_pipeline()

        X_arr = X_train_raw.values if hasattr(X_train_raw, "values") else np.array(X_train_raw)
        X_train_c = self.preproc_clf.fit_transform(X_arr, y_dir)
        X_train_r = self.preproc_reg.fit_transform(X_arr, y_price)

        if X_val_raw is not None and (hasattr(X_val_raw, "empty") and not X_val_raw.empty or
                                       not hasattr(X_val_raw, "empty")):
            X_val_arr = X_val_raw.values if hasattr(X_val_raw, "values") else np.array(X_val_raw)
            X_val_c = self.preproc_clf.transform(X_val_arr)
            X_val_r = self.preproc_reg.transform(X_val_arr)
            eval_set_lgbm = [(X_val_c, y_val_dir)]
            eval_set_xgb  = [(X_val_c, y_val_dir)]
        else:
            # 10% internal split if no explicit val provided
            split = int(len(X_train_c) * 0.9)
            X_val_c     = X_train_c[split:]
            y_val_dir   = y_dir.iloc[split:]
            X_val_r     = X_train_r[split:]
            y_val_price = y_price.iloc[split:]
            X_train_c   = X_train_c[:split]
            y_dir       = y_dir.iloc[:split]
            X_train_r   = X_train_r[:split]
            y_price     = y_price.iloc[:split]
            eval_set_lgbm = [(X_val_c, y_val_dir)]
            eval_set_xgb  = [(X_val_c, y_val_dir)]

        # ── Head A — Base learners ────────────────────────────────────
        logger.debug(f"[{self.symbol}] Training Head-A classifiers …")
        self.clf_lgbm.fit(
            X_train_c, y_dir,
            eval_set=eval_set_lgbm,
            callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                       log_evaluation(-1)],
        )
        self.clf_xgb.fit(
            X_train_c, y_dir,
            eval_set=eval_set_xgb,
            verbose=False,
        )
        self.clf_cat.fit(
            X_train_c, y_dir,
            eval_set=(X_val_c, y_val_dir),
        )

        # ── Head A — Meta-learner (OOF stacking) ─────────────────────
        oof_c = self._oof_preds_clf(X_train_c, y_dir)
        self.meta_clf.fit(oof_c, y_dir)

        # ── Head B — Base learners ────────────────────────────────────
        logger.debug(f"[{self.symbol}] Training Head-B regressors …")
        self.reg_lgbm.fit(
            X_train_r, y_price,
            eval_set=[(X_val_r, y_val_price)],
            callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                       log_evaluation(-1)],
        )
        self.reg_xgb.fit(
            X_train_r, y_price,
            eval_set=[(X_val_r, y_val_price)],
            verbose=False,
        )
        self.reg_cat.fit(
            X_train_r, y_price,
            eval_set=(X_val_r, y_val_price),
        )

        # ── Head B — Meta-learner ─────────────────────────────────────
        oof_r = self._oof_preds_reg(X_train_r, y_price)
        self.meta_reg.fit(oof_r, y_price)

        self._fitted = True
        logger.success(f"[{self.symbol}] Ensemble training complete.")

    # ── Prediction ────────────────────────────────────────────────────

    def predict_direction(self, X_raw) -> tuple:
        """Returns (predicted_class, probability_up) for Head A."""
        assert self._fitted, "Call .fit() first."
        X_arr  = X_raw.values if hasattr(X_raw, "values") else np.array(X_raw)
        X_c    = self.preproc_clf.transform(X_arr)
        stack  = self._stack_preds_clf(X_c)
        proba  = self.meta_clf.predict_proba(stack)[:, 1]
        labels = (proba >= 0.5).astype(int)
        return labels, proba

    def predict_price(self, X_raw) -> np.ndarray:
        """Returns predicted next-1h close price for Head B."""
        assert self._fitted, "Call .fit() first."
        X_arr = X_raw.values if hasattr(X_raw, "values") else np.array(X_raw)
        X_r   = self.preproc_reg.transform(X_arr)
        stack = self._stack_preds_reg(X_r)
        return self.meta_reg.predict(stack)

    def base_model_probas(self, X_raw) -> np.ndarray:
        """Returns (n_samples, 3) array of per-base-model UP probabilities."""
        X_arr = X_raw.values if hasattr(X_raw, "values") else np.array(X_raw)
        X_c = self.preproc_clf.transform(X_arr)
        p1  = self.clf_lgbm.predict_proba(X_c)[:, 1]
        p2  = self.clf_xgb.predict_proba(X_c)[:, 1]
        p3  = self.clf_cat.predict_proba(X_c)[:, 1]
        return np.column_stack([p1, p2, p3])

    # ── Evaluate ──────────────────────────────────────────────────────

    def evaluate(
        self, X_raw: pd.DataFrame, y_dir: pd.Series, y_price: pd.Series
    ) -> dict:
        preds_dir, proba = self.predict_direction(X_raw)
        preds_price      = self.predict_price(X_raw)

        y_dir_a   = y_dir.values
        y_price_a = y_price.values

        valid_p = ~np.isnan(y_price_a)
        return {
            "accuracy":    accuracy_score(y_dir_a, preds_dir),
            "f1":          f1_score(y_dir_a, preds_dir, zero_division=0),
            "mape":        mean_absolute_percentage_error(
                               y_price_a[valid_p], preds_price[valid_p]
                           ) if valid_p.any() else np.nan,
            "avg_conf":    float(np.abs(proba - 0.5).mean() * 2),  # 0=random,1=certain
        }

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, tag: str = "latest") -> Path:
        path = MODEL_DIR / f"{self.symbol}_{tag}.pkl"
        joblib.dump(self, path)
        logger.info(f"Model saved → {path}")
        return path

    @classmethod
    def load(cls, symbol: str, tag: str = "latest") -> "DualHeadEnsemble":
        path = MODEL_DIR / f"{symbol}_{tag}.pkl"
        obj  = joblib.load(path)
        logger.info(f"Model loaded ← {path}")
        return obj

    # ── Internal helpers ──────────────────────────────────────────────

    def _oof_preds_clf(self, X, y) -> np.ndarray:
        kf   = KFold(n_splits=5, shuffle=False)
        p1   = cross_val_predict(self.clf_lgbm, X, y, cv=kf, method="predict_proba")[:, 1]
        p2   = cross_val_predict(self.clf_xgb,  X, y, cv=kf, method="predict_proba")[:, 1]
        p3   = cross_val_predict(self.clf_cat,  X, y, cv=kf, method="predict_proba")[:, 1]
        return np.column_stack([p1, p2, p3])

    def _oof_preds_reg(self, X, y) -> np.ndarray:
        kf  = KFold(n_splits=5, shuffle=False)
        p1  = cross_val_predict(self.reg_lgbm, X, y, cv=kf)
        p2  = cross_val_predict(self.reg_xgb,  X, y, cv=kf)
        p3  = cross_val_predict(self.reg_cat,  X, y, cv=kf)
        return np.column_stack([p1, p2, p3])

    def _stack_preds_clf(self, X) -> np.ndarray:
        p1 = self.clf_lgbm.predict_proba(X)[:, 1]
        p2 = self.clf_xgb.predict_proba(X)[:, 1]
        p3 = self.clf_cat.predict_proba(X)[:, 1]
        return np.column_stack([p1, p2, p3])

    def _stack_preds_reg(self, X) -> np.ndarray:
        p1 = self.reg_lgbm.predict(X)
        p2 = self.reg_xgb.predict(X)
        p3 = self.reg_cat.predict(X)
        return np.column_stack([p1, p2, p3])
