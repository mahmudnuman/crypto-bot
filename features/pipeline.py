"""
features/pipeline.py — sklearn-compatible feature pipeline.

Responsibilities:
  1. RobustScaler (outlier-resistant normalisation)
  2. VarianceThreshold (drop zero-variance features)
  3. SimpleImputer (fill any remaining NaN with median)
  4. SelectFromModel (LightGBM-based feature selection — top-k important)

Critical: all fit() calls use ONLY training data, never test data.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from lightgbm import LGBMClassifier
from loguru import logger


# Feature columns that are NOT used as model inputs
NON_FEATURE_COLS = {"target_dir", "target_price"}


def _to_numpy(X):
    """Force any input to numpy float32 — prevents feature-name warnings."""
    if hasattr(X, "values"):
        return X.values.astype(np.float32)
    return np.array(X, dtype=np.float32)


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return all feature columns (exclude targets, keep NaN-safe)."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def build_classifier_pipeline() -> Pipeline:
    """
    Pipeline for Head A (binary classification).
    Step 0 converts DataFrame → numpy so SelectFromModel never sees column names.
    """
    selector = SelectFromModel(
        LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        ),
        threshold="mean",
        max_features=80,
    )
    pipeline = Pipeline([
        ("to_numpy",  FunctionTransformer(_to_numpy, validate=False)),
        ("imputer",   SimpleImputer(strategy="median")),
        ("scaler",    RobustScaler()),
        ("var_filter",VarianceThreshold(threshold=1e-5)),
        ("selector",  selector),
    ])
    return pipeline


def build_regressor_pipeline() -> Pipeline:
    """
    Pipeline for Head B (1h price regression).
    Step 0 converts DataFrame → numpy so SelectFromModel never sees column names.
    """
    from sklearn.feature_selection import SelectFromModel
    from lightgbm import LGBMRegressor
    selector = SelectFromModel(
        LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        ),
        threshold="mean",
        max_features=80,
    )
    pipeline = Pipeline([
        ("to_numpy",  FunctionTransformer(_to_numpy, validate=False)),
        ("imputer",   SimpleImputer(strategy="median")),
        ("scaler",    RobustScaler()),
        ("var_filter",VarianceThreshold(threshold=1e-5)),
        ("selector",  selector),
    ])
    return pipeline


def prepare_Xy_classifier(df: pd.DataFrame, feature_cols: list[str]):
    """Extract X, y for classification head. Drops rows with NaN target."""
    df_clean = df.dropna(subset=["target_dir"])
    X = df_clean[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = df_clean["target_dir"].astype(int)
    return X, y


def prepare_Xy_regressor(df: pd.DataFrame, feature_cols: list[str]):
    """Extract X, y for regression head. Drops rows with NaN target_price."""
    df_clean = df.dropna(subset=["target_price"])
    X = df_clean[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = df_clean["target_price"].astype(float)
    return X, y
