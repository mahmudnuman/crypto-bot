"""
models/validator.py — Overfitting / Underfitting detection and guards.

Checks performed after each walk-forward fold:
  1. Train–Test accuracy gap  (overfitting check)
  2. Minimum test accuracy    (underfitting check)
  3. Feature importance Gini  (single-feature dominance check)
  4. Fold consistency         (std across folds)
  5. Sharpe live vs backtest  (regime shift check)

Returns structured report with pass/fail status and recommended actions.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from loguru import logger
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    OVERFIT_GAP_THRESHOLD,
    UNDERFIT_MIN_ACCURACY,
    TOP_FEATURE_GINI_LIMIT,
    WFV_STD_THRESHOLD,
)


@dataclass
class FoldResult:
    fold:          int
    train_acc:     float
    test_acc:      float
    train_f1:      float
    test_f1:       float
    test_mape:     float
    n_train:       int
    n_test:        int
    top_feature:   str  = ""
    top_importance:float = 0.0


@dataclass
class ValidationReport:
    symbol:          str
    fold_results:    list[FoldResult] = field(default_factory=list)
    overfit_folds:   list[int]        = field(default_factory=list)
    underfit_folds:  list[int]        = field(default_factory=list)
    dominant_feature_folds: list[int] = field(default_factory=list)
    mean_test_acc:   float = 0.0
    std_test_acc:    float = 0.0
    is_regime_unstable: bool = False
    recommended_actions: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"=== Validation Report: {self.symbol} ===",
            f"  Folds:          {len(self.fold_results)}",
            f"  Mean test acc:  {self.mean_test_acc:.3f}",
            f"  Std  test acc:  {self.std_test_acc:.3f}",
            f"  Overfit folds:  {self.overfit_folds}",
            f"  Underfit folds: {self.underfit_folds}",
            f"  Regime unstable:{self.is_regime_unstable}",
        ]
        if self.recommended_actions:
            lines.append("  Actions:")
            for act in self.recommended_actions:
                lines.append(f"    • {act}")
        return "\n".join(lines)


def analyse_fold(
    fold_idx:   int,
    train_acc:  float,
    test_acc:   float,
    train_f1:   float,
    test_f1:    float,
    test_mape:  float,
    n_train:    int,
    n_test:     int,
    feature_importances: dict | None = None,
) -> FoldResult:
    """Analyse a single fold. Returns FoldResult."""
    top_feat = ""
    top_imp  = 0.0
    if feature_importances:
        total = sum(feature_importances.values()) + 1e-9
        normed = {k: v / total for k, v in feature_importances.items()}
        top_feat = max(normed, key=normed.get)
        top_imp  = normed[top_feat]

    return FoldResult(
        fold=fold_idx,
        train_acc=train_acc,
        test_acc=test_acc,
        train_f1=train_f1,
        test_f1=test_f1,
        test_mape=test_mape,
        n_train=n_train,
        n_test=n_test,
        top_feature=top_feat,
        top_importance=top_imp,
    )


def build_report(symbol: str, fold_results: list[FoldResult]) -> ValidationReport:
    """Aggregate fold results into a full ValidationReport."""
    report = ValidationReport(symbol=symbol, fold_results=fold_results)
    if not fold_results:
        return report

    accs = [f.test_acc for f in fold_results]
    report.mean_test_acc = float(np.mean(accs))
    report.std_test_acc  = float(np.std(accs))

    actions = []

    for fr in fold_results:
        gap = fr.train_acc - fr.test_acc

        # Overfitting check
        if gap > OVERFIT_GAP_THRESHOLD:
            report.overfit_folds.append(fr.fold)

        # Underfitting check
        if fr.test_acc < UNDERFIT_MIN_ACCURACY:
            report.underfit_folds.append(fr.fold)

        # Feature dominance check
        if fr.top_importance > TOP_FEATURE_GINI_LIMIT:
            report.dominant_feature_folds.append(fr.fold)
            logger.warning(
                f"[{symbol}] Fold {fr.fold}: feature '{fr.top_feature}' "
                f"dominates with {fr.top_importance:.1%} importance → potential leakage!"
            )

    # Regime instability
    if report.std_test_acc > WFV_STD_THRESHOLD:
        report.is_regime_unstable = True

    # Recommended actions
    overfit_rate  = len(report.overfit_folds)  / len(fold_results)
    underfit_rate = len(report.underfit_folds) / len(fold_results)

    if overfit_rate > 0.3:
        actions.append("Increase regularization (reg_alpha, reg_lambda, max_depth−1)")
        actions.append("Increase min_child_samples / min_child_weight")
        actions.append("Reduce n_estimators; rely more on early stopping")

    if underfit_rate > 0.3:
        actions.append("Add more features (more lag periods, additional indicators)")
        actions.append("Reduce regularization strength")
        actions.append("Increase max_depth or num_leaves")

    if report.dominant_feature_folds:
        actions.append("Investigate top dominant feature — may indicate data leakage")
        actions.append("Cap feature importance; consider dropping or capping that feature")

    if report.is_regime_unstable:
        actions.append("High variance across folds — consider shortening training window")
        actions.append("Add regime-awareness features (e.g., VIX-equivalent crypto vol index)")

    report.recommended_actions = actions

    logger.info(f"\n{report.summary()}")
    return report


def check_live_vs_backtest(
    backtest_sharpe: float,
    live_sharpe:     float,
    symbol:          str,
    threshold:       float = 0.5,
) -> bool:
    """
    Returns True if live performance is acceptable relative to backtest.
    Triggers a re-train flag if Sharpe degrades significantly.
    """
    ratio = live_sharpe / (abs(backtest_sharpe) + 1e-9)
    if ratio < threshold:
        logger.warning(
            f"[{symbol}] Live Sharpe ({live_sharpe:.2f}) is < {threshold:.0%} "
            f"of backtest Sharpe ({backtest_sharpe:.2f}). Re-train recommended."
        )
        return False
    return True
