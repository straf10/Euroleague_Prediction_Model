"""Evaluation module for ML models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Union

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from .calibration import fit_catboost_es


def build_wfo_periods(
    ordered_df: pd.DataFrame,
    step: Union[str, int] = "round",
) -> np.ndarray:
    """Build chronological walk-forward period ids at the requested granularity.

    Parameters
    ----------
    ordered_df : training frame already sorted by ``season``, ``round``, ``gamecode``.
    step :
        ``"round"``  – one period per season-round (most precise OOS; used for final eval),
        ``"season"`` – one period per season (coarsest; fastest, fewest refits),
        ``int N``     – blocks of ``N`` rounds within each season (e.g. 5).

    ``season * 100 + round`` is monotonic in calendar time since ``round`` is well
    under 100, so the returned ids sort in true chronological order.
    """
    if not all(c in ordered_df.columns for c in ["season", "round"]):
        return np.arange(len(ordered_df))

    season = ordered_df["season"].astype(int)
    rnd = ordered_df["round"].astype(int)

    if step == "season":
        return season.to_numpy()
    if step == "round":
        return (season * 100 + rnd).to_numpy()

    block = int(step)
    if block < 1:
        raise ValueError(f"Integer wfo_step must be >= 1, got {step}.")
    return (season * 100 + (rnd // block)).to_numpy()


@dataclass
class EvalResult:
    metrics: Dict[str, float]
    oof_proba: np.ndarray
    oof_margin: np.ndarray
    y_win: np.ndarray
    y_margin: np.ndarray


def _oof_metrics(
    oof_proba: np.ndarray,
    oof_margin: np.ndarray,
    y_win: np.ndarray,
    y_margin: np.ndarray,
    n_features: int,
    extra: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """Compute the standard metric dict from out-of-sample predictions."""
    mask = np.isfinite(oof_proba)
    y_win_eval = y_win[mask]
    y_margin_eval = y_margin[mask]
    proba_eval = oof_proba[mask]
    margin_eval = oof_margin[mask]

    metrics = {
        "n_train_samples": int(len(y_win)),
        "n_features": int(n_features),
        "home_win_base_rate": float(y_win.mean()),
        "n_oos_samples": int(mask.sum()),
        "cv_accuracy": float(accuracy_score(y_win_eval, (proba_eval > 0.5).astype(int))),
        "brier_score": float(brier_score_loss(y_win_eval, proba_eval)),
        "log_loss": float(log_loss(y_win_eval, proba_eval)),
        "margin_mae": float(mean_absolute_error(y_margin_eval, margin_eval)),
        "margin_rmse": float(np.sqrt(mean_squared_error(y_margin_eval, margin_eval))),
    }
    if extra:
        metrics.update(extra)
    return metrics


def evaluate_model(
    model_dict: Dict[str, Any],
    X: np.ndarray,
    y_win: np.ndarray,
    y_margin: np.ndarray,
    tscv: TimeSeriesSplit,
) -> EvalResult:
    """Time-series CV evaluation, returns metrics + OOF predictions + calibration data.
    
    Reads model_dict["requires_scaling"] to decide whether to apply StandardScaler per fold.
    """
    n_samples = len(y_win)
    oof_proba = np.full(n_samples, np.nan, dtype=float)
    oof_margin = np.full(n_samples, np.nan, dtype=float)
    
    requires_scaling = model_dict.get("requires_scaling", False)
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_win_train = y_win[train_idx]
        y_margin_train = y_margin[train_idx]
        
        if requires_scaling:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
        win_est = clone(model_dict["win"])
        margin_est = clone(model_dict["margin"])
        
        win_est.fit(X_train, y_win_train)
        margin_est.fit(X_train, y_margin_train)
        
        oof_proba[test_idx] = win_est.predict_proba(X_test)[:, 1]
        oof_margin[test_idx] = margin_est.predict(X_test)

    metrics = _oof_metrics(oof_proba, oof_margin, y_win, y_margin, X.shape[1])

    return EvalResult(
        metrics=metrics,
        oof_proba=oof_proba,
        oof_margin=oof_margin,
        y_win=y_win,
        y_margin=y_margin,
    )


def walk_forward_evaluate(
    model_dict: Dict[str, Any],
    X: np.ndarray,
    y_win: np.ndarray,
    y_margin: np.ndarray,
    periods: np.ndarray,
    *,
    min_train_size: int = 200,
) -> EvalResult:
    """True Walk-Forward Optimisation (WFO) evaluation.

    Iterates chronologically over ``periods`` (e.g. one period per season-round).
    At each step the model trains on the *expanding* window of all strictly
    earlier periods and predicts the immediate, unseen next period. Predictions
    are therefore strictly out-of-sample and free of temporal leakage.

    The win estimator (a :class:`BetaCalibratedClassifier`) calibrates itself on
    a held-out tail of each expanding window, so Beta calibration is applied to
    the CatBoost probabilities at every out-of-sample WFO step.

    Parameters
    ----------
    periods : integer period id per row, ascending and aligned with ``X``.
    min_train_size : minimum rows required in the expanding window before a
        period is scored (early periods without enough history are skipped,
        mirroring the warm-up folds of ``TimeSeriesSplit``).
    """
    n_samples = len(y_win)
    oof_proba = np.full(n_samples, np.nan, dtype=float)
    oof_margin = np.full(n_samples, np.nan, dtype=float)

    requires_scaling = model_dict.get("requires_scaling", False)
    unique_periods = np.unique(periods)

    n_folds = 0
    for p in unique_periods:
        test_mask = periods == p
        train_mask = periods < p
        if train_mask.sum() < min_train_size:
            continue
        # Need both outcome classes in the training window to fit a classifier.
        if len(np.unique(y_win[train_mask])) < 2:
            continue

        X_train, X_test = X[train_mask], X[test_mask]
        y_win_train = y_win[train_mask]
        y_margin_train = y_margin[train_mask]

        if requires_scaling:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        win_est = clone(model_dict["win"])
        margin_est = clone(model_dict["margin"])

        # Win estimator handles its own calibration + early-stopping tail internally;
        # the margin estimator gets early stopping via the shared helper.
        win_est.fit(X_train, y_win_train)
        margin_est = fit_catboost_es(margin_est, X_train, y_margin_train)

        oof_proba[test_mask] = win_est.predict_proba(X_test)[:, 1]
        oof_margin[test_mask] = margin_est.predict(X_test)
        n_folds += 1

    if n_folds == 0:
        raise ValueError(
            f"Walk-forward produced no scorable folds (min_train_size={min_train_size}, "
            f"{len(unique_periods)} periods, {n_samples} rows). Lower min_train_size."
        )

    metrics = _oof_metrics(
        oof_proba, oof_margin, y_win, y_margin, X.shape[1],
        extra={"n_wfo_folds": int(n_folds)},
    )

    return EvalResult(
        metrics=metrics,
        oof_proba=oof_proba,
        oof_margin=oof_margin,
        y_win=y_win,
        y_margin=y_margin,
    )
