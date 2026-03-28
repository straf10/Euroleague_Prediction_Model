"""Evaluation module for ML models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
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


@dataclass
class EvalResult:
    metrics: Dict[str, float]
    oof_proba: np.ndarray
    oof_margin: np.ndarray
    y_win: np.ndarray
    y_margin: np.ndarray


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
        
    mask = np.isfinite(oof_proba)
    y_win_eval = y_win[mask]
    y_margin_eval = y_margin[mask]
    proba_eval = oof_proba[mask]
    margin_eval = oof_margin[mask]
    
    metrics = {
        "n_train_samples": int(n_samples),
        "n_features": int(X.shape[1]),
        "home_win_base_rate": float(y_win.mean()),
        "cv_accuracy": float(accuracy_score(y_win_eval, (proba_eval > 0.5).astype(int))),
        "brier_score": float(brier_score_loss(y_win_eval, proba_eval)),
        "log_loss": float(log_loss(y_win_eval, proba_eval)),
        "margin_mae": float(mean_absolute_error(y_margin_eval, margin_eval)),
        "margin_rmse": float(np.sqrt(mean_squared_error(y_margin_eval, margin_eval))),
    }
    
    return EvalResult(
        metrics=metrics,
        oof_proba=oof_proba,
        oof_margin=oof_margin,
        y_win=y_win,
        y_margin=y_margin,
    )
