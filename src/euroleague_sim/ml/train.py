"""Train Logistic Regression + Ridge models for Euroleague predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import joblib

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)

from .features import FEATURE_COLS


def train_models(
    train_df: pd.DataFrame,
    model_dir: Path,
    *,
    logreg_C: float = 1.0,
    logreg_max_iter: int = 1000,
    ridge_alpha: float = 1.0,
    seed: int = 42,
    cv_folds: int = 5,
    verbose: bool = True,
    diagnostic_plot_path: Path | None = None,
) -> Dict[str, float]:
    """Train tuned linear ML models, evaluate, and persist.

    Models trained:
    - LogisticRegression  (binary home_win classifier)
    - Ridge               (margin regressor)

    Parameters
    ----------
    train_df : DataFrame produced by ``build_training_dataset`` (must contain
        ``FEATURE_COLS`` + ``home_win`` + ``margin``).
    model_dir : directory where trained artefacts will be saved.
    logreg_C : unused; ``C`` is tuned over
        ``[0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 10.0, 100.0]``.
    logreg_max_iter : maximum iterations for LogisticRegression solver.
    ridge_alpha : unused; ``alpha`` is tuned over
        ``[10.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 250.0, 500.0, 1000.0]``.
    seed : random state for reproducibility.
    cv_folds : number of `TimeSeriesSplit` folds.
    verbose : if True, print progress to stdout.
    diagnostic_plot_path : if set, write a 2×2 diagnostics PNG (coefficients,
        correlation heatmap, calibration curve, margin scatter).

    Returns
    -------
    Dictionary of evaluation metrics.
    """

    def _log(msg: str) -> None:
        if verbose:
            print(f"  [train] {msg}")

    # ---- Prepare X / y ----
    if all(c in train_df.columns for c in ["season", "round", "gamecode"]):
        ordered_df = train_df.sort_values(["season", "round", "gamecode"]).reset_index(drop=True)
    else:
        ordered_df = train_df.copy()

    X = ordered_df[FEATURE_COLS].values.astype(float)
    y_cls = ordered_df["home_win"].values.astype(int)
    y_reg = ordered_df["margin"].values.astype(float)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    _log(f"Training samples: {len(y_cls)}  |  Features: {X.shape[1]}")
    _log(f"Home-win base rate: {y_cls.mean():.3f}")

    # ---- Hyperparameter search (time-series aware) ----
    if len(y_cls) <= 5:
        raise ValueError("Need more than 5 training rows for TimeSeriesSplit(n_splits=5).")

    tscv = TimeSeriesSplit(n_splits=5)

    _log("Tuning Logistic Regression classifier with GridSearchCV …")
    logreg_grid = GridSearchCV(
        estimator=LogisticRegression(
            max_iter=logreg_max_iter,
            solver="lbfgs",
            random_state=seed,
        ),
        param_grid={"C": [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 10.0, 100.0]},
        cv=tscv,
        scoring="accuracy",
        refit=True,
    )
    logreg_grid.fit(X_scaled, y_cls)
    best_logreg = logreg_grid.best_estimator_

    _log("Tuning Ridge regressor with GridSearchCV …")
    ridge_grid = GridSearchCV(
        estimator=Ridge(random_state=seed),
        param_grid={
            "alpha": [
                10.0,
                50.0,
                75.0,
                100.0,
                125.0,
                150.0,
                200.0,
                250.0,
                500.0,
                1000.0,
            ],
        },
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        refit=True,
    )
    ridge_grid.fit(X_scaled, y_reg)
    best_ridge = ridge_grid.best_estimator_

    _log("Calibrating best Logistic Regression probabilities …")
    calibrated_logreg = CalibratedClassifierCV(
        best_logreg,
        method="sigmoid",
        cv=tscv,
    )
    calibrated_logreg.fit(X_scaled, y_cls)

    # ---- Evaluate ----
    _log("Evaluating (cross-validation) …")

    cv_acc_logreg = float(logreg_grid.best_score_)

    logreg_proba = calibrated_logreg.predict_proba(X_scaled)[:, 1]
    ridge_margin = best_ridge.predict(X_scaled)

    metrics: Dict[str, float] = {
        "n_train_samples":          int(len(y_cls)),
        "n_features":               int(X.shape[1]),
        "home_win_base_rate":       float(y_cls.mean()),
        "best_logreg_C":            float(logreg_grid.best_params_["C"]),
        "best_ridge_alpha":         float(ridge_grid.best_params_["alpha"]),
        "logreg_cv_accuracy":       cv_acc_logreg,
        "logreg_train_accuracy":    float(accuracy_score(y_cls, (logreg_proba > 0.5).astype(int))),
        "logreg_brier":             float(brier_score_loss(y_cls, logreg_proba)),
        "logreg_logloss":           float(log_loss(y_cls, logreg_proba)),
        "ridge_margin_mae":         float(mean_absolute_error(y_reg, ridge_margin)),
        "ridge_margin_rmse":        float(np.sqrt(mean_squared_error(y_reg, ridge_margin))),
    }

    # ---- Feature coefficients ----
    logreg_coefs = dict(zip(FEATURE_COLS, best_logreg.coef_[0].tolist()))
    ridge_coefs = dict(zip(FEATURE_COLS, best_ridge.coef_.tolist()))

    # ---- Print summary ----
    if verbose:
        print(f"\n  {'='*70}")
        print(f"  TRAINING RESULTS  ({metrics['n_train_samples']} games, "
              f"{metrics['n_features']} features)")
        print(f"  {'='*70}")
        print(f"  Best LogisticRegression C: {metrics['best_logreg_C']:.4g}")
        print(f"  Best Ridge alpha:         {metrics['best_ridge_alpha']:.4g}")
        print(f"  Cross-val accuracy   LogReg: {cv_acc_logreg:.3f}")
        print(f"  Train accuracy       LogReg: {metrics['logreg_train_accuracy']:.3f}")
        print(f"  Brier score          LogReg: {metrics['logreg_brier']:.4f}")
        print(f"  Log-loss             LogReg: {metrics['logreg_logloss']:.4f}")
        print(f"  Margin MAE           Ridge:  {metrics['ridge_margin_mae']:.2f}")
        print(f"  Margin RMSE          Ridge:  {metrics['ridge_margin_rmse']:.2f}")
        print(f"\n  Logistic Regression coefficients (|weight| descending):")
        for feat, w in sorted(logreg_coefs.items(), key=lambda x: -abs(x[1]))[:7]:
            print(f"    {feat:<22s}  {w:+.4f}")
        print(f"  {'='*70}\n")

    # ---- Persist ----
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, model_dir / "scaler.joblib")
    joblib.dump(calibrated_logreg, model_dir / "logreg.joblib")
    joblib.dump(best_ridge,  model_dir / "ridge.joblib")

    meta = {
        "feature_cols": FEATURE_COLS,
        "metrics": metrics,
        "logreg_coefficients": logreg_coefs,
        "logreg_intercept": float(best_logreg.intercept_[0]),
        "ridge_coefficients": ridge_coefs,
        "ridge_intercept": float(best_ridge.intercept_),
    }
    (model_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    _log(f"Models saved to {model_dir.resolve()}")

    if diagnostic_plot_path is not None:
        from .plots import save_training_diagnostics

        save_training_diagnostics(
            ordered_df,
            X_scaled,
            y_cls,
            y_reg,
            best_logreg,
            best_ridge,
            diagnostic_plot_path,
            feature_cols=FEATURE_COLS,
            cv_folds=cv_folds,
        )
        _log(f"Training diagnostics plot saved to {diagnostic_plot_path.resolve()}")

    return metrics
