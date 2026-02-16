"""Train Random Forest + Neural Network models for Euroleague predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
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
    rf_n_estimators: int = 300,
    rf_max_depth: int = 6,
    rf_min_samples_leaf: int = 5,
    nn_hidden_layers: Tuple[int, ...] = (64, 32),
    nn_alpha: float = 0.001,
    nn_max_iter: int = 1000,
    seed: int = 42,
    cv_folds: int = 5,
    verbose: bool = True,
) -> Dict[str, float]:
    """Train all ML models, evaluate with cross-validation, and persist.

    Parameters
    ----------
    train_df : DataFrame produced by ``build_training_dataset`` (must contain
        ``FEATURE_COLS`` + ``home_win`` + ``margin``).
    model_dir : directory where trained artefacts will be saved.
    verbose : if True, print progress to stdout.

    Returns
    -------
    Dictionary of evaluation metrics.
    """

    def _log(msg: str) -> None:
        if verbose:
            print(f"  [train] {msg}")

    # ---- Prepare X / y ----
    X = train_df[FEATURE_COLS].values.astype(float)
    y_cls = train_df["home_win"].values.astype(int)
    y_reg = train_df["margin"].values.astype(float)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    _log(f"Training samples: {len(y_cls)}  |  Features: {X.shape[1]}")
    _log(f"Home-win base rate: {y_cls.mean():.3f}")

    # ------------------------------------------------------------------ RF
    _log("Training Random Forest classifier …")
    rf_clf = RandomForestClassifier(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        min_samples_leaf=rf_min_samples_leaf,
        random_state=seed,
        n_jobs=-1,
    )
    rf_clf.fit(X_scaled, y_cls)

    _log("Training Random Forest regressor …")
    rf_reg = RandomForestRegressor(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        min_samples_leaf=rf_min_samples_leaf,
        random_state=seed,
        n_jobs=-1,
    )
    rf_reg.fit(X_scaled, y_reg)

    # ------------------------------------------------------------------ NN
    _log("Training Neural Network classifier …")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        nn_clf = MLPClassifier(
            hidden_layer_sizes=tuple(nn_hidden_layers),
            activation="relu",
            solver="adam",
            alpha=nn_alpha,
            max_iter=nn_max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=seed,
        )
        nn_clf.fit(X_scaled, y_cls)

    _log("Training Neural Network regressor …")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        nn_reg = MLPRegressor(
            hidden_layer_sizes=tuple(nn_hidden_layers),
            activation="relu",
            solver="adam",
            alpha=nn_alpha,
            max_iter=nn_max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=seed,
        )
        nn_reg.fit(X_scaled, y_reg)

    # ------------------------------------------------------------ Evaluate
    _log("Evaluating (cross-validation) …")

    cv_acc_rf = float(
        cross_val_score(rf_clf, X_scaled, y_cls, cv=cv_folds, scoring="accuracy").mean()
    )
    cv_acc_nn = float(
        cross_val_score(nn_clf, X_scaled, y_cls, cv=cv_folds, scoring="accuracy").mean()
    )

    rf_proba  = rf_clf.predict_proba(X_scaled)[:, 1]
    nn_proba  = nn_clf.predict_proba(X_scaled)[:, 1]
    ens_proba = 0.5 * rf_proba + 0.5 * nn_proba

    rf_margin  = rf_reg.predict(X_scaled)
    nn_margin  = nn_reg.predict(X_scaled)
    ens_margin = 0.5 * rf_margin + 0.5 * nn_margin

    metrics: Dict[str, float] = {
        "n_train_samples":          int(len(y_cls)),
        "n_features":               int(X.shape[1]),
        "home_win_base_rate":       float(y_cls.mean()),
        # Cross-validated accuracy
        "rf_cv_accuracy":           cv_acc_rf,
        "nn_cv_accuracy":           cv_acc_nn,
        # Training accuracy (sanity check)
        "rf_train_accuracy":        float(accuracy_score(y_cls, (rf_proba > 0.5).astype(int))),
        "nn_train_accuracy":        float(accuracy_score(y_cls, (nn_proba > 0.5).astype(int))),
        "ensemble_train_accuracy":  float(accuracy_score(y_cls, (ens_proba > 0.5).astype(int))),
        # Brier score (lower = better calibrated)
        "rf_brier":                 float(brier_score_loss(y_cls, rf_proba)),
        "nn_brier":                 float(brier_score_loss(y_cls, nn_proba)),
        "ensemble_brier":           float(brier_score_loss(y_cls, ens_proba)),
        # Log-loss
        "rf_logloss":               float(log_loss(y_cls, rf_proba)),
        "nn_logloss":               float(log_loss(y_cls, nn_proba)),
        "ensemble_logloss":         float(log_loss(y_cls, ens_proba)),
        # Margin prediction
        "rf_margin_mae":            float(mean_absolute_error(y_reg, rf_margin)),
        "nn_margin_mae":            float(mean_absolute_error(y_reg, nn_margin)),
        "ensemble_margin_mae":      float(mean_absolute_error(y_reg, ens_margin)),
        "rf_margin_rmse":           float(np.sqrt(mean_squared_error(y_reg, rf_margin))),
        "nn_margin_rmse":           float(np.sqrt(mean_squared_error(y_reg, nn_margin))),
        "ensemble_margin_rmse":     float(np.sqrt(mean_squared_error(y_reg, ens_margin))),
    }

    # Feature importances (from RF classifier)
    importance = dict(zip(FEATURE_COLS, rf_clf.feature_importances_.tolist()))

    # ---- Print summary ----
    if verbose:
        print(f"\n  {'='*60}")
        print(f"  TRAINING RESULTS  ({metrics['n_train_samples']} games, "
              f"{metrics['n_features']} features)")
        print(f"  {'='*60}")
        print(f"  Cross-val accuracy   RF: {cv_acc_rf:.3f}   NN: {cv_acc_nn:.3f}")
        print(f"  Train accuracy       RF: {metrics['rf_train_accuracy']:.3f}   "
              f"NN: {metrics['nn_train_accuracy']:.3f}   "
              f"Ensemble: {metrics['ensemble_train_accuracy']:.3f}")
        print(f"  Brier score          RF: {metrics['rf_brier']:.4f}   "
              f"NN: {metrics['nn_brier']:.4f}   "
              f"Ensemble: {metrics['ensemble_brier']:.4f}")
        print(f"  Margin MAE           RF: {metrics['rf_margin_mae']:.2f}   "
              f"NN: {metrics['nn_margin_mae']:.2f}   "
              f"Ensemble: {metrics['ensemble_margin_mae']:.2f}")
        print(f"\n  Top-5 RF feature importances:")
        for feat, imp in sorted(importance.items(), key=lambda x: -x[1])[:5]:
            print(f"    {feat:<22s}  {imp:.4f}")
        print(f"  {'='*60}\n")

    # ---- Persist ----
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler,  model_dir / "scaler.joblib")
    joblib.dump(rf_clf,  model_dir / "rf_classifier.joblib")
    joblib.dump(rf_reg,  model_dir / "rf_regressor.joblib")
    joblib.dump(nn_clf,  model_dir / "nn_classifier.joblib")
    joblib.dump(nn_reg,  model_dir / "nn_regressor.joblib")

    meta = {
        "feature_cols": FEATURE_COLS,
        "metrics": metrics,
        "feature_importances": importance,
    }
    (model_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    _log(f"Models saved to {model_dir.resolve()}")
    return metrics
