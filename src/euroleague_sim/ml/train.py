"""Train ML models for Euroleague predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from .features import FEATURE_COLS
from .registry import MODEL_REGISTRY, get_model
from .evaluate import evaluate_model


def train_models(
    train_df: pd.DataFrame,
    model_dir: Path,
    *,
    model_name: str = "baseline",
    seed: int = 42,
    cv_folds: int = 5,
    verbose: bool = True,
) -> Dict[str, float]:
    """Train linear ML models, evaluate, and persist.

    Parameters
    ----------
    train_df : DataFrame produced by ``build_training_dataset`` (must contain
        ``FEATURE_COLS`` + ``home_win`` + ``margin``).
    model_dir : directory where trained artefacts will be saved.
    model_name : name of the model to train from the registry, or "all" for leaderboard.
    seed : random state for reproducibility.
    cv_folds : number of `TimeSeriesSplit` folds.
    verbose : if True, print progress to stdout.

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

    _log(f"Training samples: {len(y_cls)}  |  Features: {X.shape[1]}")
    _log(f"Home-win base rate: {y_cls.mean():.3f}")

    if len(y_cls) <= cv_folds:
        raise ValueError(f"Need more than {cv_folds} training rows for TimeSeriesSplit(n_splits={cv_folds}).")

    tscv = TimeSeriesSplit(n_splits=cv_folds)

    if model_name == "all":
        _log("Running leaderboard evaluation for all models...")
        print(f"\n  {'='*80}")
        print(f"  {'Model':<25s} | {'Log-Loss':<9s} | {'Brier':<7s} | {'Acc':<6s} | {'MAE':<6s} | {'RMSE':<6s}")
        print(f"  {'-'*80}")
        
        for name, factory in MODEL_REGISTRY.items():
            model_dict = factory()
            eval_res = evaluate_model(model_dict, X, y_cls, y_reg, tscv)
            m = eval_res.metrics
            print(f"  {model_dict['name']:<25s} | {m['log_loss']:<9.4f} | {m['brier_score']:<7.4f} | "
                  f"{m['cv_accuracy']:<6.3f} | {m['margin_mae']:<6.2f} | {m['margin_rmse']:<6.2f}")
        print(f"  {'='*80}\n")
        return {}

    # Single model training
    model_dict = get_model(model_name)
    _log(f"Evaluating {model_dict['name']} (cross-validation) …")
    
    eval_res = evaluate_model(model_dict, X, y_cls, y_reg, tscv)
    metrics = eval_res.metrics
    
    # ---- Print summary ----
    if verbose:
        print(f"\n  {'='*70}")
        print(f"  TRAINING RESULTS: {model_dict['name']}  ({metrics['n_train_samples']} games, "
              f"{metrics['n_features']} features)")
        print(f"  {'='*70}")
        print(f"  Cross-val accuracy   Win:    {metrics['cv_accuracy']:.3f}")
        print(f"  Brier score          Win:    {metrics['brier_score']:.4f}")
        print(f"  Log-loss             Win:    {metrics['log_loss']:.4f}")
        print(f"  Margin MAE           Margin: {metrics['margin_mae']:.2f}")
        print(f"  Margin RMSE          Margin: {metrics['margin_rmse']:.2f}")
        print(f"  {'='*70}\n")

    # ---- Final Fit ----
    _log(f"Fitting final {model_dict['name']} on full dataset …")
    
    requires_scaling = model_dict.get("requires_scaling", False)
    scaler = None
    X_fit = X
    if requires_scaling:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X)
        
    win_est = model_dict["win"]
    margin_est = model_dict["margin"]
    
    win_est.fit(X_fit, y_cls)
    margin_est.fit(X_fit, y_reg)

    # ---- Persist ----
    out_dir = model_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if requires_scaling and scaler is not None:
        joblib.dump(scaler, out_dir / "scaler.joblib")
        
    joblib.dump(win_est, out_dir / "win_model.joblib")
    joblib.dump(margin_est, out_dir / "margin_model.joblib")

    meta = {
        "model_name": model_name,
        "display_name": model_dict["name"],
        "requires_scaling": requires_scaling,
        "feature_cols": FEATURE_COLS,
        "metrics": metrics,
    }
    
    from .weights import get_weights_from_estimator
    
    win_weights, win_weight_type = get_weights_from_estimator(win_est, FEATURE_COLS)
    if win_weights:
        meta[f"win_{win_weight_type}"] = win_weights
        
    margin_weights, margin_weight_type = get_weights_from_estimator(margin_est, FEATURE_COLS)
    if margin_weights:
        meta[f"margin_{margin_weight_type}"] = margin_weights

    if verbose and win_weights:
        print(f"\n  Win model {win_weight_type} (|weight| descending):")
        for feat, w in sorted(win_weights.items(), key=lambda x: -abs(x[1]))[:7]:
            print(f"    {feat:<22s}  {w:+.4f}")
        print(f"  {'='*70}\n")

    (out_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    _log(f"Models saved to {out_dir.resolve()}")

    # Diagnostic plot
    diagnostic_plot_path = out_dir / "diagnostics.png"
    from .plots import save_training_diagnostics
    save_training_diagnostics(
        ordered_df,
        X_fit,
        y_cls,
        y_reg,
        win_est,
        margin_est,
        eval_res,
        diagnostic_plot_path,
        feature_cols=FEATURE_COLS,
    )
    _log(f"Training diagnostics plot saved to {diagnostic_plot_path.resolve()}")

    return metrics
