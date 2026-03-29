"""Offline hyperparameter tuning script using Optuna.

This script is isolated from the daily training pipeline.
It is used for "careful examination and testing" of new model architectures.
Once optimal parameters are found, they should be hardcoded into `registry.py`.

Usage:
    python -m euroleague_sim.ml.tune
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import optuna
import matplotlib.pyplot as plt
from optuna.visualization.matplotlib import plot_optimization_history
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit

from ..config import ProjectConfig
from ..data.cache import Cache
from ..pipeline import prepare_training_data
from .features import FEATURE_COLS
from .evaluate import evaluate_model


def main(argv=None):
    """Run hyperparameter tuning."""
    parser = argparse.ArgumentParser(description="Offline hyperparameter tuning.")
    parser.add_argument("--trials", type=int, default=100, help="Number of Optuna trials per study")
    parser.add_argument("--cache-dir", default="data_cache", help="Folder for cached data")
    parser.add_argument("--config", default=None, help="Path to config.json (optional)")
    parser.add_argument("--season", type=int, default=2025, help="Current season start year")
    args = parser.parse_args(argv or sys.argv[1:])
    
    cfg = ProjectConfig.default()
    if args.config:
        cfg = ProjectConfig.load(Path(args.config))
        
    cache = Cache(Path(args.cache_dir))
    
    print(f"Loading training data for tuning (season {args.season})...")
    train_df = prepare_training_data(cache, cfg, args.season, verbose=False)
    
    if all(c in train_df.columns for c in ["season", "round", "gamecode"]):
        ordered_df = train_df.sort_values(["season", "round", "gamecode"]).reset_index(drop=True)
    else:
        ordered_df = train_df.copy()

    X = ordered_df[FEATURE_COLS].values.astype(float)
    y_win = ordered_df["home_win"].values.astype(int)
    y_margin = ordered_df["margin"].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    tscv = TimeSeriesSplit(n_splits=cfg.ml.cv_folds)
    
    print(f"\n{'='*80}")
    print(f"  Study 1: Tuning LogisticRegression (Win Probability)")
    print(f"{'='*80}")
    
    def objective_win(trial) -> float:
        C = trial.suggest_float("C", 0.01, 10.0, log=True)
        model_dict = {
            "name": "Tuning Win",
            "win": CalibratedClassifierCV(
                LogisticRegression(C=C, max_iter=1000, solver="lbfgs", random_state=42),
                method="sigmoid",
                cv=3,
            ),
            "margin": Ridge(alpha=75.0, random_state=42), # Fixed baseline
            "requires_scaling": True,
        }
        eval_result = evaluate_model(model_dict, X, y_win, y_margin, tscv)
        return eval_result.metrics["log_loss"]

    study_win = optuna.create_study(direction="minimize")
    study_win.optimize(objective_win, n_trials=args.trials)
    
    print(f"\n{'='*80}")
    print(f"  Study 2: Tuning Ridge (Point Margin)")
    print(f"{'='*80}")
    
    def objective_margin(trial) -> float:
        alpha = trial.suggest_float("alpha", 10.0, 200.0)
        model_dict = {
            "name": "Tuning Margin",
            "win": CalibratedClassifierCV(
                LogisticRegression(C=0.3, max_iter=1000, solver="lbfgs", random_state=42), # Fixed baseline
                method="sigmoid",
                cv=3,
            ),
            "margin": Ridge(alpha=alpha, random_state=42),
            "requires_scaling": True,
        }
        eval_result = evaluate_model(model_dict, X, y_win, y_margin, tscv)
        return eval_result.metrics["margin_mae"]

    study_margin = optuna.create_study(direction="minimize")
    study_margin.optimize(objective_margin, n_trials=args.trials)
    
    print(f"\n{'='*80}")
    print(f"  Tuning Complete")
    print(f"{'='*80}")
    print(f"  Best Win Model (LogLoss): {study_win.best_value:.4f}")
    print(f"  Best Win Params: {study_win.best_params}")
    print(f"  Best Margin Model (MAE): {study_margin.best_value:.4f}")
    print(f"  Best Margin Params: {study_margin.best_params}")
    print(f"  -> Update these in src/euroleague_sim/ml/registry.py")
    print(f"{'='*80}\n")
    
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    plot_optimization_history(study_win)
    plt.gcf().savefig(plots_dir / "tuning_history_win.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    plot_optimization_history(study_margin)
    plt.gcf().savefig(plots_dir / "tuning_history_margin.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved optimization plots to {plots_dir.resolve()}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
