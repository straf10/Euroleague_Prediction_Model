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

# import optuna
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.linear_model import LogisticRegression, Ridge


def main(argv=None):
    """Run hyperparameter tuning."""
    parser = argparse.ArgumentParser(description="Offline hyperparameter tuning.")
    parser.add_argument("--model", type=str, default="baseline", help="Model architecture to tune")
    parser.add_argument("--trials", type=int, default=100, help="Number of Optuna trials")
    args = parser.parse_args(argv or sys.argv[1:])
    
    print(f"Tuning skeleton for {args.model} with {args.trials} trials.")
    print("TODO: Implement Optuna study with TimeSeriesSplit.")
    print("TODO: Load dataset using build_training_dataset().")
    print("TODO: Define objective function returning log-loss or MAE.")
    print("TODO: Print best parameters to paste into registry.py.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
