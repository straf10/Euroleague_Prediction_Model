"""Benchmark Beta vs sigmoid (Platt) vs isotonic calibration via Walk-Forward Optimisation.

For each calibration method we run the *exact* same WFO loop the production
pipeline uses (per-round expanding window, CatBoost with our tuned hyperparams
and early stopping, same 9-feature set). Only the calibrator family inside
``CalibratedWinClassifier`` changes. We score the strictly out-of-sample OOF
probabilities with Brier, log-loss, and accuracy, then print a comparison
table so the winner is decided by data, not opinion.

Usage:
    python scripts/compare_calibration.py
    python scripts/compare_calibration.py --wfo-step round --season 2025
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from catboost import CatBoostClassifier, CatBoostRegressor  # noqa: E402

from euroleague_sim.config import ProjectConfig  # noqa: E402
from euroleague_sim.data.cache import Cache  # noqa: E402
from euroleague_sim.ml.calibration import CALIB_METHODS, CalibratedWinClassifier  # noqa: E402
from euroleague_sim.ml.evaluate import build_wfo_periods, walk_forward_evaluate  # noqa: E402
from euroleague_sim.ml.features import FEATURE_COLS  # noqa: E402
from euroleague_sim.pipeline import prepare_training_data  # noqa: E402


# Tuned CatBoost params — must stay in lock-step with registry.get_catboost_model.
CATBOOST_PARAMS = dict(
    iterations=700,
    depth=3,
    learning_rate=0.010320567583828568,
    l2_leaf_reg=1.016367295616664,
    random_seed=42,
    bootstrap_type="Bayesian",
    bagging_temperature=1.1124712815306341,
    early_stopping_rounds=50,
    allow_writing_files=False,
    verbose=False,
)


def _build_model_dict(method: str) -> dict:
    win_clf = CatBoostClassifier(loss_function="Logloss", **CATBOOST_PARAMS)
    margin_reg = CatBoostRegressor(loss_function="RMSE", **CATBOOST_PARAMS)
    return {
        "name": f"CatBoost ({method}-cal + WFO)",
        "win": CalibratedWinClassifier(
            win_clf,
            method=method,
            calib_fraction=0.25,
            min_calib_samples=30,
        ),
        "margin": margin_reg,
        "requires_scaling": False,
        "walk_forward": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(ROOT / "data_cache"))
    ap.add_argument("--config", default=None)
    ap.add_argument("--season", type=int, default=2025, help="Current season start year.")
    ap.add_argument(
        "--wfo-step",
        default="round",
        help="Walk-forward granularity: 'round' (final-eval precision), 'season', or int block size.",
    )
    ap.add_argument(
        "--methods",
        default=",".join(CALIB_METHODS),
        help="Comma-separated subset of methods to benchmark (default: all).",
    )
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in CALIB_METHODS:
            raise SystemExit(f"Unknown calibration method '{m}'. Use one of {CALIB_METHODS}.")

    cfg = ProjectConfig.default()
    if args.config:
        cfg = ProjectConfig.load(Path(args.config))
    cache = Cache(Path(args.cache_dir))

    print(f"Loading training data (season {args.season})…")
    train_df = prepare_training_data(cache, cfg, args.season, verbose=False)
    if all(c in train_df.columns for c in ["season", "round", "gamecode"]):
        ordered_df = train_df.sort_values(["season", "round", "gamecode"]).reset_index(drop=True)
    else:
        ordered_df = train_df.copy()

    X = ordered_df[FEATURE_COLS].values.astype(float)
    y_win = ordered_df["home_win"].values.astype(int)
    y_margin = ordered_df["margin"].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    wfo_step: object = int(args.wfo_step) if str(args.wfo_step).isdigit() else args.wfo_step
    periods = build_wfo_periods(ordered_df, step=wfo_step)
    print(
        f"  rows={len(y_win)}  features={X.shape[1]}  wfo_step='{wfo_step}'  "
        f"periods={len(np.unique(periods))}  min_train_size={cfg.ml.wfo_min_train_size}"
    )

    results: list[tuple[str, dict, float]] = []
    for method in methods:
        print(f"\n→ Walk-forward eval ({method} calibration)…")
        t0 = time.perf_counter()
        model_dict = _build_model_dict(method)
        eval_res = walk_forward_evaluate(
            model_dict, X, y_win, y_margin, periods,
            min_train_size=cfg.ml.wfo_min_train_size,
        )
        dt = time.perf_counter() - t0
        m = eval_res.metrics
        print(
            f"   folds={m.get('n_wfo_folds', '?')}  oos={m['n_oos_samples']}  "
            f"brier={m['brier_score']:.4f}  log_loss={m['log_loss']:.4f}  "
            f"acc={m['cv_accuracy']:.3f}  ({dt:.1f}s)"
        )
        results.append((method, m, dt))

    print("\n" + "=" * 78)
    print(f"  {'Method':<10s} | {'Brier':>8s} | {'LogLoss':>8s} | {'Accuracy':>9s} | "
          f"{'OOS':>5s} | {'Folds':>5s} | {'Time(s)':>7s}")
    print("  " + "-" * 76)
    for method, m, dt in results:
        print(
            f"  {method:<10s} | {m['brier_score']:>8.4f} | {m['log_loss']:>8.4f} | "
            f"{m['cv_accuracy']:>9.3f} | {m['n_oos_samples']:>5d} | "
            f"{m.get('n_wfo_folds', 0):>5d} | {dt:>7.1f}"
        )
    print("=" * 78)

    # Pick winner by Brier score (lower is better); tie-break by log-loss.
    best = min(results, key=lambda r: (r[1]["brier_score"], r[1]["log_loss"]))
    print(f"\n  Winner (lowest Brier → lowest log-loss): {best[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
