"""Evaluation script: compare 3-season vs 5-season model performance & overfitting."""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss,
    mean_absolute_error, mean_squared_error,
)
from xgboost import XGBClassifier, XGBRegressor

sys.path.insert(0, str(Path(__file__).parent / "src"))

from euroleague_sim.config import ProjectConfig
from euroleague_sim.data.cache import Cache
from euroleague_sim.features.elo import run_elo
from euroleague_sim.ml.features import build_training_dataset, FEATURE_COLS

cfg = ProjectConfig.default()
cache = Cache(Path("data_cache"))

SEED = 42
CV_FOLDS = 5

OLD_METRICS = {
    "n_train_samples": 939,
    "rf_cv_accuracy": 0.6368,
    "xgb_cv_accuracy": None,
    "nn_cv_accuracy": 0.6443,
    "rf_train_accuracy": 0.7529,
    "xgb_train_accuracy": None,
    "nn_train_accuracy": 0.6390,
    "ensemble_train_accuracy": 0.7061,
    "rf_brier": 0.1691,
    "nn_brier": 0.2244,
    "ensemble_brier": 0.1935,
    "rf_margin_mae": 7.82,
    "nn_margin_mae": 9.46,
    "ensemble_margin_mae": 8.59,
    "rf_margin_rmse": 9.99,
    "nn_margin_rmse": 12.10,
    "ensemble_margin_rmse": 10.98,
}


def load_seasons(season_list):
    """Load training data for given seasons."""
    seasons_data = {}
    for s in season_list:
        games_k = f"feat_games_E{s}"
        tg_k = f"feat_team_game_E{s}"
        if not cache.has_df(games_k) or not cache.has_df(tg_k):
            print(f"  Skipping season {s}: not in cache")
            continue
        games_df = cache.load_df(games_k)
        team_game_df = cache.load_df(tg_k)
        if games_df.empty:
            continue
        elo_result = run_elo(
            games_df, base=cfg.elo.base, k=cfg.elo.k,
            home_advantage=cfg.elo.home_advantage,
        )
        seasons_data[s] = (games_df, team_game_df, elo_result.game_elos)
    return seasons_data


def build_models(seed=SEED):
    """Create fresh model instances."""
    rf_clf = RandomForestClassifier(
        n_estimators=300, max_depth=4, min_samples_leaf=5,
        min_samples_split=10, random_state=seed, n_jobs=-1,
    )
    rf_reg = RandomForestRegressor(
        n_estimators=300, max_depth=4, min_samples_leaf=5,
        min_samples_split=10, random_state=seed, n_jobs=-1,
    )
    xgb_clf = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="logloss",
        use_label_encoder=False, random_state=seed, n_jobs=-1, verbosity=0,
    )
    xgb_reg = XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        objective="reg:squarederror", random_state=seed, n_jobs=-1, verbosity=0,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        nn_clf = MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
            alpha=0.001, max_iter=1000, early_stopping=True,
            validation_fraction=0.15, random_state=seed,
        )
        nn_reg = MLPRegressor(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
            alpha=0.001, max_iter=1000, early_stopping=True,
            validation_fraction=0.15, random_state=seed,
        )
    return rf_clf, rf_reg, xgb_clf, xgb_reg, nn_clf, nn_reg


def evaluate_dataset(train_df, label=""):
    """Full evaluation: train accuracy, CV accuracy, overfitting gap."""
    X = train_df[FEATURE_COLS].values.astype(float)
    y_cls = train_df["home_win"].values.astype(int)
    y_reg = train_df["margin"].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\n{'='*75}")
    print(f"  {label}")
    print(f"  Samples: {len(y_cls)}  |  Features: {X.shape[1]}  |  "
          f"Home-win rate: {y_cls.mean():.3f}")
    print(f"{'='*75}")

    rf_clf, rf_reg, xgb_clf, xgb_reg, nn_clf, nn_reg = build_models()

    # --- Train all models ---
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        rf_clf.fit(X_scaled, y_cls)
        rf_reg.fit(X_scaled, y_reg)
        xgb_clf.fit(X_scaled, y_cls)
        xgb_reg.fit(X_scaled, y_reg)
        nn_clf.fit(X_scaled, y_cls)
        nn_reg.fit(X_scaled, y_reg)

    # --- Cross-val accuracy ---
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        cv_acc_rf = cross_val_score(
            RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=5,
                                   min_samples_split=10, random_state=SEED, n_jobs=-1),
            X_scaled, y_cls, cv=CV_FOLDS, scoring="accuracy",
        ).mean()
        cv_acc_xgb = cross_val_score(
            XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                          reg_lambda=1.0, objective="binary:logistic",
                          eval_metric="logloss", use_label_encoder=False,
                          random_state=SEED, n_jobs=-1, verbosity=0),
            X_scaled, y_cls, cv=CV_FOLDS, scoring="accuracy",
        ).mean()
        cv_acc_nn = cross_val_score(
            MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                          solver="adam", alpha=0.001, max_iter=1000,
                          early_stopping=True, validation_fraction=0.15,
                          random_state=SEED),
            X_scaled, y_cls, cv=CV_FOLDS, scoring="accuracy",
        ).mean()

    # --- Cross-val margin MAE ---
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        cv_mae_rf = -cross_val_score(
            RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=5,
                                  min_samples_split=10, random_state=SEED, n_jobs=-1),
            X_scaled, y_reg, cv=CV_FOLDS, scoring="neg_mean_absolute_error",
        ).mean()
        cv_mae_xgb = -cross_val_score(
            XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                         reg_lambda=1.0, objective="reg:squarederror",
                         random_state=SEED, n_jobs=-1, verbosity=0),
            X_scaled, y_reg, cv=CV_FOLDS, scoring="neg_mean_absolute_error",
        ).mean()
        cv_mae_nn = -cross_val_score(
            MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                         solver="adam", alpha=0.001, max_iter=1000,
                         early_stopping=True, validation_fraction=0.15,
                         random_state=SEED),
            X_scaled, y_reg, cv=CV_FOLDS, scoring="neg_mean_absolute_error",
        ).mean()

    # --- Train-set metrics ---
    rf_proba = rf_clf.predict_proba(X_scaled)[:, 1]
    xgb_proba = xgb_clf.predict_proba(X_scaled)[:, 1]
    nn_proba = nn_clf.predict_proba(X_scaled)[:, 1]
    ens_proba = (rf_proba + xgb_proba + nn_proba) / 3.0

    rf_margin = rf_reg.predict(X_scaled)
    xgb_margin = xgb_reg.predict(X_scaled)
    nn_margin = nn_reg.predict(X_scaled)
    ens_margin = (rf_margin + xgb_margin + nn_margin) / 3.0

    train_acc_rf = accuracy_score(y_cls, (rf_proba > 0.5).astype(int))
    train_acc_xgb = accuracy_score(y_cls, (xgb_proba > 0.5).astype(int))
    train_acc_nn = accuracy_score(y_cls, (nn_proba > 0.5).astype(int))
    train_acc_ens = accuracy_score(y_cls, (ens_proba > 0.5).astype(int))

    brier_rf = brier_score_loss(y_cls, rf_proba)
    brier_xgb = brier_score_loss(y_cls, xgb_proba)
    brier_nn = brier_score_loss(y_cls, nn_proba)
    brier_ens = brier_score_loss(y_cls, ens_proba)

    logloss_rf = log_loss(y_cls, rf_proba)
    logloss_xgb = log_loss(y_cls, xgb_proba)
    logloss_nn = log_loss(y_cls, nn_proba)
    logloss_ens = log_loss(y_cls, ens_proba)

    mae_rf = mean_absolute_error(y_reg, rf_margin)
    mae_xgb = mean_absolute_error(y_reg, xgb_margin)
    mae_nn = mean_absolute_error(y_reg, nn_margin)
    mae_ens = mean_absolute_error(y_reg, ens_margin)

    rmse_rf = np.sqrt(mean_squared_error(y_reg, rf_margin))
    rmse_xgb = np.sqrt(mean_squared_error(y_reg, xgb_margin))
    rmse_nn = np.sqrt(mean_squared_error(y_reg, nn_margin))
    rmse_ens = np.sqrt(mean_squared_error(y_reg, ens_margin))

    # --- CV margin RMSE ---
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        cv_rmse_rf = np.sqrt(-cross_val_score(
            RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=5,
                                  min_samples_split=10, random_state=SEED, n_jobs=-1),
            X_scaled, y_reg, cv=CV_FOLDS, scoring="neg_mean_squared_error",
        ).mean())
        cv_rmse_xgb = np.sqrt(-cross_val_score(
            XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                         reg_lambda=1.0, objective="reg:squarederror",
                         random_state=SEED, n_jobs=-1, verbosity=0),
            X_scaled, y_reg, cv=CV_FOLDS, scoring="neg_mean_squared_error",
        ).mean())
        cv_rmse_nn = np.sqrt(-cross_val_score(
            MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                         solver="adam", alpha=0.001, max_iter=1000,
                         early_stopping=True, validation_fraction=0.15,
                         random_state=SEED),
            X_scaled, y_reg, cv=CV_FOLDS, scoring="neg_mean_squared_error",
        ).mean())

    # --- Print ---
    header = f"{'Metric':<25s}  {'RF':>8s}  {'XGB':>8s}  {'NN':>8s}  {'Ensemble':>8s}"
    print(f"\n  {header}")
    print(f"  {'-'*len(header)}")

    print(f"  {'CV Accuracy':<25s}  {cv_acc_rf:8.4f}  {cv_acc_xgb:8.4f}  {cv_acc_nn:8.4f}  {'—':>8s}")
    print(f"  {'Train Accuracy':<25s}  {train_acc_rf:8.4f}  {train_acc_xgb:8.4f}  {train_acc_nn:8.4f}  {train_acc_ens:8.4f}")
    gap_rf = train_acc_rf - cv_acc_rf
    gap_xgb = train_acc_xgb - cv_acc_xgb
    gap_nn = train_acc_nn - cv_acc_nn
    print(f"  {'Overfit gap (Tr-CV)':<25s}  {gap_rf:+8.4f}  {gap_xgb:+8.4f}  {gap_nn:+8.4f}  {'—':>8s}")
    print()
    print(f"  {'Brier (train, lower=better)':<25s}  {brier_rf:8.4f}  {brier_xgb:8.4f}  {brier_nn:8.4f}  {brier_ens:8.4f}")
    print(f"  {'Log-loss (train, lower=b.)':<25s}  {logloss_rf:8.4f}  {logloss_xgb:8.4f}  {logloss_nn:8.4f}  {logloss_ens:8.4f}")
    print()
    print(f"  {'Margin MAE (train)':<25s}  {mae_rf:8.2f}  {mae_xgb:8.2f}  {mae_nn:8.2f}  {mae_ens:8.2f}")
    print(f"  {'Margin MAE (CV)':<25s}  {cv_mae_rf:8.2f}  {cv_mae_xgb:8.2f}  {cv_mae_nn:8.2f}  {'—':>8s}")
    print(f"  {'Margin RMSE (train)':<25s}  {rmse_rf:8.2f}  {rmse_xgb:8.2f}  {rmse_nn:8.2f}  {rmse_ens:8.2f}")
    print(f"  {'Margin RMSE (CV)':<25s}  {cv_rmse_rf:8.2f}  {cv_rmse_xgb:8.2f}  {cv_rmse_nn:8.2f}  {'—':>8s}")

    return {
        "n_train_samples": len(y_cls),
        "rf_cv_accuracy": cv_acc_rf, "xgb_cv_accuracy": cv_acc_xgb, "nn_cv_accuracy": cv_acc_nn,
        "rf_train_accuracy": train_acc_rf, "xgb_train_accuracy": train_acc_xgb,
        "nn_train_accuracy": train_acc_nn, "ensemble_train_accuracy": train_acc_ens,
        "rf_brier": brier_rf, "xgb_brier": brier_xgb, "nn_brier": brier_nn, "ensemble_brier": brier_ens,
        "rf_logloss": logloss_rf, "xgb_logloss": logloss_xgb, "nn_logloss": logloss_nn, "ensemble_logloss": logloss_ens,
        "rf_margin_mae": mae_rf, "xgb_margin_mae": mae_xgb, "nn_margin_mae": mae_nn, "ensemble_margin_mae": mae_ens,
        "rf_margin_rmse": rmse_rf, "xgb_margin_rmse": rmse_xgb, "nn_margin_rmse": rmse_nn, "ensemble_margin_rmse": rmse_ens,
        "rf_cv_mae": cv_mae_rf, "xgb_cv_mae": cv_mae_xgb, "nn_cv_mae": cv_mae_nn,
        "rf_cv_rmse": cv_rmse_rf, "xgb_cv_rmse": cv_rmse_xgb, "nn_cv_rmse": cv_rmse_nn,
        "overfit_gap_rf": gap_rf, "overfit_gap_xgb": gap_xgb, "overfit_gap_nn": gap_nn,
    }


def temporal_validation(all_seasons_data, label=""):
    """Train on seasons 2021-2023, test on 2024. Train on 2021-2024, test on 2025."""
    print(f"\n{'='*75}")
    print(f"  TEMPORAL VALIDATION (train on past, test on future)")
    print(f"{'='*75}")

    splits = [
        ("Train 2021-2023 -> Test 2024", [2021, 2022, 2023], [2024]),
        ("Train 2021-2024 -> Test 2025", [2021, 2022, 2023, 2024], [2025]),
        ("Train 2023-2024 -> Test 2025 (old scope)", [2023, 2024], [2025]),
    ]

    for split_name, train_seasons, test_seasons in splits:
        train_data = {s: all_seasons_data[s] for s in train_seasons if s in all_seasons_data}
        test_data = {s: all_seasons_data[s] for s in test_seasons if s in all_seasons_data}

        if not train_data or not test_data:
            print(f"\n  {split_name}: skipped (missing data)")
            continue

        train_df = build_training_dataset(train_data)
        test_df = build_training_dataset(test_data)

        if train_df.empty or test_df.empty:
            print(f"\n  {split_name}: skipped (empty dataset)")
            continue

        X_train = train_df[FEATURE_COLS].values.astype(float)
        y_cls_train = train_df["home_win"].values.astype(int)
        y_reg_train = train_df["margin"].values.astype(float)
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

        X_test = test_df[FEATURE_COLS].values.astype(float)
        y_cls_test = test_df["home_win"].values.astype(int)
        y_reg_test = test_df["margin"].values.astype(float)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        rf_clf, rf_reg, xgb_clf, xgb_reg, nn_clf, nn_reg = build_models()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            rf_clf.fit(X_train_sc, y_cls_train)
            rf_reg.fit(X_train_sc, y_reg_train)
            xgb_clf.fit(X_train_sc, y_cls_train)
            xgb_reg.fit(X_train_sc, y_reg_train)
            nn_clf.fit(X_train_sc, y_cls_train)
            nn_reg.fit(X_train_sc, y_reg_train)

        rf_p = rf_clf.predict_proba(X_test_sc)[:, 1]
        xgb_p = xgb_clf.predict_proba(X_test_sc)[:, 1]
        nn_p = nn_clf.predict_proba(X_test_sc)[:, 1]
        ens_p = (rf_p + xgb_p + nn_p) / 3.0

        rf_m = rf_reg.predict(X_test_sc)
        xgb_m = xgb_reg.predict(X_test_sc)
        nn_m = nn_reg.predict(X_test_sc)
        ens_m = (rf_m + xgb_m + nn_m) / 3.0

        acc_rf = accuracy_score(y_cls_test, (rf_p > 0.5).astype(int))
        acc_xgb = accuracy_score(y_cls_test, (xgb_p > 0.5).astype(int))
        acc_nn = accuracy_score(y_cls_test, (nn_p > 0.5).astype(int))
        acc_ens = accuracy_score(y_cls_test, (ens_p > 0.5).astype(int))

        mae_rf = mean_absolute_error(y_reg_test, rf_m)
        mae_xgb = mean_absolute_error(y_reg_test, xgb_m)
        mae_nn = mean_absolute_error(y_reg_test, nn_m)
        mae_ens = mean_absolute_error(y_reg_test, ens_m)

        brier_ens = brier_score_loss(y_cls_test, ens_p)

        print(f"\n  {split_name}")
        print(f"  Train: {len(y_cls_train)} games | Test: {len(y_cls_test)} games")
        print(f"  Test Accuracy   RF: {acc_rf:.4f}  XGB: {acc_xgb:.4f}  NN: {acc_nn:.4f}  Ens: {acc_ens:.4f}")
        print(f"  Test Margin MAE RF: {mae_rf:.2f}    XGB: {mae_xgb:.2f}    NN: {mae_nn:.2f}    Ens: {mae_ens:.2f}")
        print(f"  Test Brier Ens: {brier_ens:.4f}")


def comparison(old, new):
    """Print old vs new comparison."""
    print(f"\n{'='*75}")
    print(f"  COMPARISON: 3 SEASONS (OLD) vs 5 SEASONS (NEW)")
    print(f"{'='*75}")

    rows = [
        ("Samples", "n_train_samples", "{:.0f}"),
        ("RF CV Accuracy", "rf_cv_accuracy", "{:.4f}"),
        ("XGB CV Accuracy", "xgb_cv_accuracy", "{:.4f}"),
        ("NN CV Accuracy", "nn_cv_accuracy", "{:.4f}"),
        ("RF Train Accuracy", "rf_train_accuracy", "{:.4f}"),
        ("XGB Train Accuracy", "xgb_train_accuracy", "{:.4f}"),
        ("NN Train Accuracy", "nn_train_accuracy", "{:.4f}"),
        ("Ensemble Train Acc", "ensemble_train_accuracy", "{:.4f}"),
        ("RF Overfit gap", "overfit_gap_rf", "{:+.4f}"),
        ("XGB Overfit gap", "overfit_gap_xgb", "{:+.4f}"),
        ("NN Overfit gap", "overfit_gap_nn", "{:+.4f}"),
        ("Ensemble Brier (lower=b.)", "ensemble_brier", "{:.4f}"),
        ("Ensemble Margin MAE(lower)", "ensemble_margin_mae", "{:.2f}"),
        ("Ensemble Margin RMSE(low.)", "ensemble_margin_rmse", "{:.2f}"),
    ]

    print(f"\n  {'Metric':<25s}  {'Old (3s)':>10s}  {'New (5s)':>10s}  {'Delta':>10s}  {'Note':>12s}")
    print(f"  {'-'*75}")

    for name, key, fmt in rows:
        ov = old.get(key)
        nv = new.get(key)
        if ov is None and nv is None:
            continue
        o_str = fmt.format(ov) if ov is not None else "—"
        n_str = fmt.format(nv) if nv is not None else "—"

        if ov is not None and nv is not None:
            delta = nv - ov
            d_str = f"{delta:+.4f}"
            if "overfit" in key.lower() or "gap" in key.lower():
                note = "better" if delta < 0 else "worse"
            elif "accuracy" in key.lower() or "acc" in key.lower():
                note = "better" if delta > 0 else "worse"
            else:
                note = "better" if delta < 0 else "worse"
        else:
            d_str = "—"
            note = "new"

        print(f"  {name:<25s}  {o_str:>10s}  {n_str:>10s}  {d_str:>10s}  {note:>12s}")


if __name__ == "__main__":
    print("Loading season data ...")
    all_seasons = load_seasons([2021, 2022, 2023, 2024, 2025])
    print(f"Loaded {len(all_seasons)} seasons: {sorted(all_seasons.keys())}")

    # OLD: 3 seasons (2023-2025) — just CV metrics from existing metadata
    old_3s = load_seasons([2023, 2024, 2025])
    print("\n" + "#"*75)
    print("#  A. OLD SETUP: 3 Seasons (2023-2025)")
    print("#"*75)
    old_metrics = evaluate_dataset(
        build_training_dataset(old_3s),
        label="3 SEASONS (2023, 2024, 2025) — baseline"
    )

    # NEW: 5 seasons (2021-2025)
    print("\n" + "#"*75)
    print("#  B. NEW SETUP: 5 Seasons (2021-2025)")
    print("#"*75)
    new_metrics = evaluate_dataset(
        build_training_dataset(all_seasons),
        label="5 SEASONS (2021, 2022, 2023, 2024, 2025) — expanded"
    )

    # Temporal validation
    print("\n" + "#"*75)
    print("#  C. TEMPORAL VALIDATION")
    print("#"*75)
    temporal_validation(all_seasons)

    # Comparison
    comparison(old_metrics, new_metrics)

    print(f"\n{'='*75}")
    print("  DONE")
    print(f"{'='*75}\n")
