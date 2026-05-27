# EuroLeague Prediction Model

A machine-learning pipeline for predicting EuroLeague Basketball game outcomes — win probabilities and point margins — using historical box-score data, Elo ratings, and four-factor team-efficiency features.

---

## Architecture

```
EuroLeague API
     │
     ▼
Data Fetching & Caching          euroleague-sim update-data
     │  raw box-scores, game codes, club info → data_cache/
     ▼
Feature Engineering
     │  Elo ratings · 5-game WMA four-factor form · schedule rest · round progress
     ▼
ML Training (model registry)     euroleague-sim train --model <name>
     │  Win probability  → CatBoostClassifier + Beta calibration
     │  Point margin     → CatBoostRegressor
     │  Evaluation       → Walk-Forward Optimization (round-by-round OOS)
     ▼
Monte Carlo Simulation
     │  n_sims draws over margin distribution → probability table
     ▼
Predictions CSV                  euroleague-sim predict --round next
```

---

## Features

Seven engineered features feed all ML models (defined in `src/euroleague_sim/ml/features.py`):

| # | Feature | Description |
|---|---------|-------------|
| 1 | `elo_diff_scaled` | (Elo_home − Elo_away) / 25 |
| 2 | `net_efg_wma5` | 5-game WMA effective FG% differential (home − away) |
| 3 | `net_tov_wma5` | 5-game WMA turnover rate differential |
| 4 | `net_orb_wma5` | 5-game WMA offensive rebound rate differential |
| 5 | `net_ftr_wma5` | 5-game WMA free-throw rate differential |
| 6 | `round_progress` | Current round / total rounds |
| 7 | `el_rest_days_diff` | Capped EuroLeague rest days (home − away) |

All rolling windows use `shift(1)` during training — the current game is excluded from its own feature window, eliminating data leakage.

---

## Models

Two entries are registered in `src/euroleague_sim/ml/registry.py`:

### CatBoost *(primary)*

| Component | Detail |
|-----------|--------|
| Win model | `CatBoostClassifier` (Logloss) + **Beta calibration** |
| Margin model | `CatBoostRegressor` (RMSE) |
| Evaluation | **Walk-Forward Optimization** — round-by-round expanding window, 107 OOS folds |
| Scaling | None — tree-based models are scale-invariant |
| Device | CPU (GPU is 4× slower at this dataset size) |

Beta calibration replaces Platt/sigmoid scaling: it fits a 3-parameter Beta-distribution link on a held-out chronological calibration tail, correcting both probability tails without temporal leakage.

### Baseline

| Component | Detail |
|-----------|--------|
| Win model | `LogisticRegression` + Platt scaling (`CalibratedClassifierCV`) |
| Margin model | `Ridge` regression |
| Evaluation | `TimeSeriesSplit` (5 folds) |
| Scaling | `StandardScaler` |

---

## Performance

Evaluation schemes differ: baseline uses `TimeSeriesSplit` (5 folds); CatBoost uses the stricter round-by-round WFO (107 OOS folds). Dataset: ~1,035 games across 3 seasons (2023–2025).

| Metric | Baseline (LogReg + CV) | CatBoost (Beta-cal + WFO) |
|--------|----------------------|--------------------------|
| OOS Accuracy | 0.640 | 0.609 |
| Brier score | 0.2238 | 0.2332 |
| Log-loss | 0.6384 | 0.6593 *(untuned)* |
| Margin MAE | 9.44 pts | 9.53 pts *(untuned)* |
| Margin RMSE | 12.12 pts | 12.27 pts *(untuned)* |

> CatBoost metrics reflect untuned defaults. Offline Optuna tuning (see §Tuning below) targets WFO log-loss directly and closes the gap.

---

## Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

---

## Usage

### Fetch & cache data

```bash
euroleague-sim update-data --season 2025
# Force re-download:
euroleague-sim update-data --season 2025 --force
```

### Train a model

```bash
# Train CatBoost (saves to models/catboost/)
euroleague-sim train --season 2025 --model catboost

# Train baseline for comparison
euroleague-sim train --season 2025 --model baseline

# Leaderboard — evaluate all registered models, print comparison table
euroleague-sim train --season 2025 --model all
```

### Predict

```bash
# Next unplayed round
euroleague-sim predict --season 2025 --round next --model catboost

# Specific round with more Monte Carlo draws
euroleague-sim predict --season 2025 --round 22 --model catboost --n-sims 50000

# Write to CSV
euroleague-sim predict --season 2025 --round next --model catboost --out outputs/round_22.csv
```

### Optional config override

```bash
# Use a custom config file
euroleague-sim --config my_config.json train --season 2025 --model catboost

# Dump default config
euroleague-sim --dump-config default_config.json
```

---

## Hyperparameter Tuning

Tuning is intentionally isolated from the daily pipeline. Run Optuna studies offline, then hardcode the best parameters into `registry.py`.

```bash
# Run all three studies (LogReg, Ridge, CatBoost) — 100 trials, coarse WFO step
python -m euroleague_sim.ml.tune --trials 100 --wfo-step 5 --season 2025

# Run CatBoost study only — 200 trials, full round-by-round WFO (most precise)
python -m euroleague_sim.ml.tune --trials 200 --wfo-step round --season 2025 --study 3

# Run studies 1 and 2 only
python -m euroleague_sim.ml.tune --trials 100 --study 1,2
```

`--wfo-step` controls the walk-forward granularity during tuning:

| `--wfo-step` | WFO folds | Use case |
|---|---|---|
| `round` | 107 | Most precise, slowest |
| `5` | ~23 | Balanced speed / precision |
| `season` | 2 | Fast sweep (~53× fewer refits) |

---

## Project Structure

```
src/euroleague_sim/
├── cli.py                  Entry point (argparse)
├── config.py               ProjectConfig dataclass
├── pipeline.py             Shared training-data preparation
├── data/
│   ├── fetch.py            EuroLeague API client
│   └── cache.py            Disk-based feature cache
├── features/
│   ├── elo.py              Elo rating computation
│   ├── net_rating.py       Four-factor WMA features
│   ├── possessions.py      Possession estimation
│   └── context.py          Schedule rest, round progress
├── ml/
│   ├── registry.py         Model definitions & hyperparameters
│   ├── features.py         FEATURE_COLS constant
│   ├── train.py            Training & CV/WFO dispatch
│   ├── evaluate.py         walk_forward_evaluate, build_wfo_periods
│   ├── calibration.py      BetaCalibratedClassifier, fit_catboost_es
│   ├── predict.py          Inference logic
│   ├── plots.py            Diagnostics plots
│   ├── weights.py          Sample-weight utilities
│   └── tune.py             Offline Optuna tuning (isolated)
└── sim/
    ├── engine.py           Monte Carlo simulation engine
    └── model.py            Simulation data model
```

---

## Tech Stack

- **Python 3.12**
- **CatBoost** — gradient boosting with oblivious trees
- **Optuna** — hyperparameter optimisation
- **betacal** — Beta calibration
- **scikit-learn** — baseline models, cross-validation utilities
- **pandas / numpy** — data manipulation
- **EuroLeague API** (`euroleague_api`) — data source
