# Euroleague Prediction Model

A machine learning prediction system for **EuroLeague basketball** (current season: 2025-26, `seasonCode E2025`).

Combines three ML models in a weighted ensemble with Monte Carlo simulations to produce win probabilities and margin distributions for upcoming games.

## Model Architecture

| Component | Description |
|---|---|
| **Random Forest** | Classifier + regressor (300 trees, max depth 6) for baseline predictions |
| **XGBoost** | Gradient-boosted classifier + regressor (300 rounds, learning rate 0.05) for capturing non-linear feature interactions |
| **Neural Network** | MLP classifier + regressor (64-32 hidden layers, early stopping) for learning complex patterns |
| **Ensemble** | Weighted average of all three models (RF 35%, XGBoost 35%, NN 30%) |
| **Monte Carlo** | `margin ~ Normal(mu, sigma)` with 20,000 simulations per game for margin distributions |

### Features (20 total)

- **Differential features**: net rating diff, Elo diff, offensive/defensive matchups, win% diff, form (last 5 games), pace diff
- **Four Factors differentials**: eFG%, turnover rate, offensive rebound rate, free-throw rate
- **Absolute ratings**: offensive/defensive ratings and win% for both home and away teams, Elo ratings
- **Context**: round progress (current round / total rounds)

### Supporting Components

| Component | Description |
|---|---|
| **Net Rating** | Offensive/Defensive rating per 100 possessions with home/away splits and early-season shrinkage |
| **Elo** | Historic prior blended from past 2 seasons (65/35 weight), updated game-by-game in the current season |
| **Possessions** | Estimated via `FGA + 0.44*FTA - OREB + TOV` per team-game |

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

### Dependencies

- `euroleague-api` — EuroLeague API wrapper for data fetching
- `pandas`, `numpy` — data manipulation
- `scikit-learn` — Random Forest and Neural Network models
- `xgboost` — gradient boosting model
- `scipy` — statistical functions
- `joblib` — model serialisation

## Quick Start

### 1. Fetch and cache data

Downloads the current season plus 2 prior seasons (needed for Elo historic prior):

```bash
euroleague-sim update-data --season 2025
```

### 2. Train ML models

Trains Random Forest + XGBoost + Neural Network on all available seasons:

```bash
euroleague-sim train --season 2025
```

### 3. Predict next round

Automatically detects the next unplayed round:

```bash
euroleague-sim predict --season 2025 --round next
```

Predict a specific round with custom simulation count:

```bash
euroleague-sim predict --season 2025 --round 22 --n-sims 50000
```

## Season Codes

- **2025** = current season 2025-26 (`seasonCode E2025`)
- Elo computation requires the 2 preceding seasons (2023, 2024), fetched automatically with `--history 2`

## Configuration

Export and customise all model parameters:

```bash
euroleague-sim --dump-config config.json
# Edit config.json (Elo weights, MC parameters, ML hyperparameters, ensemble weights, etc.)
euroleague-sim --config config.json train --season 2025
euroleague-sim --config config.json predict --season 2025
```

### Configurable Parameters

| Section | Parameters |
|---|---|
| **Elo** | base rating, K-factor, home advantage, blend weights |
| **Monte Carlo** | alpha1-3 coefficients, sigma, number of simulations |
| **Shrinkage** | k_games for early-season rating stabilisation |
| **ML** | RF/XGBoost/NN hyperparameters, ensemble weights, CV folds |

## Output

Predictions are saved to `outputs/round_{R}_predictions.csv`.

Key columns:
- `pHomeWin_ml` — ensemble win probability (primary prediction)
- `pHomeWin_rf`, `pHomeWin_xgb`, `pHomeWin_nn` — individual model probabilities
- `pHomeWin` — Monte Carlo win probability
- `q10`, `q50`, `q90` — margin distribution quantiles
- `EloCurrent_home`, `EloCurrent_away` — current Elo ratings

## Data Cache

Raw and processed data are stored in `./data_cache/` as pickle and JSON files. Re-run `update-data` to refresh.

## Project Structure

```
src/euroleague_sim/
  __init__.py
  config.py            # all parameters (Elo, MC, shrinkage, ML hyperparameters)
  pipeline.py          # orchestrates: fetch -> features -> Elo -> train -> predict
  cli.py               # command-line interface (update-data, train, predict)
  data/
    fetch.py           # euroleague-api wrapper + v3 HTTP fallback
    cache.py           # pickle/JSON filesystem cache
  features/
    possessions.py     # team possessions from boxscore data
    net_rating.py      # OffRtg/DefRtg/NetRtg per 100 poss + home/away splits + shrinkage
    elo.py             # Elo engine: historic blend, current season update
  ml/
    features.py        # feature engineering (20 features, point-in-time for training)
    train.py           # train RF + XGBoost + NN (classifier + regressor each)
    predict.py         # EnsemblePredictor: load models + weighted ensemble inference
  sim/
    model.py           # matchup features (A, B) from net ratings and Elo
    engine.py          # Monte Carlo margin simulation (Normal distribution)
models/                # persisted model artefacts (.joblib) + metadata.json
outputs/               # prediction CSVs per round
data_cache/            # cached raw + processed data
```

## Pipeline Flow

```
1. update-data     Fetch raw boxscore + gamecodes from EuroLeague API
                   Build possessions -> games -> team ratings -> Elo
                   Cache everything to data_cache/

2. train           Load 3 seasons of cached features
                   Build point-in-time training dataset (no lookahead)
                   Train RF + XGBoost + NN (classifier + regressor)
                   Evaluate with cross-validation
                   Save models to models/

3. predict         Load current season features + Elo ratings
                   Fetch schedule for target round
                   Build prediction features from current stats
                   Run 3-model ensemble for P(HomeWin) + expected margin
                   Run Monte Carlo for margin distribution
                   Save to outputs/
```
