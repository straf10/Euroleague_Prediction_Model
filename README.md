# Euroleague Prediction Model

Baseline prediction system for EuroLeague basketball (current season example: `E2025`).

The current ML pipeline is strictly linear:
- `LogisticRegression` predicts home win probability (`pHomeWin_ml`)
- `Ridge` predicts expected margin (`margin_ml`)
- Monte Carlo converts margin assumptions into probability and quantiles

## Model Architecture

| Component | Description |
|---|---|
| **Logistic Regression** | Binary classifier for home win probability |
| **Ridge Regression** | Linear regressor for home-away point margin |
| **Monte Carlo** | `margin ~ Normal(mu, sigma)` with configurable simulation count |

### Features (16 total)

- **Differential:** `net_rtg_diff`, `elo_diff_scaled`, `off_matchup`, `def_matchup`, `win_pct_diff`, `form_diff`, `pace_diff`
- **Home absolute:** `home_off_rtg`, `home_def_rtg`, `home_win_pct`, `elo_home`
- **Away absolute:** `away_off_rtg`, `away_def_rtg`, `away_win_pct`, `elo_away`
- **Context:** `round_progress`

### Supporting Components

| Component | Description |
|---|---|
| **Net Rating** | Offensive/Defensive ratings per 100 possessions with home/away splits and early-season shrinkage |
| **Elo** | Historic prior from past seasons, then updated game-by-game in current season |
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

- `euroleague-api` for data fetching
- `pandas`, `numpy` for data processing
- `scikit-learn` for Logistic Regression, Ridge, scaling, and metrics
- `scipy` for statistics utilities
- `joblib` for model persistence

## Quick Start

### 1. Fetch and cache data

Downloads the selected season plus prior seasons (default `--history 2`):

```bash
euroleague-sim update-data --season 2025
```

### 2. Train models

```bash
euroleague-sim train --season 2025
```

### 3. Predict next round

```bash
euroleague-sim predict --season 2025 --round next
```

Predict a specific round with custom simulation count:

```bash
euroleague-sim predict --season 2025 --round 22 --n-sims 50000
```

## Configuration

Generate and customize config values:

```bash
euroleague-sim --dump-config config.json
euroleague-sim --config config.json train --season 2025
euroleague-sim --config config.json predict --season 2025
```

### Important: reset old local configs

If you generated `config.json` before the linear baseline migration, it may still include removed ML keys. Regenerate it before running with `--config`:

```bash
rm -f config.json
euroleague-sim --dump-config config.json
```

### Configurable Parameters

| Section | Parameters |
|---|---|
| **Elo** | `base`, `k`, `home_advantage`, `blend_recent`, `blend_older` |
| **Monte Carlo** | `alpha1`, `alpha2`, `alpha3`, `sigma`, `n_sims` |
| **Shrinkage** | `k_games` |
| **ML** | `logreg_C`, `logreg_max_iter`, `ridge_alpha`, `cv_folds`, `model_dir` |

## Output

Predictions are saved to `outputs/round_{R}_predictions.csv`.

Key columns:
- `pHomeWin_ml` — Logistic Regression home win probability
- `margin_ml` — Ridge expected margin
- `pHomeWin` — Monte Carlo home win probability
- `q10`, `q50`, `q90` — simulated margin quantiles
- `EloCurrent_home`, `EloCurrent_away` — current Elo ratings

## Data Cache

Raw and processed data are cached in `data_cache/` as pickle and JSON files.
Use `update-data --force` to re-download and rebuild.

## Project Structure

```
src/euroleague_sim/
  config.py            # project parameters (Elo, MC, shrinkage, ML)
  pipeline.py          # fetch -> features -> Elo -> train -> predict orchestration
  cli.py               # commands: update-data, train, predict
  data/
    fetch.py           # euroleague-api wrapper + v3 HTTP fallback
    cache.py           # pickle/JSON filesystem cache
  features/
    possessions.py     # team possessions from boxscore data
    net_rating.py      # OffRtg/DefRtg/NetRtg per 100 + splits + shrinkage
    elo.py             # Elo engine: historic blend + current-season updates
  ml/
    features.py        # 16-feature engineering for train/predict modes
    train.py           # train Logistic Regression + Ridge
    predict.py         # LinearPredictor: load and infer with saved models
  sim/
    model.py           # matchup features (A, B) from ratings and Elo
    engine.py          # Monte Carlo margin simulation
models/                # scaler.joblib, logreg.joblib, ridge.joblib, metadata.json
outputs/               # prediction CSVs
data_cache/            # cached raw + processed datasets
```

## Pipeline Flow

```
1. update-data     Fetch raw gamecodes + player boxscore
                   Build possessions -> games -> team ratings
                   Cache artefacts in data_cache/

2. train           Load cached historical features
                   Build point-in-time dataset (no lookahead)
                   Train Logistic Regression + Ridge
                   Evaluate with cross-validation
                   Save artefacts to models/

3. predict         Load current features + Elo
                   Fetch target round schedule
                   Build prediction feature matrix
                   Predict pHomeWin_ml + margin_ml
                   Run Monte Carlo simulation
                   Save CSV to outputs/
```
