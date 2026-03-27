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

### Features (10 total)

Defined in `src/euroleague_sim/ml/features.py` as `FEATURE_COLS`:

- **Rating gap:** `elo_diff_scaled`
- **Four-factors matchups (home offense vs away defense):** `home_off_efg_matchup`, `home_off_tov_matchup`, `home_off_orb_matchup`, `home_off_ftr_matchup`
- **Four-factors matchups (away offense vs home defense):** `away_off_efg_matchup`, `away_off_tov_matchup`, `away_off_orb_matchup`, `away_off_ftr_matchup`
- **Context:** `round_progress`

### Supporting Components

| Component | Description |
|---|---|
| **Net Rating** | Offensive/Defensive ratings per 100 possessions with home/away splits and early-season shrinkage |
| **Elo** | Historic prior from past seasons, then updated game-by-game in current season |
| **Possessions** | Estimated via `FGA + 0.44*FTA - OREB + TOV` per team-game |

## Conclusion from current training data

**Home court in EuroLeague, as read through this model on the data we actually trained on, looks fundamentally like a defensive story.** Home teams do not need to “shoot the lights out” to carry an edge. The fitted linear models assign substantial weight to matchup structure where the home side’s environment and familiarity show up as **pressure on the visitor**: higher turnover pressure, contested shots (eFG defense), and control of the glass (offensive vs defensive rebounding matchups)—the same mechanisms fans describe as “suffocating defense,” now visible in standardized coefficients rather than only in narrative.

Simplifying the feature set makes this easier to see: with fewer, interpretable inputs, the logistic head’s weights highlight **which** stable team traits move `P(home win)`, and the four-factors-style matchup columns are where defensive leverage lives. That is an insight **supported by the current training window and diagnostics below**, not an axiom; retrain after major rule or league-quality shifts and re-check the coefficient panel.

## Training diagnostics (evidence)

After `euroleague-sim train`, the pipeline writes **`plots/training_diagnostics.png`** (see [`src/euroleague_sim/ml/plots.py`](src/euroleague_sim/ml/plots.py)). Use it to sanity-check both model behaviour and the home-court hypothesis:

| Panel | What it shows | Why it matters for the hypothesis |
|---|---|---|
| **Top-left** | Logistic regression coefficients on **standardized** features (green → higher `P(home win)`, red → lower). | **Primary evidence:** large-magnitude bars on turnover, rebounding, and eFG **matchup** features indicate that *defensive* execution and pressure (not raw home offensive explosion) dominate the linear signal for who wins. |
| **Top-right** | Pearson correlation heatmap across training features. | Shows redundancy and shared variance; helps confirm that a slim feature set is not hiding a single collinear “trick” feature. |
| **Bottom-left** | **Probability calibration** using **time-series cross-validated** out-of-fold probabilities. | Checks that predicted home-win rates line up with realized frequencies **without** peeking at the future—guards against a hollow “always pick home” shortcut. |
| **Bottom-right** | Ridge **actual vs predicted margin** (in-sample scatter). | Validates that the margin model is not only classifying wins but tracking point differential in a structured way. |

![Training diagnostics (2×2 figure from `train`)](plots/training_diagnostics.png)

Regenerate this file whenever you retrain so the README’s figure stays aligned with the models in `models/`.

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

Use the packaged CLI as the single entry point (`euroleague-sim`); no separate wrapper scripts are required.

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

## Pipeline Flow

```
1. update-data     Fetch raw gamecodes + player boxscore
                   Build possessions -> games -> team ratings
                   Cache artefacts in data_cache/

2. train           Load cached historical features
                   Build point-in-time dataset (no lookahead)
                   Train Logistic Regression + Ridge
                   Evaluate with TimeSeriesSplit cross-validation
                   Save models to models/; diagnostics PNG to plots/

3. predict         Load current features + Elo
                   Fetch target round schedule
                   Build prediction feature matrix
                   Predict pHomeWin_ml + margin_ml
                   Run Monte Carlo simulation
                   Save CSV to outputs/
```
