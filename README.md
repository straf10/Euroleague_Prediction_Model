# Euroleague Prediction Model

## What this project does

This repo is a **EuroLeague basketball prediction pipeline**. It pulls game and box-score data via the EuroLeague API, builds team-level features (Elo, 5-game weighted recent form on four-factor rates, round progress, and **EuroLeague schedule rest**—capped days since the previous EL game, home minus away), and trains **two linear models**: a **logistic regression** for home-win probability and a **ridge regression** for expected point margin. Predictions combine those outputs with a **Monte Carlo** simulation over margin. You run everything through one CLI command, `euroleague-sim`.

## Conclusion

After Elo, the logistic model leans most on **rebounding and ball security**: a positive weight on `net_orb_wma5` and a negative weight on `net_tov_wma5` mean that, holding Elo fixed, home teams look better when they crash the glass and take care of the ball relative to the visitor—consistent with defense and possession quality mattering beyond raw shooting. **Effective shooting and free-throw rate** (`net_efg_wma5`, `net_ftr_wma5`) still help, but rank behind ORB/TOV in this fit. **`el_rest_days_diff`** adds a small positive tilt when the home side has more EL rest than the away side; **`round_progress`** is effectively flat here, so time-in-season is mostly handled elsewhere. See **Training diagnostics** (below the feature list and figure) for the latest numbers from your machine.

## The 7 features

These are the columns fed to the ML models (see `src/euroleague_sim/ml/features.py`, `FEATURE_COLS`). The model is intentionally lean: **identity** (Elo + where you are in the season) plus **recent form** (home-minus-away differentials on four-factor rates, each a 5-game linear WMA with the **newest** game weighted highest), plus **schedule density** (rest gap in the EuroLeague calendar).

1. `elo_diff_scaled` — (Elo_home − Elo_away) / 25
2. `net_efg_wma5` — 5-game WMA recent form: eFG% (home − away); in training, `shift(1)` excludes the current game from the rolling window
3. `net_tov_wma5` — 5-game WMA recent form: TOV% (home − away)
4. `net_orb_wma5` — 5-game WMA recent form: ORB% (home − away)
5. `net_ftr_wma5` — 5-game WMA recent form: FT rate (home − away)
6. `round_progress` — round / max_rounds
7. `el_rest_days_diff` — capped EL rest days for home minus away (days since previous EuroLeague game, clipped; uses `game_date` on team-game rows)

![Euroleague ML training diagnostics](./plots/training_diagnostics.png)

## Training diagnostics

The figure above is produced when you run `euroleague-sim train` (saved as `plots/training_diagnostics.png`). It shows logistic coefficients for the seven features, feature correlations, out-of-fold probability calibration, and ridge margin fit.

Example run (**969 games, 7 features**):

| Metric | Value |
| --- | --- |
| Best LogisticRegression C | 0.3 |
| Best Ridge α | 75 |
| Cross-val accuracy (LogReg) | 0.642 |
| Train accuracy (LogReg) | 0.655 |
| Brier score (LogReg) | 0.2180 |
| Log-loss (LogReg) | 0.6255 |
| Margin MAE (Ridge) | 9.19 |
| Margin RMSE (Ridge) | 11.90 |

Logistic regression coefficients (sorted by \|weight\|, descending):

| Feature | Coefficient |
| --- | ---: |
| `elo_diff_scaled` | +0.4467 |
| `net_orb_wma5` | +0.1497 |
| `net_tov_wma5` | −0.1116 |
| `net_efg_wma5` | +0.0827 |
| `net_ftr_wma5` | +0.0498 |
| `el_rest_days_diff` | +0.0400 |
| `round_progress` | +0.0004 |

## How to run

Create a virtual environment, install dependencies and the package in editable mode, then run the three steps in order.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

```bash
euroleague-sim update-data --season 2025
euroleague-sim train --season 2025
euroleague-sim predict --season 2025 --round next
```

To re-fetch data and refresh the cache, use:

```bash
euroleague-sim update-data --season 2025 --force
```

Optional: predict a fixed round or change simulation count:

```bash
euroleague-sim predict --season 2025 --round 22 --n-sims 50000
```
