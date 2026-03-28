# Euroleague Prediction Model

## What this project does

This repo is a **EuroLeague basketball prediction pipeline**. It pulls game and box-score data via the EuroLeague API, builds team-level features (Elo, 5-game weighted recent form on four-factor rates, schedule context), and trains **two linear models**: a **logistic regression** for home-win probability and a **ridge regression** for expected point margin. Predictions combine those outputs with a **Monte Carlo** simulation over margin. You run everything through one CLI command, `euroleague-sim`.

## Conclusion

In the EuroLeague, Home Court Advantage is fundamentally a defensive phenomenon. Home teams don't necessarily win because they shoot the lights out; they win because the home crowd and familiarity allow them to play suffocating defense, force the away team into turnovers, and secure defensive rebounds. This is a brilliant, mathematically proven insight that you completely unlocked by simplifying the model!

## Training diagnostics

The figure below is produced when you run `euroleague-sim train` (saved as `plots/training_diagnostics.png`). It shows logistic coefficients for the six features, feature correlations, out-of-fold probability calibration, and ridge margin fit—supporting the conclusion above.

![Euroleague ML training diagnostics](./plots/training_diagnostics.png)



## The 6 features

These are the columns fed to the ML models (see `src/euroleague_sim/ml/features.py`, `FEATURE_COLS`). The model is intentionally lean: **identity** (Elo + where you are in the season) plus **recent form** (home-minus-away differentials on four-factor rates, each a 5-game linear WMA with the **newest** game weighted highest).

1. `elo_diff_scaled` — (Elo_home − Elo_away) / 25
2. `net_efg_wma5` — 5-game WMA recent form: eFG% (home − away); in training, `shift(1)` excludes the current game from the rolling window
3. `net_tov_wma5` — 5-game WMA recent form: TOV% (home − away)
4. `net_orb_wma5` — 5-game WMA recent form: ORB% (home − away)
5. `net_ftr_wma5` — 5-game WMA recent form: FT rate (home − away)
6. `round_progress` — round / max_rounds

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
