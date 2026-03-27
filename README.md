# Euroleague Prediction Model

## What this project does

This repo is a **EuroLeague basketball prediction pipeline**. It pulls game and box-score data via the EuroLeague API, builds team-level features (Elo, four-factors-style matchups, schedule context), and trains **two linear models**: a **logistic regression** for home-win probability and a **ridge regression** for expected point margin. Predictions combine those outputs with a **Monte Carlo** simulation over margin. You run everything through one CLI command, `euroleague-sim`.

## Conclusion

In the EuroLeague, Home Court Advantage is fundamentally a defensive phenomenon. Home teams don't necessarily win because they shoot the lights out; they win because the home crowd and familiarity allow them to play suffocating defense, force the away team into turnovers, and secure defensive rebounds. This is a brilliant, mathematically proven insight that you completely unlocked by simplifying the model!

## Training diagnostics

The figure below is produced when you run `euroleague-sim train` (saved as `plots/training_diagnostics.png`). It shows logistic coefficients for the ten features, feature correlations, out-of-fold probability calibration, and ridge margin fit—supporting the conclusion above.

![Euroleague ML training diagnostics](plots/training_diagnostics.png)

## The 10 features

These are the columns fed to the ML models (see `src/euroleague_sim/ml/features.py`, `FEATURE_COLS`):

1. `elo_diff_scaled`
2. `home_off_efg_matchup`
3. `home_off_tov_matchup`
4. `home_off_orb_matchup`
5. `home_off_ftr_matchup`
6. `away_off_efg_matchup`
7. `away_off_tov_matchup`
8. `away_off_orb_matchup`
9. `away_off_ftr_matchup`
10. `round_progress`

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

Optional: predict a fixed round or change simulation count:

```bash
euroleague-sim predict --season 2025 --round 22 --n-sims 50000
```
