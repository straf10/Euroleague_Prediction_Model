# Euroleague Prediction Model

## What this project does

This repo is a **EuroLeague basketball prediction pipeline**. It pulls game and box-score data via the EuroLeague API, builds team-level features (Elo, four-factors-style matchups, schedule context), and trains **two linear models**: a **logistic regression** for home-win probability and a **ridge regression** for expected point margin. Predictions combine those outputs with a **Monte Carlo** simulation over margin. You run everything through one CLI command, `euroleague-sim`.

## Conclusion

In the EuroLeague, Home Court Advantage is fundamentally a defensive phenomenon. Home teams don't necessarily win because they shoot the lights out; they win because the home crowd and familiarity allow them to play suffocating defense, force the away team into turnovers, and secure defensive rebounds. This is a brilliant, mathematically proven insight that you completely unlocked by simplifying the model!

## Training diagnostics

The figure below is produced when you run `euroleague-sim train` (saved as `plots/training_diagnostics.png`). It shows logistic coefficients for the ten features, feature correlations, out-of-fold probability calibration, and ridge margin fit—supporting the conclusion above.

![Euroleague ML training diagnostics](./plots/training_diagnostics.png)

If the preview is blank, confirm `plots/training_diagnostics.png` exists (run `euroleague-sim train`). The PNG is tracked in git so it also renders on GitHub after you push.

## The 6 features

These are the columns fed to the ML models (see `src/euroleague_sim/ml/features.py`, `FEATURE_COLS`):

1. `elo_diff_scaled`
2. `net_efg`
3. `net_tov`
4. `net_orb`
5. `net_ftr`
6. `round_progress`

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
