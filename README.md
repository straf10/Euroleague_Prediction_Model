# Euroleague Prediction Model

Prediction model for the **current EuroLeague season (2025-26, seasonCode E2025)**.

## Model overview

| Component | Description |
|---|---|
| **Net Rating** | Offensive/Defensive rating per 100 possessions, with home/away splits and early-season shrinkage |
| **Elo** | Historic prior from past 2 seasons (0.65/0.35 blend), updated game-by-game in current season |
| **Logistic projection** | `P(HomeWin) = sigmoid(w1*A + w2*B + w3)` where A = net rating differential, B = Elo differential/25 |
| **Monte Carlo** | `margin ~ Normal(alpha1*A + alpha2*B + alpha3, sigma)` with 20,000 simulations per game |

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

## Season codes

- **2025** = φετινή σεζόν **2025-26** (seasonCode E2025). Πρέπει να κατεβάσεις δεδομένα για `--season 2025`.
- Για Elo χρειάζονται και οι 2 προηγούμενες σεζόν (2023, 2024), που κατεβαίνουν με `--history 2`.

## Quick start

1) **Πρώτα** κατέβασε και cache δεδομένα (φετινή + 2 προηγούμενες σεζόν για Elo):

```bash
euroleague-sim update-data --season 2025
```

2) Πρόβλεψη επόμενης αγωνιστικής (`next` = μικρότερη αγωνιστική με μη παιγμένα, ή max_round+1 αν το API δίνει μόνο παιγμένα):

```bash
euroleague-sim predict --season 2025 --round next
```

3) Predict a specific round with custom simulations:

```bash
euroleague-sim predict --season 2025 --round 22 --n-sims 50000
```

## Configuration

Dump default config and tweak parameters:

```bash
euroleague-sim --dump-config config.json
# edit config.json (Elo weights, projection w1/w2/w3, MC alpha1-3, sigma, etc.)
euroleague-sim --config config.json predict --season 2025
```

## Output

Predictions are saved to `outputs/round_{R}_predictions.csv` by default.

Columns: `Round, Gamecode, home_team, away_team, pHomeWin_logistic, pHomeWin, muMargin, meanMargin, q10, q50, q90, EloCurrent_home, EloCurrent_away, A, B, n_sims`

## Data cache

Raw & processed data stored in `./data_cache/` (pickle files).

## Project structure

```
src/euroleague_sim/
  config.py          # all model parameters (Elo, projection, MC, shrinkage)
  pipeline.py        # orchestrates fetch -> features -> Elo -> predict -> save
  cli.py             # command-line interface
  data/
    fetch.py         # euroleague-api wrapper + v3 HTTP
    cache.py         # pickle/JSON filesystem cache
  features/
    possessions.py   # team possessions from boxscore
    net_rating.py    # OffRtg/DefRtg/NetRtg per 100 poss + splits + shrinkage
    elo.py           # Elo engine, historic blend, current season update
  sim/
    model.py         # matchup features (A, B) + logistic P(HomeWin)
    engine.py        # Monte Carlo margin simulation
```
