# Technical Design — Euroleague Prediction Model

Step-by-step technical specification for the prediction + simulation system
targeting the **current EuroLeague season (2025-26)**.

Data sourced via the [euroleague-api](https://github.com/giasemidis/euroleague-api)
wrapper and the EuroLeague API v3 (`https://api-live.euroleague.net/v3`).

---

## 0. Identifiers

| Key | Value |
|---|---|
| `competitionCode` | `E` (EuroLeague) |
| `seasonCode` | `E2025` (start year 2025) |
| `season` (int, for wrapper) | `2025` |

---

## 1. Project Layout

```
euroleague-sim/
  pyproject.toml              # build config + dependencies
  requirements.txt
  README.md
  TECHNICAL_DESIGN.md         # this file
  data_cache/                 # pickle/JSON filesystem cache
  models/                     # persisted ML artefacts
  outputs/                    # prediction CSVs
  src/euroleague_sim/
    __init__.py
    config.py                 # ProjectConfig dataclass (Elo, MC, shrinkage, ML)
    pipeline.py               # orchestration: fetch -> features -> Elo -> train -> predict
    cli.py                    # CLI entry point (update-data, train, predict)
    data/
      fetch.py                # EuroleagueFetcher: API wrapper + v3 HTTP fallback
      cache.py                # Cache: pickle DataFrames + JSON dicts
    features/
      possessions.py          # team possessions from player boxscore
      net_rating.py           # OffRtg/DefRtg/NetRtg per 100 poss + splits + shrinkage
      elo.py                  # Elo engine: historic blend + current season update
    ml/
      features.py             # 20-feature engineering (training + prediction modes)
      train.py                # train RF + XGBoost + NN (classifier + regressor)
      predict.py              # EnsemblePredictor: weighted 3-model inference
    sim/
      model.py                # matchup features A, B
      engine.py               # Monte Carlo margin simulation
```

### Dependencies

```
euroleague-api>=0.0.21    # EuroLeague data wrapper
pandas>=2.0               # data manipulation
numpy>=1.24               # numerical operations
requests>=2.31            # HTTP requests
scipy>=1.11               # statistical functions
scikit-learn>=1.3         # Random Forest + Neural Network
xgboost>=2.0              # gradient boosting
joblib>=1.3               # model serialisation
tqdm>=4.66                # progress bars
python-dateutil>=2.9      # date parsing
```

---

## 2. Data Model

### Cached DataFrames (pickle, in `data_cache/`)

**raw_gamecodes_{season}** — one row per game
- `Gamecode` (int), `Round` (int), `home_team`, `away_team`, `played` (bool)
- Source: `EuroleagueFetcher.gamecodes_season()`

**raw_boxscore_{season}** — one row per player-game
- All player box score stats (PTS, FGA, FGM2, FGM3, FTA, OREB, TOV, etc.)
- Source: `EuroleagueFetcher.player_boxscore_stats_season()`

**feat_possessions_{season}** — one row per team-game
- `Team`, `Gamecode`, `Possessions` (estimated)

**feat_games_{season}** — one row per game
- `Gamecode`, `Round`, `home_team`, `away_team`, `home_points`, `away_points`
- `possessions_game`, `margin_home`

**feat_team_game_{season}** — one row per team-game
- `Team`, `Gamecode`, `Round`, `IsHome`, `PointsFor`, `PointsAgainst`
- `Possessions`, `OffPer100`, `DefPer100`, `NetPer100`
- Four Factors columns: `FGA`, `FGM2`, `FGM3`, `FTA`, `ORB`, `TOV`, `opp_DRB`

**feat_team_ratings_{season}** — one row per team (season aggregate)
- `Team`, `Overall_OffPer100`, `Overall_DefPer100`, `Overall_NetPer100`
- `Home_OffPer100`, `Home_DefPer100`, `Home_NetPer100`
- `Away_OffPer100`, `Away_DefPer100`, `Away_NetPer100`

### Cached JSON

**season_summary_{season}** — league-wide aggregates
- `possessions_per_game`, `margin_sigma_points`, `league_home_adv_points`

**feat_elo_current_{season}** — `{teamCode: eloRating}` dict

### Model Artefacts (in `models/`)

- `scaler.joblib` — StandardScaler fitted on training data
- `rf_classifier.joblib`, `rf_regressor.joblib` — Random Forest
- `xgb_classifier.joblib`, `xgb_regressor.joblib` — XGBoost
- `nn_classifier.joblib`, `nn_regressor.joblib` — Neural Network (MLP)
- `metadata.json` — feature list, metrics, feature importances

---

## 3. Data Ingestion (Steps 1-2 in pipeline)

### 3.1 Clubs/Teams

`EuroleagueFetcher.clubs_v3()` — cached as `raw_clubs_v3`.
All internal joins use `teamCode` (3-letter code, e.g. `PAO`, `OLY`).

### 3.2 Gamecodes (schedule + results)

`EuroleagueFetcher.gamecodes_season(season)` — returns all games for the season
with columns: `Gamecode`, `Round`, home/away teams, `played` flag.

For round-specific schedule: `EuroleagueFetcher.gamecodes_round(season, round)`.

### 3.3 Player Boxscore

`EuroleagueFetcher.player_boxscore_stats_season(season)` — all player stats
for all played games. Used to compute team possessions and Four Factors.

---

## 4. Feature Engineering (Steps 3-6 in pipeline)

### 4.1 Possessions (per team-game)

```
Possessions = FGA + 0.44 * FTA - OREB + TOV
```

Computed from aggregated player boxscore data per team-game.

### 4.2 Net Ratings (per 100 possessions)

```
OffRtg = 100 * PointsFor / Possessions
DefRtg = 100 * PointsAgainst / Possessions
NetRtg = OffRtg - DefRtg
```

Aggregated into overall, home-split, and away-split ratings.

### 4.3 Early-Season Shrinkage

```
NetRtg_adj = (n / (n + k)) * NetRtg + (k / (n + k)) * 0
```

Where `k = 6` games (configurable). Pulls early-season extreme ratings towards zero.

### 4.4 Four Factors (per team, cumulative)

| Factor | Formula |
|---|---|
| eFG% | `(FGM2 + 1.5 * FGM3) / FGA` |
| TOV% | `TOV / Possessions` |
| ORB% | `ORB / (ORB + opp_DRB)` |
| FT Rate | `FTA / FGA` |

---

## 5. Elo Ratings (Step 7 in pipeline)

### 5.1 Elo Update Rule

```
Expected = 1 / (1 + 10^(-((Elo_H + HCA) - Elo_A) / 400))
Elo_H += K * (S - Expected)
Elo_A -= K * (S - Expected)
```

Where `S = 1` if home win, `0` otherwise.

| Parameter | Default |
|---|---|
| Base rating | 1500 |
| K-factor | 20 |
| Home advantage (HCA) | 65 Elo points |

### 5.2 Historic Blend (EloHist)

From the 2 preceding seasons:
```
EloHist = 0.65 * Elo_season(N-1) + 0.35 * Elo_season(N-2)
```

### 5.3 Current Season Elo (EloCurrent)

Initialised from EloHist, then updated game-by-game through all played games
in the current season.

---

## 6. ML Feature Matrix (20 features)

### Training Mode (point-in-time, no lookahead)

For each historical game, features reflect only data available *before* that game
(expanding-window cumulative stats with shift-by-one).

### Prediction Mode

Uses full-season cumulative stats and current Elo values.

### Feature List

| # | Feature | Description |
|---|---|---|
| 1 | `net_rtg_diff` | Home net rating (split) minus away net rating (split) |
| 2 | `elo_diff_scaled` | (Elo_home - Elo_away) / 25 |
| 3 | `off_matchup` | Home offensive rating minus away defensive rating |
| 4 | `def_matchup` | Away offensive rating minus home defensive rating |
| 5 | `win_pct_diff` | Home win% minus away win% |
| 6 | `form_diff` | Home last-5-game NetPer100 minus away last-5-game NetPer100 |
| 7 | `pace_diff` | Home possessions/game minus away possessions/game |
| 8 | `efg_diff` | Effective FG% differential |
| 9 | `tov_pct_diff` | Turnover rate differential |
| 10 | `orb_pct_diff` | Offensive rebound rate differential |
| 11 | `ft_rate_diff` | Free-throw rate differential |
| 12 | `home_off_rtg` | Home team offensive rating |
| 13 | `home_def_rtg` | Home team defensive rating |
| 14 | `home_win_pct` | Home team win percentage |
| 15 | `elo_home` | Home team Elo rating |
| 16 | `away_off_rtg` | Away team offensive rating |
| 17 | `away_def_rtg` | Away team defensive rating |
| 18 | `away_win_pct` | Away team win percentage |
| 19 | `elo_away` | Away team Elo rating |
| 20 | `round_progress` | Current round / max rounds |

---

## 7. ML Models

### 7.1 Random Forest (RF)

| Parameter | Default |
|---|---|
| n_estimators | 300 |
| max_depth | 6 |
| min_samples_leaf | 5 |

Trained on scaled features (StandardScaler). Provides both classification
(win/loss) and regression (margin).

### 7.2 XGBoost (XGB)

| Parameter | Default |
|---|---|
| n_estimators | 300 |
| max_depth | 4 |
| learning_rate | 0.05 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| reg_alpha (L1) | 0.1 |
| reg_lambda (L2) | 1.0 |

Classifier uses `binary:logistic`, regressor uses `reg:squarederror`.

### 7.3 Neural Network (NN / MLP)

| Parameter | Default |
|---|---|
| hidden_layer_sizes | (64, 32) |
| activation | relu |
| solver | adam |
| alpha (L2 penalty) | 0.001 |
| max_iter | 1000 |
| early_stopping | True (15% validation) |

### 7.4 Ensemble

```
P(HomeWin) = (w_rf * P_rf + w_xgb * P_xgb + w_nn * P_nn) / (w_rf + w_xgb + w_nn)
Margin     = (w_rf * M_rf + w_xgb * M_xgb + w_nn * M_nn) / (w_rf + w_xgb + w_nn)
```

Default weights: RF = 0.35, XGBoost = 0.35, NN = 0.30.

### 7.5 Evaluation Metrics

- **Cross-validated accuracy** (5-fold)
- **Brier score** (calibration quality, lower = better)
- **Log-loss** (probability quality, lower = better)
- **Margin MAE** (mean absolute error in points)
- **Margin RMSE** (root mean squared error in points)
- **Feature importances** (averaged from RF + XGBoost)

---

## 8. Monte Carlo Simulation

### 8.1 Margin Distribution

```
mu = ML_predicted_margin   (or alpha1*A + alpha2*B + alpha3 if no ML models)
margin ~ Normal(mu, sigma)
home_win if margin > 0
```

| Parameter | Default |
|---|---|
| alpha1 | 0.9 |
| alpha2 | 1.0 |
| alpha3 | 1.5 (home-court edge in points) |
| sigma | 11.5 (or computed from historical data) |
| N simulations | 20,000 |

### 8.2 Outputs

- `pHomeWin` — MC win probability
- `meanMargin` — average simulated margin
- `q10`, `q50`, `q90` — margin quantiles

---

## 9. Execution Pipeline

Each prediction run follows this sequence:

```
Step  1   fetch_clubs()                   cached, refreshed weekly
Step  2   gamecodes_season()              schedule + results for all seasons
Step  3   player_boxscore_stats()         raw boxscore for all played games
Step  4   compute_team_possessions()      possessions per team-game
Step  5   build_games_with_possessions()  one-row-per-game with margin
Step  6   build_team_game_net_ratings()   OffRtg/DefRtg/NetRtg per team-game
Step  7   aggregate_team_ratings()        season aggregate with splits + shrinkage
Step  8   build_elo_hist()                blend past 2 seasons
Step  9   build_current_season_elo()      update game-by-game
Step 10   train_models()                  RF + XGBoost + NN on 3 seasons (optional)
Step 11   build_prediction_features()     20-feature matrix for upcoming games
Step 12   ensemble.predict()              3-model weighted ensemble
Step 13   simulate_next_round()           Monte Carlo for margin distribution
Step 14   save_predictions()              CSV to outputs/
```

---

## 10. Practical Notes

- **Game codes**: The wrapper uses `game_code` as an integer (e.g. `47`), derived
  from the full code `E2025_47`.
- **Round detection**: next round = smallest round with at least one unplayed game,
  or `max_played_round + 1` if the API returns only played games.
- **Caching**: all raw and processed data cached in `data_cache/`. Use `--force`
  flag to re-download and rebuild.
- **Team codes**: always join on `teamCode` (3-letter code), never on display names.
- **Backward compatibility**: if XGBoost models are missing (trained before v0.3),
  the predictor falls back to RF + NN ensemble automatically.
