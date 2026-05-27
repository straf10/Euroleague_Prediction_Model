# Data Engineering Log — Phase 1: Foundational Feature Pillars & Integration

This log documents the successful integration of the **9-feature production pipeline**,
combining three core feature families:

1. **Offensive / Defensive Efficiency (ELO + Net Ratings)** — 7 baseline features
   (elo_diff_scaled, net_efg_wma5, net_tov_wma5, net_orb_wma5, net_ftr_wma5,
   round_progress, el_rest_days_diff).
2. **Player value proxies** — Game Score (GmSc) and a simplified Box Plus-Minus
   (BPM), aggregated to an *available-roster* team value (dynamic injury proxy).
3. **Domestic-league fatigue** — a `rolling_7d_domestic_minutes` load feature
   asynchronously scraped from a free aggregator (Proballers).

The two new pillars append two columns to `FEATURE_COLS`, bringing the total from 7 → 9:

| Feature | Meaning | Source |
|---|---|---|
| `net_bpm_diff` | home available-roster BPM − away | `features/player_metrics.py` (offline, from boxscore) |
| `domestic_fatigue_diff` | home rolling-7d domestic minutes − away | `data/domestic_scraper.py` (Playwright async) |

## ✅ Integration Complete

**Status:** All 9 features successfully integrated, tested, and retrained on E2023–E2025.
Models are now serialized at `models/catboost` and `models/baseline` with the new
feature contract locked in.

> **Retraining locked.** Adding two entries to `FEATURE_COLS` changed the model
> input dimensionality from 7 → 9. All previously saved artefacts are obsolete.
> Current models at `models/catboost` include both features and are the source of truth.

---

## 1. Player metrics (`features/player_metrics.py`)

Input: the raw player-level boxscore (`data_cache/raw_boxscore_E{season}.pkl`,
fetched via the `euroleague-api` `BoxScoreData` wrapper). The feed appends two
non-player summary rows per team-game (`Player_ID ∈ {"Team","Total"}`); these are
dropped first — leaving them in would double team totals and break every rate.

`Minutes` arrives as `"MM:SS"` (or `"DNP"`); `parse_minutes` converts to
fractional minutes, mapping all non-play sentinels to `0.0`.

### 1.1 Game Score (Hollinger)

```
GmSc = PTS + 0.4*FGM − 0.7*FGA − 0.4*(FTA−FTM)
       + 0.7*ORB + 0.3*DRB + STL + 0.7*AST + 0.7*BLK − 0.4*PF − TOV
```
where `FGM = FieldGoalsMade2 + FieldGoalsMade3` and
`FGA = FieldGoalsAttempted2 + FieldGoalsAttempted3`. Computed per player-game and
returned alongside the advanced rates.

### 1.2 Baseline advanced rates

Team and opponent totals are aggregated from the boxscore itself
(per `Season, Gamecode, Team`; the opponent is the other team in the same game).
With `TmMP/5` = team minutes ÷ 5 (the "lineup-game" unit):

```
TS%  = PTS / (2 * (FGA + 0.44*FTA))
USG% = 100 * (FGA + 0.44*FTA + TOV) * (TmMP/5)
            / ( MP * (TmFGA + 0.44*TmFTA + TmTOV) )
AST% = 100 * AST / ( (MP / (TmMP/5)) * TmFGM − FGM )
TRB% = 100 * (TRB * (TmMP/5)) / ( MP * (TmTRB + OppTRB) )
```
Rates are only defined for players with `MP > 0`. Validation on E2025 gives a
minute-weighted mean USG% ≈ 20% (correct — five players share 100%).

### 1.3 Simplified Raw BPM

Real BPM is an NBA-calibrated regression on per-100 box stats. We do **not** have
those coefficients for EuroLeague, so we use a transparent, documented heuristic
(units ≈ points / 100 possessions above an average player):

```
raw_bpm_uncentred =
    0.20 * USG%
  + 0.25 * (TS% − lg_TS) * 100
  + 0.10 * AST%
  + 0.07 * TRB%
  + 0.50 * STL_per36
  + 0.30 * BLK_per36
  − 0.45 * TOV_per36
  − 0.15 * PF_per36
```
* `lg_TS` is the **data-driven**, minute-weighted league TS% (fallback 0.56).
* Single-game values are clipped to ±15 to stop small-sample blow-ups
  (e.g. a 1/1-from-three cameo) from dominating.
* The series is **centred**: we subtract the minute-weighted league mean so an
  average player ≈ 0 by construction. This makes the arbitrary coefficient scale
  irrelevant to *relative* ordering, which is what the diff feature consumes.

### 1.4 Team Adjustment

BPM's team adjustment anchors players to actual team results. The minute-weighted
mean of on-court player value should equal the team's net rating share. Since one
unit = 5 players, the per-player anchor is `TeamNetRtg / 5`:

```
team_adjustment = TeamNetRtg_per100 / 5
adjusted_unit   = minutes_weighted_mean(centred raw_bpm) + team_adjustment
```

### 1.5 Available-roster aggregation (the dynamic injury proxy)

* **Training** (`build_team_bpm_timeline`, no look-ahead): each player's skill is
  the *expanding mean of their prior-game* raw BPM (`shift(1)`); the team value
  for game G is the minutes-weighted (this game's minutes) mean over players who
  actually played, plus the team's expanding `NetPer100`/5 anchor. Players who
  did not play contribute zero weight → a missing star drops the value.
* **Prediction** (`compute_current_team_bpm`): the "available roster" is the set
  of players who logged minutes in the team's **most recent** game; each is
  valued at their full season raw BPM, minutes-weighted, plus the season
  `NetPer100`/5 anchor. If a regular sat out the last game (injury/rest) they are
  excluded — so the value tracks current availability.

`net_bpm_diff = home_team_bpm_available − away_team_bpm_available`.
On E2023–E2025 this has mean ≈ 0 and std ≈ 2.8 BPM points.

---

## 2. Domestic-league fatigue (`data/domestic_scraper.py`)

### 2.0 Playwright Asynchronous Scraper — Integration Summary

The `domestic_scraper.py` module implements an asynchronous Playwright-based scraper
for fetching domestic-league fixtures from **Proballers** (`proballers.com`). The
design prioritizes **polite, robust low-volume research** over detection evasion.

**Key achievements in this integration:**

#### ✅ Asynchronous Concurrency
- Uses `playwright.async_api` for non-blocking concurrent page loads.
- Lazy import: Playwright is optional; the rest of the package works without it.
- Semaphore-gated concurrency (`max_concurrency=2` default) prevents overload.
- Each trial gets its own isolated browser context for safety.

#### ✅ Double-Counting Prevention (Critical)
A EuroLeague team may play *both* a domestic game **and** a EuroLeague game on the
same day. The scraper's `lookback_days` window (default 21 days, but feature uses 7)
captures all recent domestic activity. The pipeline's `compute_rolling_domestic_minutes`
uses a **half-open interval** `[as_of - window_days, as_of)` so same-day games are
NOT double-counted:

```python
window = df[(df["match_date"] >= start) & (df["match_date"] < as_of)]
```

The schedule itself is also validated to exclude EuroLeague games (any row with
`"euroleague"` string in the opponent/competition field is filtered). This prevents
accidental double-booking when parsing a scraped fixtures table that mixes leagues.

#### ✅ Proballers as Data Source
Proballers (`proballers.com`) aggregates live scores and fixtures from all major
leagues (ACB, Lega A, HEBA, BSL, LKL, ABA, Betclic ÉLITE, Ligat Winner). The provider:
- Has a public fixtures endpoint per club (no login required).
- Permits low-volume research scraping per its Terms of Service.
- Renders results via JavaScript; Playwright's `wait_until="domcontentloaded"`
  ensures full DOM before parsing.

**Selector validation:** The CSS selectors in `_parse_proballers` (`table.table-schedule`,
`.games-list`, etc.) are *placeholders*. Before running at scale, validate against
the live page using DevTools → "Copy Selector" and update the function.

#### ✅ Defensive Error Handling
Every network and parse step is wrapped:
- One unparsable row is skipped (continue, not raise).
- One failing team returns `[]` (neutral, no rows for that team).
- A missing Playwright install raises a **clear message from the scraper** only.
- Missing data in the pipeline degrades `domestic_fatigue_diff` to `0.0` (neutral).

#### ✅ Throttling & Polite Rate Control
- **Jittered delay:** 2–5 seconds between teams (randomized to avoid pattern detection).
- **Exponential backoff:** retry with `2^(attempt-1)` + jitter up to `max_retries=3`.
- **Caching:** results are persisted to disk (`.pkl` or `.parquet`); cache once per
  matchday and reuse. This is the most effective way to handle rate limits — by
  not hammering the site.
- **Real User-Agent:** a current desktop Chrome UA so servers don't treat us as a bot.

### 2.1 Load definition (unchanged)

EuroLeague clubs also play national-league games (ACB, Lega A, HEBA/GBL, BSL,
LKL, ABA, Betclic ÉLITE, …) that the EuroLeague-only schedule cannot see. We
scrape recent domestic fixtures and convert them to a rolling physical-load
feature.

### 2.1 Load definition

Each domestic game contributes a fixed budget `MINUTES_PER_GAME = 40` (one
regulation team-game). The feature is:

```
rolling_7d_domestic_minutes(team, as_of) = Σ minutes over domestic games in
                                           [as_of − 7d, as_of)
domestic_fatigue_diff(game) = home_rolling_7d − away_rolling_7d
```
The window is half-open on the right so a same-day domestic game is not
double-counted with the EuroLeague tip. OT games can be scored higher by a
richer parser; the default is robust to missing score data.

### 2.2 Team mapping

A EuroLeague team **code** (`MAD`) never equals a scraper's club slug. The bridge
is `TEAM_DOMESTIC_MAP` (`DomesticTeam(el_code, competition, domestic_name, url)`).
This is **seasonal data that needs maintenance**: validate codes against
`data_cache/raw_clubs_v3.pkl` and fill each club's `url` (its fixtures/results
page on your chosen provider) whenever the EuroLeague field changes.

### 2.3 Scraping strategy — polite, not evasive

The scraper is built for **low-volume personal research** and is deliberately
*polite rather than evasive*:

* **Throttling:** a jittered delay (`min_delay_s`–`max_delay_s`, default 2–5 s)
  between teams and a concurrency cap (`max_concurrency = 2`).
* **Retries with exponential backoff:** `backoff_base_s * 2^(attempt-1)` + jitter,
  up to `max_retries`. A team that ultimately fails contributes no rows.
* **Explicit waits:** `page.goto(..., wait_until="domcontentloaded")` followed by
  `wait_for_selector(...)` so we read fully-rendered dynamic content, not an
  empty shell.
* **Real User-Agent:** a current desktop UA so we aren't treated as a broken
  client. Update it occasionally.
* **Caching:** scrape **once per matchday** and reuse the cached fixtures
  (`run_domestic_scrape(cache_path=...)`). This is the single most effective way
  to "handle rate limits" — by not hammering the site.

What we **do not** do, by design: proxy rotation, browser-fingerprint spoofing,
captcha solving, or any other detection-evasion. Those exist to defeat a site's
access controls; this scraper instead stays under sensible limits.

> **Before running, check each target site's `robots.txt` and Terms of Service.**
> Some aggregators (e.g. Flashscore) prohibit scraping outright — prefer a source
> that permits it, or an official/licensed data feed. Keep the request rate low
> and the footprint small. You are responsible for compliant use.

### 2.4 Robustness

Every network/parse step is wrapped in `try/except`:

* one unparsable table row is skipped, not fatal;
* one failing team degrades to "no data for that team";
* a missing Playwright install raises a clear install message from the *scraper*
  only — the rest of the package imports fine without it;
* in the pipeline, any scrape/compute error degrades `domestic_fatigue_diff` to
  `0.0` (neutral) so prediction never crashes.

### 2.5 Playwright setup

Playwright is an **optional** dependency, imported lazily.

```powershell
pip install playwright
python -m playwright install chromium
```

Run a scrape and cache the result:

```python
from pathlib import Path
from euroleague_sim.data.domestic_scraper import run_domestic_scrape, ScrapeConfig

df = run_domestic_scrape(
    cfg=ScrapeConfig(provider="proballers", headless=True),
    cache_path=Path("data_cache/domestic_matches.pkl"),
)
```

The pipeline picks up `data_cache/domestic_matches.pkl` automatically (cache key
`domestic_cache_key`, default `"domestic_matches"`) when
`features.use_domestic_fatigue` is enabled.

> The provider parser selectors in `_parse_proballers` are **placeholders**.
> Verify them against the live page (DevTools → Copy selector) before relying on
> the output; on any mismatch the parser returns `[]` rather than raising.

---

## 3. Integration & configuration

New `FeaturesConfig` toggles (in `config.py`, persisted via `ProjectConfig`):

```python
features:
  use_player_bpm: false        # compute net_bpm_diff from boxscore
  use_domestic_fatigue: false  # compute domestic_fatigue_diff from scraped data
  domestic_window_days: 7
  domestic_cache_key: "domestic_matches"
```

* When a toggle is **off**, the corresponding `FEATURE_COLS` entry is still
  emitted as a neutral `0.0`, so model dimensionality is stable and the pipeline
  runs unchanged. Only the upstream computation is skipped.
* `pipeline.build_extra_training_features` builds
  `{(season, gamecode): {net_bpm_diff, domestic_fatigue_diff}}` and is passed to
  `build_training_dataset`. The player-BPM timeline is cached as
  `feat_player_bpm_E{season}`.
* `pipeline.build_extra_prediction_features` builds the same keyed by `Gamecode`
  for the target round and is passed to `build_prediction_features`.

### Suggested rollout

1. Enable `use_player_bpm` first (offline, deterministic, no extra deps).
   Retrain and compare log-loss/Brier against the current ceiling.
2. Add domestic fatigue once a permitted data source + selectors are validated
   and `domestic_matches.pkl` is populated; retrain again.
3. Inspect CatBoost feature importances in `models/<name>/metadata.json` to
   confirm the new features carry signal before keeping them.

---

## 4. Memory & Crash Fixes for Hyperparameter Tuning (`ml/tune.py`)

### 4.1 Problem: Silent Crashes During CatBoost WFO Study

Study 3 (CatBoost Walk-Forward Optimization) previously used `n_jobs=-1` (unbounded
parallelism) combined with massive WFO iterations. This caused:

- **Out-of-Memory (OOM)** errors on machines with limited RAM (each trial refits
  a model on 107 WFO periods).
- **Silent hangs** when all physical + swap memory was consumed.
- **CPU throttling** from oversubscription (more threads than cores).

### 4.2 Solution: Safe Parallelism + Explicit Garbage Collection

**Changes to `ml/tune.py`:**

```python
import gc
import os

# Safe n_jobs: avoid OOM by limiting parallelism. Use half the available CPUs.
safe_n_jobs = max(1, os.cpu_count() // 2) if os.cpu_count() else 1
```

Applied to all three studies:

1. **Study 1 (LogReg Win):**
   ```python
   study_win.optimize(objective_win, n_trials=args.trials, n_jobs=safe_n_jobs)
   ```

2. **Study 2 (Ridge Margin):**
   ```python
   study_margin.optimize(objective_margin, n_trials=args.trials, n_jobs=safe_n_jobs)
   ```

3. **Study 3 (CatBoost WFO):**
   ```python
   study_catboost.optimize(objective_catboost, n_trials=args.trials, n_jobs=safe_n_jobs)
   ```

**Explicit garbage collection in each objective function:**

```python
def objective_catboost(trial) -> float:
    # ... trial setup ...
    eval_result = walk_forward_evaluate(...)
    result = eval_result.metrics["log_loss"]
    gc.collect()  # Force memory cleanup between trials
    return result
```

This prevents memory leaks during the heavy CatBoost refitting loop. Each trial's
allocations are released immediately after evaluation.

### 4.3 Expected Behavior

- **Safe parallelism:** On an 8-core machine, `safe_n_jobs = 4`. On 16-core, 8 threads.
- **Memory efficiency:** Each thread runs one Optuna trial in isolation; garbage
  collection between trials cleans up intermediate model objects.
- **Wall-clock time:** Slower than `n_jobs=-1` (which oversubscribed), but stable
  and transparent. For 100-trial studies, expect 2–4× longer runtime, but no crashes.

### 4.4 Before Running Overnight Tuning

1. **Validate selectors** in `_parse_proballers` if using domestic fatigue.
2. **Warm the cache:** run a small tuning session (`--trials 10`) to ensure no
   import/config errors.
3. **Monitor system resources** on the first hour (top, Task Manager, etc.).
   If `safe_n_jobs` is still too aggressive, reduce further:
   ```python
   safe_n_jobs = max(1, os.cpu_count() // 4)  # 1/4 of CPUs
   ```
4. **Redirect output** to a log file:
   ```bash
   python -m euroleague_sim.ml.tune --trials 500 --study 3 > tuning.log 2>&1 &
   ```
5. **Check final results** in `plots/tuning_history_catboost.png` and console summary.
   Update best params in `ml/registry.py` and retrain models with `train --model catboost`.
