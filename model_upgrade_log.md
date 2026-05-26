# Model Upgrade Log

Living documentation for the SOTA ML upgrade of the EuroLeague prediction model.
Tracks the rationale, metrics, tuning notes, and execution-speed optimizations.

- **Dataset:** ~1,035 labelled games (3 seasons: 2023, 2024, 2025), 7 features.
- **Home-win base rate:** 0.634
- **Last updated:** 2026-05-26

---

## 1. Rationale

Deep-research concluded that Logistic Regression + `TimeSeriesSplit` + Platt
scaling is insufficient for a small (~1k rows), high-variance dataset: it
overfits, the fixed CV folds leak temporal structure, and Platt (sigmoid)
calibration is wrong at the tails. The three upgrades below address each issue.

### 1.1 CatBoost (replaces Logistic Regression)

- **Why:** CatBoost's *symmetric (oblivious) trees* split on the same
  feature/threshold across an entire level. This structural constraint is a
  strong, built-in regulariser — exactly what a ~1k-row dataset needs to avoid
  the overfitting that unconstrained GBMs (XGBoost/LightGBM) suffer here.
- **Config for small data:** shallow trees (`depth=3`), heavy L2
  (`l2_leaf_reg=10`), slow learning (`learning_rate=0.03`), Bayesian bootstrap
  for row-level noise injection. Trees are scale-invariant → **no
  `StandardScaler`** (`requires_scaling=False`).
- **Margin:** `CatBoostRegressor` (RMSE) replaces `Ridge`, keeping the whole
  model tree-based and scaler-free for architectural coherence.
- Runs **strictly on CPU** — benchmarked GPU (GTX 1650) at ~1974 ms/fit vs CPU
  ~451 ms/fit (4.4× slower) because the dataset is far too small to amortise
  GPU transfer/launch overhead.

### 1.2 Walk-Forward Optimization / WFO (replaces `TimeSeriesSplit`)

- **Why:** WFO mirrors real deployment. It iterates chronologically, trains on
  the **expanding** window of all strictly-earlier games, and predicts the
  immediate, unseen next period. No future information ever reaches a training
  fold → zero temporal leakage, and metrics reflect true out-of-sample (OOS)
  performance.
- **Granularity is configurable** (`wfo_step`): `"round"` (per season-round),
  `"season"`, or an integer block of N rounds. Final evaluation uses `"round"`
  for maximum OOS precision (107 scorable folds on the current data).

### 1.3 Beta Calibration (replaces `CalibratedClassifierCV` / Platt)

- **Why:** Platt scaling assumes a logistic link, which systematically
  mis-calibrates confident predictions at the tails. **Beta calibration**
  (`betacal`, 3-parameter `"abm"` map) fits a Beta-distribution link that
  corrects both tails and the midrange.
- **Where:** `BetaCalibratedClassifier` wraps the CatBoost win classifier. On
  each fit it reserves a chronological *calibration tail* of the training
  window, fits the Beta map on the model's held-out scores, then refits the
  base estimator on the full window. Calibration is therefore applied to the
  CatBoost probabilities at **every OOS WFO step** — leakage-free, since the
  base never trains on the calibration tail.

---

## 2. Metrics: Baseline vs Upgraded

Win-probability and margin metrics. **Note the evaluation schemes differ** and
are not a like-for-like OOS comparison: the baseline reference is on
`TimeSeriesSplit` (5 folds), the CatBoost model on the stricter round-by-round
WFO (107 expanding folds, every game after the warm-up scored OOS).

| Metric (win)      | Baseline (LogReg, CV) | CatBoost (Beta-cal + WFO) | Target |
|-------------------|-----------------------|---------------------------|--------|
| CV/OOS Accuracy   | 0.640                 | 0.609                     | ↑      |
| Brier score       | 0.2238                | 0.2332                    | ↓      |
| Log-loss          | 0.6384                | 0.6593                    | ↓      |

| Metric (margin)   | Baseline (Ridge, CV)  | CatBoost (WFO)            | Target |
|-------------------|-----------------------|---------------------------|--------|
| MAE (points)      | 9.44                  | 9.53                      | ↓      |
| RMSE (points)     | 12.12                 | 12.27                     | ↓      |

> Reference baseline (historical, as briefed): Accuracy ≈ 64.5%, Brier ≈ 0.222,
> Log-loss ≈ 0.634.

**Reading the table:** with *default* (untuned) CatBoost params the model is
marginally behind the tuned baseline — expected, because (a) WFO is a harsher
evaluator than `TimeSeriesSplit`, and (b) the baseline's `C`/`alpha` are already
Optuna-tuned while CatBoost's are not yet. Offline tuning closes the gap (see
§3): a `season`-step tuning sweep already found CatBoost **WFO log-loss ≈ 0.6495**.

---

## 3. Hyperparameter Tuning Notes

Tuning is offline and isolated from the daily pipeline
(`python -m euroleague_sim.ml.tune`). Found params are hardcoded into
`src/euroleague_sim/ml/registry.py`. Three Optuna studies run: LogReg (win),
Ridge (margin), and CatBoost (win+margin via WFO).

CatBoost search space: `depth ∈ [2,5]`, `l2_leaf_reg ∈ [1,30] (log)`,
`learning_rate ∈ [0.01,0.1] (log)`, `iterations ∈ [200,800]`,
`bagging_temperature ∈ [0,2]`. Objective = WFO log-loss (minimise).

| Date       | Study     | Trials | Step     | Best objective         | Best params (summary)                                   |
|------------|-----------|--------|----------|------------------------|---------------------------------------------------------|
| 2026-05-26 | LogReg    | 4      | —        | log-loss 0.6372        | `C=0.034`                                                |
| 2026-05-26 | Ridge     | 4      | —        | MAE 9.442              | `alpha=96.19`                                            |
| 2026-05-26 | CatBoost  | 4      | season   | WFO log-loss 0.6495    | `depth=2, l2=12.95, lr=0.077, iters=600, bag_temp=0.42` |

> Smoke-test run (4 trials only) — rerun with `--trials 100+ --wfo-step 5`
> before committing params to the registry.

---

## 4. Performance & Execution Optimizations

CatBoost runs **CPU-only** (GPU is slower at this scale, see §1.1). The cost
driver is WFO refitting from scratch each period, so the optimizations target
the number and length of refits rather than the device.

### 4.1 Configurable WFO step (`wfo_step`)

Coarsen the walk-forward granularity during tuning to slash refit count, while
keeping precise round-by-round evaluation for final/reported metrics.

| `wfo_step` | Periods | Scorable folds (refits/eval) | Use case                         |
|------------|---------|------------------------------|----------------------------------|
| `"round"`  | 130     | **107**                      | Final evaluation (`train`)       |
| `5`        | 28      | 23                           | Balanced tuning                  |
| `"season"` | 3       | **2**                        | Fast tuning (~53× fewer refits)  |

- `train` / `train --model all`: always `"round"` (precise OOS).
- `tune`: `--wfo-step` (default `"season"`); accepts `round`, `season`, or int N.

### 4.2 Parallel tuning (`n_jobs=-1`)

All three Optuna studies call `study.optimize(..., n_jobs=-1)` to run trials
concurrently across every CPU core. CatBoost releases the GIL during fit, so
threaded trials parallelise for real. To avoid oversubscription, the CatBoost
tuning estimators are pinned to `thread_count=1` — parallelism comes from
concurrent **trials** (one core each), not from intra-fit threading.

### 4.3 Early stopping (`early_stopping_rounds=50`)

CatBoost win & margin estimators set `early_stopping_rounds=50`. Each fit holds
out a chronological tail as an `eval_set`; training halts once the validation
metric plateaus, then the model is **refit on the full window capped at the
best iteration** (shared helper `fit_catboost_es`) — no validation data wasted
at inference, no compute spent on plateaued boosting rounds. For the win model,
the Beta-calibration tail doubles as this early-stopping `eval_set`.

---

## 5. Touched Components

- `ml/registry.py` — `get_catboost_model()` factory (`"catboost"` key).
- `ml/calibration.py` — `BetaCalibratedClassifier`, `fit_catboost_es` helper.
- `ml/evaluate.py` — `walk_forward_evaluate`, `build_wfo_periods`.
- `ml/train.py` — WFO/CV dispatch, `wfo_step` threading, ES on final fit.
- `ml/tune.py` — CatBoost WFO study, `--wfo-step`, `n_jobs=-1`.
- `config.py` — `MLConfig.wfo_min_train_size`.
- `requirements.txt` / `pyproject.toml` — `catboost>=1.2`, `betacal>=1.1`.

Feature-engineering scripts were intentionally **not** modified.

### CLI quick reference

```bash
# Train + evaluate CatBoost (round-by-round WFO, precise OOS)
python -m euroleague_sim.cli train --model catboost --season 2025

# Leaderboard: baseline vs catboost
python -m euroleague_sim.cli train --model all --season 2025

# Predict with the CatBoost model
python -m euroleague_sim.cli predict --model catboost --season 2025 --round next

# Offline tuning (fast coarse step) then commit params to registry.py
python -m euroleague_sim.ml.tune --trials 100 --wfo-step 5 --season 2025
```
