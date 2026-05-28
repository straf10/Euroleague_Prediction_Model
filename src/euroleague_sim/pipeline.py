from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from .config import ProjectConfig
from .data.cache import Cache
from .data.fetch import EuroleagueFetcher, FetchParams, _find_schedule_columns
from .features.possessions import compute_team_possessions_from_boxscore
from .features.net_rating import (
    build_games_with_possessions,
    build_team_game_net_ratings,
    aggregate_team_ratings,
    season_summary,
)
from .features.elo import (
    build_elo_hist,
    build_current_season_elo,
    run_elo,
)
from .sim.model import compute_matchup_features
from .sim.engine import simulate_next_round
from .ml.features import FEATURE_COLS, build_training_dataset, build_prediction_features
from .ml.train import train_models
from .ml.predict import load_predictor
from .features.player_metrics import (
    build_team_bpm_timeline,
    team_bpm_game_diff,
    compute_current_team_bpm,
    current_team_bpm_to_game_diff,
)
from .data.domestic_scraper import domestic_fatigue_diff_for_schedule
from .data.team_registry import load_season_meta, max_rounds_from_meta


def _key(prefix: str, season: int) -> str:
    return f"{prefix}_E{season}"


# ---------------------------------------------------------------------------
# Step 1-4: Fetch + cache raw data
# ---------------------------------------------------------------------------

def update_season_cache(
    cache: Cache,
    fetcher: EuroleagueFetcher,
    season: int,
    force: bool = False,
) -> None:
    """Fetch raw data for a season and store in cache."""
    k_gc = _key("raw_gamecodes", season)
    k_bx = _key("raw_boxscore", season)
    k_cl = "raw_clubs_v3"

    if force or not cache.has_df(k_gc):
        df = fetcher.gamecodes_season(season)
        cache.save_df(k_gc, df)

    if force or not cache.has_df(k_bx):
        df = fetcher.player_boxscore_stats_season(season)
        cache.save_df(k_bx, df)

    if force or not cache.has_df(k_cl):
        df = fetcher.clubs_v3()
        cache.save_df(k_cl, df)

    # Phase-1: regenerate the season meta + team registry every refresh so the
    # UI and the feature pipeline (round_progress normaliser) stay in sync
    # with whatever the API currently reports.
    try:
        from .data.team_registry import write_registry_and_meta
        gamecodes = cache.load_df(k_gc)
        clubs = cache.load_df(k_cl) if cache.has_df(k_cl) else None
        write_registry_and_meta(cache.root, season, gamecodes, clubs)
    except Exception as exc:  # noqa: BLE001 - never break the fetch step
        print(f"  [registry] skipped: {exc}")


# ---------------------------------------------------------------------------
# Step 5-6: Build possessions -> net ratings -> splits
# ---------------------------------------------------------------------------

def build_features_for_season(
    cache: Cache,
    season: int,
    cfg: ProjectConfig | None = None,
    force: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Build possessions -> games -> team ratings for a season."""
    k_gc = _key("raw_gamecodes", season)
    k_bx = _key("raw_boxscore", season)
    gamecodes = cache.load_df(k_gc)
    boxscore = cache.load_df(k_bx)

    shrink_games = (cfg.shrinkage.k_games if cfg else 6)

    k_poss = _key("feat_possessions", season)
    if force or not cache.has_df(k_poss):
        poss = compute_team_possessions_from_boxscore(boxscore)
        cache.save_df(k_poss, poss)
    else:
        poss = cache.load_df(k_poss)

    k_games = _key("feat_games", season)
    if force or not cache.has_df(k_games):
        games = build_games_with_possessions(gamecodes, poss)
        cache.save_df(k_games, games)
    else:
        games = cache.load_df(k_games)

    k_teamgame = _key("feat_team_game", season)
    if force or not cache.has_df(k_teamgame):
        team_game = build_team_game_net_ratings(games)
        cache.save_df(k_teamgame, team_game)
    else:
        team_game = cache.load_df(k_teamgame)

    # Shrinkage: k_games * avg_poss ≈ possessions to shrink towards zero net
    avg_poss = games["possessions_game"].mean() if not games.empty else 72.0
    shrink_possessions = shrink_games * avg_poss

    k_teamratings = _key("feat_team_ratings", season)
    if force or not cache.has_df(k_teamratings):
        team_ratings = aggregate_team_ratings(
            team_game,
            shrink_possessions=shrink_possessions,
        )
        cache.save_df(k_teamratings, team_ratings)
    else:
        team_ratings = cache.load_df(k_teamratings)

    summ = season_summary(games)
    summ_dict = {
        "possessions_per_game": summ.possessions_per_game,
        "margin_sigma_points": summ.margin_sigma_points,
        "league_home_adv_points": summ.league_home_adv_points,
        "league_home_adv_net100": summ.league_home_adv_net100,
    }
    cache.save_json(_key("season_summary", season), summ_dict)

    return games, team_game, team_ratings, summ_dict


# ---------------------------------------------------------------------------
# Step 7: Elo (historic + current season)
# ---------------------------------------------------------------------------

def build_current_elo(
    cache: Cache,
    cfg: ProjectConfig,
    current_season: int,
    force: bool = False,
) -> Dict[str, float]:
    """Compute EloCurrent for the current season.

    1) Build EloHist from past seasons.
    2) Initialize current season from EloHist.
    3) Update game-by-game through all played current-season games.
    """
    k_elo = _key("feat_elo_current", current_season)
    if not force and cache.has_json(k_elo):
        return cache.load_json(k_elo)

    # EloHist
    past_games: Dict[int, pd.DataFrame] = {}
    for s in range(current_season - cfg.season.history_seasons, current_season):
        games_k = _key("feat_games", s)
        if cache.has_df(games_k):
            past_games[s] = cache.load_df(games_k)

    elo_hist = build_elo_hist(
        season_games=past_games,
        current_season=current_season,
        base=cfg.elo.base,
        k=cfg.elo.k,
        home_advantage=cfg.elo.home_advantage,
        blend_recent=cfg.elo.blend_recent,
        blend_older=cfg.elo.blend_older,
    )

    # Current season games
    cur_games_k = _key("feat_games", current_season)
    if not cache.has_df(cur_games_k):
        return elo_hist if elo_hist else {}

    cur_games = cache.load_df(cur_games_k)
    if cur_games.empty:
        return elo_hist if elo_hist else {}

    result = build_current_season_elo(
        current_games_df=cur_games,
        elo_hist=elo_hist,
        base=cfg.elo.base,
        k=cfg.elo.k,
        home_advantage=cfg.elo.home_advantage,
    )

    cache.save_json(k_elo, result.final_elos)
    return result.final_elos


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_any_col(df: pd.DataFrame, candidates) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    raise KeyError(f"Could not find any of {candidates} in columns.")


def get_next_round_number(gamecodes_df: pd.DataFrame) -> int:
    """Find the next round to predict.

    1) If there are unplayed games: return the smallest round that has at least one.
    2) Otherwise (e.g. API returns only played games): return max(round) + 1.
    """
    round_col = _pick_any_col(gamecodes_df, ["Round", "round"])
    df = gamecodes_df.copy()
    df[round_col] = pd.to_numeric(df[round_col], errors="coerce")

    # Try: smallest round with at least one unplayed game
    played_col = _pick_any_col(gamecodes_df, ["played", "Played", "isPlayed"])
    if df[played_col].dtype != bool:
        df[played_col] = (
            df[played_col].astype(str).str.lower()
            .map({"true": True, "false": False}).fillna(False)
        )
    future = df.loc[~df[played_col]].copy()
    if not future.empty:
        return int(future[round_col].min())

    # Fallback: API often returns only played games (v1 results) → next = max_round + 1
    max_round = df[round_col].max()
    if pd.isna(max_round) or max_round < 0:
        raise ValueError(
            "Could not determine next round: gamecodes have no valid round numbers. "
            "Run: euroleague-sim update-data --season 2025"
        )
    return int(max_round) + 1


def build_round_schedule_v2(
    fetcher: EuroleagueFetcher,
    season: int,
    round_number: int,
) -> pd.DataFrame:
    """Get the schedule (home/away team codes) for a specific round.

    Uses v2 wrapper first; if column names don't match (v2 API shape), falls back to v3.
    """
    df = fetcher.gamecodes_round(season, round_number)
    if df.empty:
        return df

    cols = _find_schedule_columns(df)
    if cols:
        out = pd.DataFrame({
            "Round": int(round_number),
            "Gamecode": pd.to_numeric(df[cols["game"]], errors="coerce"),
            "home_team": df[cols["home"]].astype(str).str.strip(),
            "away_team": df[cols["away"]].astype(str).str.strip(),
        })
        out = out.dropna(subset=["Gamecode", "home_team", "away_team"])
        out["Gamecode"] = out["Gamecode"].astype(int)

        from .data.fetch import _extract_schedule_date
        _date_parsed = _extract_schedule_date(df)
        if _date_parsed is not None and len(_date_parsed) == len(out):
            out["game_date"] = _date_parsed.values
        else:
            out["game_date"] = pd.NaT

        return out.sort_values(["Round", "Gamecode"]).reset_index(drop=True)

    # v2 returned different column names → try v3
    df_v3 = fetcher.schedule_round_v3(season, round_number)
    if not df_v3.empty:
        return df_v3

    raise ValueError(
        "Could not find home/away team columns in round schedule. "
        f"Available columns: {list(df.columns)}. "
        "Check euroleague-api v2/v3 response structure."
    )


# ---------------------------------------------------------------------------
# Step 10: ML training pipeline
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Phase-1 additive features (player BPM + domestic fatigue)
# ---------------------------------------------------------------------------

def build_player_bpm_timeline(cache: Cache, season: int, force: bool = False) -> pd.DataFrame:
    """Build (and cache) the per team-game available-roster BPM timeline."""
    k_bpm = _key("feat_player_bpm", season)
    if not force and cache.has_df(k_bpm):
        return cache.load_df(k_bpm)

    k_bx = _key("raw_boxscore", season)
    k_tg = _key("feat_team_game", season)
    if not cache.has_df(k_bx) or not cache.has_df(k_tg):
        return pd.DataFrame(columns=["Season", "Gamecode", "Round", "Team", "team_bpm_available"])

    box = cache.load_df(k_bx)
    team_game = cache.load_df(k_tg)
    timeline = build_team_bpm_timeline(box, team_game)
    cache.save_df(k_bpm, timeline)
    return timeline


def build_extra_training_features(
    cache: Cache,
    cfg: ProjectConfig,
    current_season: int,
    verbose: bool = True,
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """Assemble ``{(season, gamecode): {net_bpm_diff, domestic_fatigue_diff}}``.

    Honours the ``cfg.features`` toggles; disabled features simply contribute
    nothing (defaulting to ``0.0`` downstream). Never raises — missing data
    degrades gracefully.
    """
    extra: Dict[Tuple[int, int], Dict[str, float]] = {}
    if not (cfg.features.use_player_bpm or cfg.features.use_domestic_fatigue):
        return extra

    start = current_season - cfg.season.history_seasons
    domestic_matches = None
    if cfg.features.use_domestic_fatigue and cache.has_df(cfg.features.domestic_cache_key):
        domestic_matches = cache.load_df(cfg.features.domestic_cache_key)

    for s in range(start, current_season + 1):
        games_k = _key("feat_games", s)
        if not cache.has_df(games_k):
            continue
        games_df = cache.load_df(games_k)
        if games_df.empty:
            continue

        if cfg.features.use_player_bpm:
            try:
                timeline = build_player_bpm_timeline(cache, s)
                bpm_diff = team_bpm_game_diff(games_df, timeline)
                for _, r in bpm_diff.iterrows():
                    extra.setdefault((s, int(r["Gamecode"])), {})["net_bpm_diff"] = float(r["net_bpm_diff"])
            except Exception as exc:  # noqa: BLE001 - never break training
                if verbose:
                    print(f"  [features] BPM skipped for season {s}: {exc}")

        if cfg.features.use_domestic_fatigue and domestic_matches is not None:
            try:
                sched = games_df.rename(columns={})[["Gamecode", "home_team", "away_team", "game_date"]]
                dom = domestic_fatigue_diff_for_schedule(
                    sched, domestic_matches, window_days=cfg.features.domestic_window_days,
                )
                for gc, val in dom.items():
                    extra.setdefault((s, int(gc)), {})["domestic_fatigue_diff"] = float(val)
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"  [features] domestic fatigue skipped for season {s}: {exc}")

    return extra


def build_extra_prediction_features(
    cache: Cache,
    cfg: ProjectConfig,
    season: int,
    schedule_df: pd.DataFrame,
) -> Dict[int, Dict[str, float]]:
    """Assemble ``{gamecode: {net_bpm_diff, domestic_fatigue_diff}}`` for a round."""
    extra: Dict[int, Dict[str, float]] = {}
    if not (cfg.features.use_player_bpm or cfg.features.use_domestic_fatigue):
        return extra

    if cfg.features.use_player_bpm:
        k_bx = _key("raw_boxscore", season)
        k_tg = _key("feat_team_game", season)
        if cache.has_df(k_bx) and cache.has_df(k_tg):
            try:
                current_bpm = compute_current_team_bpm(
                    cache.load_df(k_bx), cache.load_df(k_tg),
                )
                for gc, val in current_team_bpm_to_game_diff(schedule_df, current_bpm).items():
                    extra.setdefault(int(gc), {})["net_bpm_diff"] = float(val)
            except Exception as exc:  # noqa: BLE001
                print(f"  [features] current BPM skipped: {exc}")

    if cfg.features.use_domestic_fatigue and cache.has_df(cfg.features.domestic_cache_key):
        try:
            dom = domestic_fatigue_diff_for_schedule(
                schedule_df, cache.load_df(cfg.features.domestic_cache_key),
                window_days=cfg.features.domestic_window_days,
            )
            for gc, val in dom.items():
                extra.setdefault(int(gc), {})["domestic_fatigue_diff"] = float(val)
        except Exception as exc:  # noqa: BLE001
            print(f"  [features] domestic fatigue (predict) skipped: {exc}")

    return extra


def prepare_training_data(
    cache: Cache,
    cfg: ProjectConfig,
    current_season: int,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load cached seasonal features, run Elo, and build the shared training dataset."""
    seasons_data: Dict[int, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    start = current_season - cfg.season.history_seasons

    for s in range(start, current_season + 1):
        games_k = _key("feat_games", s)
        tg_k    = _key("feat_team_game", s)
        if not cache.has_df(games_k) or not cache.has_df(tg_k):
            if verbose:
                print(f"  [train] Skipping season {s}: cached features not found.")
            continue

        games_df     = cache.load_df(games_k)
        team_game_df = cache.load_df(tg_k)

        if games_df.empty:
            continue

        elo_result = run_elo(
            games_df,
            base=cfg.elo.base,
            k=cfg.elo.k,
            home_advantage=cfg.elo.home_advantage,
        )
        seasons_data[s] = (games_df, team_game_df, elo_result.game_elos)

    if not seasons_data:
        raise RuntimeError(
            "No training data available. Run `update-data` first for all seasons."
        )

    if verbose:
        total = sum(len(g) for g, _, _ in seasons_data.values())
        print(f"  [train] Building training dataset: {len(seasons_data)} season(s), "
              f"{total} games …")

    extra_features = build_extra_training_features(cache, cfg, current_season, verbose)
    meta = load_season_meta(Path("data_cache"), current_season)
    max_rounds = max_rounds_from_meta(meta)
    if verbose:
        print(f"  [train] round_progress normaliser max_rounds={max_rounds} "
              f"({'from season_meta' if meta else 'default'})")
    train_df = build_training_dataset(
        seasons_data,
        max_rounds=max_rounds,
        extra_features_by_gamecode=extra_features,
    )

    if verbose:
        print(f"  [train] Training dataset: {len(train_df)} rows x "
              f"{len(train_df.columns)} columns")
        
    return train_df


def train_ml_pipeline(
    cache: Cache,
    cfg: ProjectConfig,
    current_season: int,
    model_name: str = "baseline",
    verbose: bool = True,
) -> Dict[str, float]:
    """Build training data from multiple seasons, train linear models, save artefacts.

    Returns the evaluation metrics dict.
    """
    train_df = prepare_training_data(cache, cfg, current_season, verbose)

    model_dir = Path(cfg.ml.model_dir)
    metrics = train_models(
        train_df,
        model_dir=model_dir,
        model_name=model_name,
        seed=42,
        cv_folds=cfg.ml.cv_folds,
        wfo_min_train_size=cfg.ml.wfo_min_train_size,
        verbose=verbose,
    )
    return metrics


# ---------------------------------------------------------------------------
# Prediction (updated with ML integration)
# ---------------------------------------------------------------------------

def predict_next_round(
    cache: Cache,
    fetcher: EuroleagueFetcher,
    cfg: ProjectConfig,
    season: int,
    round_number: Optional[int] = None,
    model_name: str = "baseline",
    n_sims: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Full prediction pipeline for a given round.

    Steps:
    1) Ensure features (net ratings) are built.
    2) Compute EloCurrent (hist + current season update).
    3) Get schedule for the target round.
    4) Compute matchup features A, B.
    5) Run ML models (LogReg + Ridge) if trained artefacts are available.
    6) Run Monte Carlo for margin distribution (uses ML margin if available).
    7) Return combined predictions DataFrame.
    """
    # 1) Net ratings
    _, team_game, team_ratings, summ = build_features_for_season(
        cache, season, cfg=cfg, force=False,
    )

    # 2) EloCurrent
    current_elos = build_current_elo(cache, cfg, season, force=False)

    # 3) Round schedule
    gamecodes = cache.load_df(_key("raw_gamecodes", season))
    user_requested_round = round_number is not None
    if round_number is None:
        round_number = get_next_round_number(gamecodes)

    schedule = build_round_schedule_v2(fetcher, season, int(round_number))
    is_offseason = False
    offseason_fallback_round: Optional[int] = None
    if schedule.empty:
        # Off-season: `get_next_round_number` returned `max_played + 1` but the API
        # has no fixtures for it. Fall back to the last round that actually has
        # games so the dashboard never goes blank. Only do this when the caller
        # didn't pin a specific round.
        if user_requested_round:
            raise ValueError(f"No schedule found for season {season} round {round_number}")

        round_col = _pick_any_col(gamecodes, ["Round", "round"])
        last_played = pd.to_numeric(gamecodes[round_col], errors="coerce").max()
        if pd.isna(last_played):
            raise ValueError(
                f"No upcoming games and no played rounds found for season {season}."
            )
        offseason_fallback_round = int(last_played)
        schedule = build_round_schedule_v2(fetcher, season, offseason_fallback_round)
        if schedule.empty:
            raise ValueError(
                f"Off-season fallback to round {offseason_fallback_round} also "
                f"returned no schedule for season {season}."
            )
        round_number = offseason_fallback_round
        is_offseason = True

    # 4) Matchup features (A, B — for MC fallback when ML models not available)
    matchup = compute_matchup_features(
        schedule_df=schedule,
        team_ratings_df=team_ratings,
        current_elos=current_elos,
        elo_base=cfg.elo.base,
    )

    if "game_date" in schedule.columns:
        matchup = matchup.merge(
            schedule[["Gamecode", "game_date"]], on="Gamecode", how="left",
        )

    # 5) ML ensemble prediction
    model_dir = Path(cfg.ml.model_dir)
    predictor = load_predictor(model_dir, model_name=model_name)
    ml_margin = None

    if predictor is not None:
        extra_pred = build_extra_prediction_features(cache, cfg, season, schedule)
        meta = load_season_meta(Path("data_cache"), season)
        max_rounds = max_rounds_from_meta(meta)
        ml_features = build_prediction_features(
            schedule_df=schedule,
            team_ratings_df=team_ratings,
            current_elos=current_elos,
            team_game_df=team_game,
            round_number=int(round_number),
            elo_base=cfg.elo.base,
            max_rounds=max_rounds,
            extra_features_by_gamecode=extra_pred,
        )
        has_ml_ready = ml_features[FEATURE_COLS].notna().all(axis=1)

        for col in ("elo_diff_scaled", "net_bpm_diff", "domestic_fatigue_diff"):
            if col in ml_features.columns:
                matchup[col] = ml_features[col].values

        if has_ml_ready.any():
            ml_pred = predictor.predict(ml_features.loc[has_ml_ready])
            matchup.loc[has_ml_ready.values, "pHomeWin_ml"] = ml_pred["pHomeWin_ml"].values
            matchup.loc[has_ml_ready.values, "margin_ml"]   = ml_pred["margin_ml"].values

        if has_ml_ready.all():
            ml_margin = matchup["margin_ml"].values
        elif has_ml_ready.any():
            a_vals = matchup["A"].to_numpy(dtype=float)
            b_vals = matchup["B"].to_numpy(dtype=float)
            fallback = cfg.mc.alpha1 * a_vals + cfg.mc.alpha2 * b_vals + cfg.mc.alpha3
            ml_margin = np.where(
                has_ml_ready.values,
                matchup["margin_ml"].values,
                fallback,
            )

    # 6) Monte Carlo simulation
    sigma_eff = cfg.mc.sigma
    if summ.get("margin_sigma_points"):
        sigma_from_data = float(summ["margin_sigma_points"])
        if sigma_from_data > 0:
            sigma_eff = sigma_from_data

    n_sims_eff = int(n_sims or cfg.mc.n_sims)

    pred = simulate_next_round(
        matchup_df=matchup,
        alpha1=cfg.mc.alpha1,
        alpha2=cfg.mc.alpha2,
        alpha3=cfg.mc.alpha3,
        sigma=sigma_eff,
        n_sims=n_sims_eff,
        seed=seed,
        mu_override=ml_margin,
    )

    # 7) Clean output
    output_cols = [
        "Round", "Gamecode", "game_date", "home_team", "away_team",
        # ML
        "pHomeWin_ml",
        # Monte Carlo
        "pHomeWin", "muMargin", "meanMargin",
        "q10", "q50", "q90",
        # Context
        "EloCurrent_home", "EloCurrent_away",
        "elo_diff_scaled", "net_bpm_diff", "domestic_fatigue_diff",
        "A", "B", "margin_ml", "n_sims",
    ]
    output_cols = [c for c in output_cols if c in pred.columns]
    out = pred[output_cols].copy()
    out["is_offseason"] = bool(is_offseason)
    return out


def save_predictions(pred_df: pd.DataFrame, season: int, round_number: int, output_dir: Path) -> Path:
    """Save predictions CSV to outputs/ folder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = f"round_{round_number}_predictions.csv"
    path = output_dir / fname
    pred_df.to_csv(path, index=False)
    return path
