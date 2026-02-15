from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np

from .config import ProjectConfig
from .data.cache import Cache
from .data.fetch import EuroleagueFetcher, FetchParams
from .features.possessions import compute_team_possessions_from_boxscore, _pick_col
from .features.net_rating import (
    build_games_with_possessions,
    build_team_game_net_ratings,
    aggregate_team_ratings,
    season_summary,
)
from .features.elo import (
    build_elo_prior_from_past_seasons,
    build_elo_hist,
    build_current_season_elo,
)
from .sim.model import compute_matchup_features, logistic_projection
from .sim.engine import simulate_next_round


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

def build_elo_prior(
    cache: Cache,
    cfg: ProjectConfig,
    current_season: int,
    force: bool = False,
) -> pd.DataFrame:
    """Build EloHist from past seasons and return DataFrame(Team, EloPrior)."""
    k = _key("feat_elo_prior", current_season)
    if not force and cache.has_df(k):
        return cache.load_df(k)

    season_games: Dict[int, pd.DataFrame] = {}
    for s in range(current_season - cfg.season.history_seasons, current_season):
        games_k = _key("feat_games", s)
        if not cache.has_df(games_k):
            raise FileNotFoundError(
                f"Missing cached games for season {s}. Run update-data first."
            )
        season_games[s] = cache.load_df(games_k)

    elo_prior = build_elo_prior_from_past_seasons(
        season_games=season_games,
        current_season=current_season,
        base=cfg.elo.base,
        k=cfg.elo.k,
        home_advantage=cfg.elo.home_advantage,
        blend_recent=cfg.elo.blend_recent,
        blend_older=cfg.elo.blend_older,
    )
    cache.save_df(k, elo_prior)
    return elo_prior


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
    """Find the smallest round with at least one unplayed game."""
    played_col = _pick_any_col(gamecodes_df, ["played", "Played", "isPlayed"])
    round_col = _pick_any_col(gamecodes_df, ["Round", "round"])
    df = gamecodes_df.copy()
    if df[played_col].dtype != bool:
        df[played_col] = (
            df[played_col].astype(str).str.lower()
            .map({"true": True, "false": False}).fillna(False)
        )

    future = df.loc[~df[played_col]].copy()
    if future.empty:
        raise ValueError("No future (unplayed) games found in gamecodes season data.")

    future[round_col] = pd.to_numeric(future[round_col], errors="coerce")
    return int(future[round_col].min())


def build_round_schedule_v2(
    fetcher: EuroleagueFetcher,
    season: int,
    round_number: int,
) -> pd.DataFrame:
    """Get the schedule (home/away team codes) for a specific round."""
    df = fetcher.gamecodes_round(season, round_number)
    if df.empty:
        return df

    game_col = _pick_any_col(df, ["gameCode", "Gamecode", "GameCode", "gamecode"])
    round_col = _pick_any_col(df, ["Round", "round"])

    home_col = None
    away_col = None
    for c in df.columns:
        cl = c.lower()
        if home_col is None and ("hometeam" in cl and cl.endswith(".code")):
            home_col = c
        if away_col is None and ("awayteam" in cl and cl.endswith(".code")):
            away_col = c

    if home_col is None:
        home_col = _pick_any_col(df, ["homeTeam.code", "homeTeamCode"])
    if away_col is None:
        away_col = _pick_any_col(df, ["awayTeam.code", "awayTeamCode"])

    out = pd.DataFrame({
        "Round": pd.to_numeric(df[round_col], errors="coerce").astype(int),
        "Gamecode": pd.to_numeric(df[game_col], errors="coerce").astype(int),
        "home_team": df[home_col].astype(str),
        "away_team": df[away_col].astype(str),
    })

    for cand, outcol in [
        (["gameDate", "date", "startDate"], "gameDate"),
        (["gameTime", "time", "startTime"], "gameTime"),
    ]:
        for c in df.columns:
            if c in cand:
                out[outcol] = df[c]
                break

    return out.sort_values(["Round", "Gamecode"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 8-9: Prediction pipeline
# ---------------------------------------------------------------------------

def predict_next_round(
    cache: Cache,
    fetcher: EuroleagueFetcher,
    cfg: ProjectConfig,
    season: int,
    round_number: Optional[int] = None,
    n_sims: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Full prediction pipeline for a given round.

    Steps:
    1) Ensure features (net ratings) are built.
    2) Compute EloCurrent (hist + current season update).
    3) Get schedule for the target round.
    4) Compute matchup features A, B.
    5) Run logistic projection for P(HomeWin).
    6) Run Monte Carlo for margin distribution.
    7) Return combined predictions DataFrame.
    """
    # 1) Net ratings
    _, _, team_ratings, summ = build_features_for_season(cache, season, cfg=cfg, force=False)

    # 2) EloCurrent
    current_elos = build_current_elo(cache, cfg, season, force=False)

    # 3) Round schedule
    gamecodes = cache.load_df(_key("raw_gamecodes", season))
    if round_number is None:
        round_number = get_next_round_number(gamecodes)

    schedule = build_round_schedule_v2(fetcher, season, int(round_number))
    if schedule.empty:
        raise ValueError(f"No schedule found for season {season} round {round_number}")

    # 4) Matchup features
    matchup = compute_matchup_features(
        schedule_df=schedule,
        team_ratings_df=team_ratings,
        current_elos=current_elos,
        elo_base=cfg.elo.base,
    )

    # 5) Logistic projection
    matchup = logistic_projection(
        matchup,
        w1=cfg.projection.w1,
        w2=cfg.projection.w2,
        w3=cfg.projection.w3,
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
    )

    # 7) Clean output
    output_cols = [
        "Round", "Gamecode", "home_team", "away_team",
        "pHomeWin_logistic", "pHomeWin",
        "muMargin", "meanMargin", "q10", "q50", "q90",
        "EloCurrent_home", "EloCurrent_away",
        "A", "B", "n_sims",
    ]
    output_cols = [c for c in output_cols if c in pred.columns]
    return pred[output_cols].copy()


def save_predictions(pred_df: pd.DataFrame, season: int, round_number: int, output_dir: Path) -> Path:
    """Save predictions CSV to outputs/ folder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = f"round_{round_number}_predictions.csv"
    path = output_dir / fname
    pred_df.to_csv(path, index=False)
    return path
