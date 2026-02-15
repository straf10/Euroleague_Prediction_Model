from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class EloResult:
    game_elos: pd.DataFrame   # per-game Elo log
    final_elos: Dict[str, float]  # team -> final Elo after all games


def _expected_score(elo_home: float, elo_away: float) -> float:
    """P(home win) = 1 / (1 + 10^(-((EloH + HCA) - EloA) / 400))."""
    return 1.0 / (1.0 + 10.0 ** ((elo_away - elo_home) / 400.0))


def run_elo(
    games_df: pd.DataFrame,
    initial_elos: Dict[str, float] | None = None,
    base: float = 1500.0,
    k: float = 20.0,
    home_advantage: float = 65.0,
) -> EloResult:
    """Run Elo over a chronologically sorted list of games.

    games_df needs columns:
      - home_team, away_team, home_points, away_points
      - Round, Gamecode (for ordering)

    initial_elos: if provided, used instead of base for known teams.
    """
    required = ["home_team", "away_team", "home_points", "away_points", "Round", "Gamecode"]
    missing = [c for c in required if c not in games_df.columns]
    if missing:
        raise KeyError(f"Missing columns for Elo: {missing}")

    gdf = games_df.copy()
    gdf = gdf.sort_values(["Round", "Gamecode"]).reset_index(drop=True)

    teams = pd.unique(pd.concat([gdf["home_team"], gdf["away_team"]], ignore_index=True))

    elo: Dict[str, float] = {}
    for t in teams:
        t_str = str(t)
        if initial_elos and t_str in initial_elos:
            elo[t_str] = float(initial_elos[t_str])
        else:
            elo[t_str] = float(base)

    rows: List[dict] = []
    for _, g in gdf.iterrows():
        ht = str(g["home_team"])
        at = str(g["away_team"])
        hp = float(g["home_points"])
        ap = float(g["away_points"])

        elo_home_pre = elo.get(ht, base)
        elo_away_pre = elo.get(at, base)

        elo_home_adj = elo_home_pre + home_advantage
        exp_home = _expected_score(elo_home_adj, elo_away_pre)

        outcome_home = 1.0 if hp > ap else 0.0
        delta = k * (outcome_home - exp_home)

        elo[ht] = elo_home_pre + delta
        elo[at] = elo_away_pre - delta

        rows.append({
            "Round": int(g["Round"]),
            "Gamecode": int(g["Gamecode"]),
            "home_team": ht,
            "away_team": at,
            "home_points": hp,
            "away_points": ap,
            "elo_home_pre": elo_home_pre,
            "elo_away_pre": elo_away_pre,
            "exp_home": exp_home,
            "outcome_home": outcome_home,
            "elo_home_post": elo[ht],
            "elo_away_post": elo[at],
            "delta": delta,
        })

    return EloResult(game_elos=pd.DataFrame(rows), final_elos=elo)


def build_elo_hist(
    season_games: Dict[int, pd.DataFrame],
    current_season: int,
    base: float = 1500.0,
    k: float = 20.0,
    home_advantage: float = 65.0,
    blend_recent: float = 0.65,
    blend_older: float = 0.35,
) -> Dict[str, float]:
    """Compute EloHist from the 2 most recent past seasons.

    Per spec section 6.3:
        EloHist = blend_recent * Elo_{season-1} + blend_older * Elo_{season-2}

    No regression-to-mean is applied; the raw final Elo of each season is used.
    """
    past_seasons = sorted(
        [s for s in season_games.keys() if s < current_season], reverse=True
    )[:2]

    if not past_seasons:
        return {}

    finals: Dict[int, Dict[str, float]] = {}
    for s in past_seasons:
        res = run_elo(season_games[s], base=base, k=k, home_advantage=home_advantage)
        finals[s] = res.final_elos

    all_teams = set()
    for d in finals.values():
        all_teams.update(d.keys())

    if len(past_seasons) == 1:
        return finals[past_seasons[0]]

    s_recent, s_older = past_seasons[0], past_seasons[1]
    elo_hist: Dict[str, float] = {}
    for t in all_teams:
        e_recent = finals[s_recent].get(t, base)
        e_older = finals[s_older].get(t, base)
        elo_hist[t] = blend_recent * e_recent + blend_older * e_older

    return elo_hist


def build_current_season_elo(
    current_games_df: pd.DataFrame,
    elo_hist: Dict[str, float],
    base: float = 1500.0,
    k: float = 20.0,
    home_advantage: float = 65.0,
) -> EloResult:
    """Run Elo on current season, initializing from EloHist (spec section 6.4).

    EloCurrent starts at EloHist for each team, then updates game-by-game
    through all played games in the current season.
    """
    return run_elo(
        current_games_df,
        initial_elos=elo_hist,
        base=base,
        k=k,
        home_advantage=home_advantage,
    )


def build_elo_prior_from_past_seasons(
    season_games: Dict[int, pd.DataFrame],
    current_season: int,
    base: float = 1500.0,
    k: float = 20.0,
    home_advantage: float = 65.0,
    blend_recent: float = 0.65,
    blend_older: float = 0.35,
) -> pd.DataFrame:
    """Build Elo prior table (Team, EloPrior) for the current season.

    Convenience wrapper that returns a DataFrame compatible with pipeline usage.
    """
    elo_hist = build_elo_hist(
        season_games=season_games,
        current_season=current_season,
        base=base, k=k, home_advantage=home_advantage,
        blend_recent=blend_recent, blend_older=blend_older,
    )
    if not elo_hist:
        return pd.DataFrame(columns=["Team", "EloPrior"])

    return pd.DataFrame({
        "Team": list(elo_hist.keys()),
        "EloPrior": list(elo_hist.values()),
    })
