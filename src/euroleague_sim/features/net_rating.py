from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

from .possessions import _pick_col


@dataclass(frozen=True)
class SeasonSummary:
    possessions_per_game: float
    margin_sigma_points: float
    league_home_adv_points: float  # average home margin (points)
    league_home_adv_net100: float  # average home net/100 poss


def build_games_with_possessions(
    gamecodes_df: pd.DataFrame,
    team_poss_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge game metadata (scores/round) with home/away team codes + possessions."""
    gc = gamecodes_df.copy()
    poss = team_poss_df.copy()

    # columns from wrapper's get_gamecodes_season
    season_col = _pick_col(gc, ["Season", "season"]) if "Season" in gc.columns or "season" in gc.columns else None
    game_col = _pick_col(gc, ["gameCode", "Gamecode", "GameCode", "gamecode"])
    round_col = _pick_col(gc, ["Round", "round"])
    phase_col = _pick_col(gc, ["Phase", "phase"])
    played_col = _pick_col(gc, ["played", "Played", "isPlayed"])
    hs_col = _pick_col(gc, ["homescore", "homeScore", "HomeScore", "homescore"])
    as_col = _pick_col(gc, ["awayscore", "awayScore", "AwayScore", "awayscore"])

    # Parse game date if available (v1 results expose 'date' as e.g. "Sep 30, 2025")
    _date_candidates = ["date", "Date", "gameDate"]
    _date_col = None
    for _dc in _date_candidates:
        if _dc in gc.columns:
            _date_col = _dc
            break
    if _date_col is not None:
        gc["game_date"] = pd.to_datetime(gc[_date_col], errors="coerce").dt.normalize()
    else:
        gc["game_date"] = pd.NaT

    # Ensure types
    gc[game_col] = pd.to_numeric(gc[game_col], errors="coerce").astype(int)
    gc[round_col] = pd.to_numeric(gc[round_col], errors="coerce")
    gc[hs_col] = pd.to_numeric(gc[hs_col], errors="coerce")
    gc[as_col] = pd.to_numeric(gc[as_col], errors="coerce")
    # played might be boolean already or strings
    if gc[played_col].dtype != bool:
        gc[played_col] = gc[played_col].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)

    # Only played games (scores exist)
    played_gc = gc.loc[gc[played_col]].copy()

    # home/away team codes come from possessions table (Team + Home flag)
    # Four Factors columns (available after possessions rebuild)
    ff_cols = [
        "FGA", "FGM", "FGM2", "FGM3", "FTA", "ORB", "DRB", "TOV",
        "opp_DRB", "opp_ORB", "opp_FGA", "opp_FGM", "opp_FGM3", "opp_FTA", "opp_TOV",
    ]
    avail_ff = [c for c in ff_cols if c in poss.columns]

    base_cols = ["Season", "Gamecode", "Team", "possessions", "possessions_simple"]
    base_rename_h = {"Team": "home_team", "possessions": "home_poss", "possessions_simple": "home_poss_simple"}
    base_rename_a = {"Team": "away_team", "possessions": "away_poss", "possessions_simple": "away_poss_simple"}
    for c in avail_ff:
        base_rename_h[c] = f"home_{c}"
        base_rename_a[c] = f"away_{c}"

    home = poss.loc[poss["Home"] == 1, base_cols + avail_ff].rename(columns=base_rename_h)
    away = poss.loc[poss["Home"] == 0, base_cols + avail_ff].rename(columns=base_rename_a)

    # join by Season+Gamecode (wrapper uses Gamecode naming)
    played_gc = played_gc.rename(columns={game_col: "Gamecode"})
    if season_col:
        played_gc = played_gc.rename(columns={season_col: "Season"})
    else:
        # if season not present, infer from poss
        played_gc["Season"] = poss["Season"].iloc[0]

    out = played_gc.merge(home, on=["Season", "Gamecode"], how="left").merge(away, on=["Season", "Gamecode"], how="left")

    out["home_points"] = out[hs_col]
    out["away_points"] = out[as_col]
    out["Round"] = out[round_col].astype(int)
    out["Phase"] = out[phase_col].astype(str)

    # possessions per game (use average of teams)
    out["possessions_game"] = out[["home_poss", "away_poss"]].mean(axis=1)

    out["margin_home"] = out["home_points"] - out["away_points"]
    ff_game_cols = [f"home_{c}" for c in avail_ff] + [f"away_{c}" for c in avail_ff]
    return out[
        [
            "Season", "Gamecode", "Phase", "Round", "game_date",
            "home_team", "away_team",
            "home_points", "away_points",
            "home_poss", "away_poss", "possessions_game",
            "margin_home",
        ] + ff_game_cols
    ]


def build_team_game_net_ratings(games_df: pd.DataFrame) -> pd.DataFrame:
    """Expands games_df (one row per game) into team-game rows with net rating per 100 possessions."""

    # Four Factors columns (present after possessions rebuild with --force)
    ff_raw = [
        "FGA", "FGM", "FGM2", "FGM3", "FTA", "ORB", "DRB", "TOV",
        "opp_DRB", "opp_ORB", "opp_FGA", "opp_FGM", "opp_FGM3", "opp_FTA", "opp_TOV",
    ]
    has_ff = all(f"home_{c}" in games_df.columns for c in ff_raw)

    rows = []

    for _, g in games_df.iterrows():
        poss = float(g["possessions_game"]) if pd.notnull(g["possessions_game"]) else np.nan
        if not np.isfinite(poss) or poss <= 0:
            continue

        _game_date = g.get("game_date", pd.NaT)

        base_home = {
            "Season": int(g["Season"]),
            "Gamecode": int(g["Gamecode"]),
            "Round": int(g["Round"]),
            "Phase": str(g["Phase"]),
            "game_date": _game_date,
            "Team": str(g["home_team"]),
            "Opponent": str(g["away_team"]),
            "IsHome": 1,
            "PointsFor": float(g["home_points"]),
            "PointsAgainst": float(g["away_points"]),
            "Possessions": poss,
        }
        base_away = {
            "Season": int(g["Season"]),
            "Gamecode": int(g["Gamecode"]),
            "Round": int(g["Round"]),
            "Phase": str(g["Phase"]),
            "game_date": _game_date,
            "Team": str(g["away_team"]),
            "Opponent": str(g["home_team"]),
            "IsHome": 0,
            "PointsFor": float(g["away_points"]),
            "PointsAgainst": float(g["home_points"]),
            "Possessions": poss,
        }

        if has_ff:
            for c in ff_raw:
                base_home[c] = float(g.get(f"home_{c}", 0))
                base_away[c] = float(g.get(f"away_{c}", 0))

        rows.append(base_home)
        rows.append(base_away)

    tdf = pd.DataFrame(rows)
    if tdf.empty:
        return tdf

    tdf["NetPer100"] = 100.0 * (tdf["PointsFor"] - tdf["PointsAgainst"]) / tdf["Possessions"]
    tdf["OffPer100"] = 100.0 * tdf["PointsFor"] / tdf["Possessions"]
    tdf["DefPer100"] = 100.0 * tdf["PointsAgainst"] / tdf["Possessions"]
    return tdf


def aggregate_team_ratings(
    team_game_df: pd.DataFrame,
    shrink_possessions: float = 400.0,
    shrink_target_net100: float = 0.0
) -> pd.DataFrame:
    """Aggregates net ratings overall and home/away, with shrinkage on NetPer100."""
    if team_game_df.empty:
        return team_game_df

    def agg(group: pd.DataFrame) -> Dict[str, float]:
        poss = group["Possessions"].sum()
        pf = group["PointsFor"].sum()
        pa = group["PointsAgainst"].sum()
        net = 100.0 * (pf - pa) / poss if poss > 0 else 0.0

        # shrinkage: treat prior as shrink_possessions with target net
        net_shrunk = (net * poss + shrink_target_net100 * shrink_possessions) / (poss + shrink_possessions) if (poss + shrink_possessions) > 0 else net
        off = 100.0 * pf / poss if poss > 0 else 0.0
        deff = 100.0 * pa / poss if poss > 0 else 0.0
        return {
            "Games": float(len(group)),
            "Possessions": float(poss),
            "OffPer100": float(off),
            "DefPer100": float(deff),
            "NetPer100_raw": float(net),
            "NetPer100": float(net_shrunk),
        }

    overall = team_game_df.groupby(["Season", "Team"], as_index=False).apply(lambda g: pd.Series(agg(g))).reset_index()
    overall = overall.drop(columns=["index"]) if "index" in overall.columns else overall

    home = team_game_df[team_game_df["IsHome"] == 1].groupby(["Season", "Team"], as_index=False) \
        .apply(lambda g: pd.Series(agg(g))).reset_index()
    home = home.drop(columns=["index"]) if "index" in home.columns else home
    home = home.rename(columns={c: f"Home_{c}" for c in home.columns if c not in ["Season", "Team"]})

    away = team_game_df[team_game_df["IsHome"] == 0].groupby(["Season", "Team"], as_index=False) \
        .apply(lambda g: pd.Series(agg(g))).reset_index()
    away = away.drop(columns=["index"]) if "index" in away.columns else away
    away = away.rename(columns={c: f"Away_{c}" for c in away.columns if c not in ["Season", "Team"]})

    out = overall.merge(home, on=["Season", "Team"], how="left").merge(away, on=["Season", "Team"], how="left")

    # if home/away missing (no games), fill with overall
    for prefix in ["Home_", "Away_"]:
        for col in ["NetPer100", "OffPer100", "DefPer100", "Possessions", "Games"]:
            c = f"{prefix}{col}"
            if c in out.columns:
                out[c] = out[c].fillna(out[col] if col in out.columns else 0.0)

    return out


def season_summary(games_df: pd.DataFrame) -> SeasonSummary:
    if games_df.empty:
        return SeasonSummary(possessions_per_game=72.0, margin_sigma_points=12.0, league_home_adv_points=0.0, league_home_adv_net100=0.0)

    poss_per_game = float(games_df["possessions_game"].mean())
    margin_sigma = float(games_df["margin_home"].std(ddof=0))
    home_adv_points = float(games_df["margin_home"].mean())
    home_adv_net100 = float(100.0 * home_adv_points / poss_per_game) if poss_per_game > 0 else 0.0
    return SeasonSummary(
        possessions_per_game=poss_per_game,
        margin_sigma_points=margin_sigma if np.isfinite(margin_sigma) and margin_sigma > 0 else 12.0,
        league_home_adv_points=home_adv_points,
        league_home_adv_net100=home_adv_net100
    )
