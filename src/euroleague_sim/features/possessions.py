from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns found: {candidates}")


def compute_team_possessions_from_boxscore(player_boxscore_df: pd.DataFrame) -> pd.DataFrame:
    """Compute possessions from boxscore totals (team totals per game).

    Expects output of `BoxScoreData.get_player_boxscore_stats_single_season`.
    Uses the same formulas as the wrapper's `TeamStats.get_team_advanced_stats_single_game`.

    Returns a team-game table with possessions.
    """
    df = player_boxscore_df.copy()

    # Identify total rows (team totals)
    total_mask = False
    if "Player_ID" in df.columns:
        total_mask = df["Player_ID"].astype(str).str.lower().eq("total")
    elif "Player" in df.columns:
        total_mask = df["Player"].astype(str).str.lower().eq("total")
    else:
        raise KeyError("Could not find Player_ID or Player columns to select totals.")

    totals = df.loc[total_mask].copy()
    if totals.empty:
        raise ValueError("No totals rows found in boxscore dataframe.")

    # Required id columns (should exist)
    season_col = _pick_col(totals, ["Season", "season"])
    game_col = _pick_col(totals, ["Gamecode", "gamecode", "GameCode", "gameCode"])
    team_col = _pick_col(totals, ["Team", "team", "CODETEAM"])
    home_col = _pick_col(totals, ["Home", "home", "IsHome", "isHome"])

    # Stat columns (wrapper uses these exact names)
    fga2 = _pick_col(totals, ["FieldGoalsAttempted2", "FGA2"])
    fga3 = _pick_col(totals, ["FieldGoalsAttempted3", "FGA3"])
    fgm2 = _pick_col(totals, ["FieldGoalsMade2", "FGM2"])
    fgm3 = _pick_col(totals, ["FieldGoalsMade3", "FGM3"])
    fta = _pick_col(totals, ["FreeThrowsAttempted", "FTA"])
    orb = _pick_col(totals, ["OffensiveRebounds", "ORB"])
    drb = _pick_col(totals, ["DefensiveRebounds", "DRB"])
    tov = _pick_col(totals, ["Turnovers", "TOV"])

    # numeric
    for c in [fga2, fga3, fgm2, fgm3, fta, orb, drb, tov]:
        totals[c] = pd.to_numeric(totals[c], errors="coerce").fillna(0.0)

    totals[home_col] = pd.to_numeric(totals[home_col], errors="coerce").fillna(0).astype(int)

    # compute possessions per team
    totals["FGA"] = totals[fga2] + totals[fga3]
    totals["FGM"] = totals[fgm2] + totals[fgm3]
    totals["possessions_simple"] = totals["FGA"] + 0.44 * totals[fta] - totals[orb] + totals[tov]

    # Need opponent DRB per game to compute the full possessions formula.
    # We'll self-join within each game: opp_drb = total_drb of the other team.
    opp_drb = (
        totals[[season_col, game_col, team_col, drb]]
        .rename(columns={team_col: "opp_team", drb: "opp_drb"})
    )
    merged = totals.merge(
        opp_drb,
        on=[season_col, game_col],
        how="left",
        suffixes=("", "")
    )
    merged = merged.loc[merged["opp_team"] != merged[team_col]].copy()
    # There should be exactly one opp row per team-game
    merged.sort_values([season_col, game_col, team_col], inplace=True)
    merged = merged.drop_duplicates(subset=[season_col, game_col, team_col], keep="first")

    # Full formula:
    # poss = FGA + 0.44*FTA - 1.07*(ORB/oppDRB)*(FGA-FGM) + TOV
    merged["orb_over_oppdrb"] = np.where(
        merged["opp_drb"].astype(float) > 0,
        merged[orb].astype(float) / merged["opp_drb"].astype(float),
        0.0
    )
    merged["possessions"] = (
        merged["FGA"]
        + 0.44 * merged[fta]
        - 1.07 * merged["orb_over_oppdrb"] * (merged["FGA"] - merged["FGM"])
        + merged[tov]
    )

    # Four Factors raw stats (standardised names)
    merged["FGM2_std"] = merged[fgm2].astype(float)
    merged["FGM3_std"] = merged[fgm3].astype(float)
    merged["FTA_std"]  = merged[fta].astype(float)
    merged["ORB_std"]  = merged[orb].astype(float)
    merged["TOV_std"]  = merged[tov].astype(float)

    out_cols = [
        season_col, game_col, team_col, home_col,
        "possessions", "possessions_simple",
        "FGA", "FGM2_std", "FGM3_std", "FTA_std",
        "ORB_std", "TOV_std", "opp_drb",
    ]
    out = merged[out_cols].copy()
    out.rename(columns={
        season_col: "Season",
        game_col: "Gamecode",
        team_col: "Team",
        home_col: "Home",
        "FGM2_std": "FGM2",
        "FGM3_std": "FGM3",
        "FTA_std":  "FTA",
        "ORB_std":  "ORB",
        "TOV_std":  "TOV",
        "opp_drb":  "opp_DRB",
    }, inplace=True)

    # Sanity
    out["Season"] = out["Season"].astype(int)
    out["Gamecode"] = out["Gamecode"].astype(int)
    out["Home"] = out["Home"].astype(int)

    return out
