from __future__ import annotations

from typing import Dict
import pandas as pd
import numpy as np


def compute_matchup_features(
    schedule_df: pd.DataFrame,
    team_ratings_df: pd.DataFrame,
    current_elos: Dict[str, float],
    elo_base: float = 1500.0,
) -> pd.DataFrame:
    """Build per-matchup features A and B for every game in the schedule.

    A = HomeNetRtg_homeSplit(home) - AwayNetRtg_awaySplit(away)
    B = (EloCurrent(home) - EloCurrent(away)) / 25

    Returns schedule_df augmented with columns: A, B, EloCurrent_home, EloCurrent_away.
    """
    df = schedule_df.copy()
    tr = team_ratings_df.set_index("Team") if "Team" in team_ratings_df.columns else team_ratings_df

    home_net = df["home_team"].map(
        tr["Home_NetPer100"] if "Home_NetPer100" in tr.columns else tr.get("NetPer100", pd.Series(dtype=float))
    ).fillna(0.0)

    away_net = df["away_team"].map(
        tr["Away_NetPer100"] if "Away_NetPer100" in tr.columns else tr.get("NetPer100", pd.Series(dtype=float))
    ).fillna(0.0)

    df["A"] = home_net.values - away_net.values

    elo_home = df["home_team"].map(current_elos).fillna(elo_base)
    elo_away = df["away_team"].map(current_elos).fillna(elo_base)
    df["EloCurrent_home"] = elo_home.values
    df["EloCurrent_away"] = elo_away.values
    df["B"] = (elo_home.values - elo_away.values) / 25.0

    return df
