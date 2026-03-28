"""EuroLeague game-context helpers: schedule-density rest between EL games."""

from __future__ import annotations

from typing import List

import pandas as pd

# ---------------------------------------------------------------------------
# EuroLeague rest-days (schedule density)
# ---------------------------------------------------------------------------

REST_CAP_DAYS = 4


def compute_team_el_rest(
    team_game_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add ``el_rest_days`` per team-game row (EuroLeague schedule density).

    For each team, ``el_rest_days`` is the number of calendar days since the
    team's previous EuroLeague game, clipped to ``REST_CAP_DAYS``.

    Round 1 / first appearance → ``el_rest_days = REST_CAP_DAYS`` (capped
    maximum, equivalent to "well-rested within the EuroLeague schedule").

    Requires ``game_date`` (datetime-like) and ``Team`` columns.
    """
    df = team_game_df.sort_values(["Team", "game_date", "Round", "Gamecode"]).copy()
    df["_prev_date"] = df.groupby("Team")["game_date"].shift(1)
    df["el_rest_days"] = (df["game_date"] - df["_prev_date"]).dt.days
    df["el_rest_days"] = df["el_rest_days"].clip(upper=REST_CAP_DAYS).fillna(REST_CAP_DAYS)
    df.drop(columns=["_prev_date"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Game-level context features (rest differential)
# ---------------------------------------------------------------------------

def build_game_context(
    games_df: pd.DataFrame,
    team_game_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return a per-game DataFrame with ``el_rest_days_diff`` for ML.

    ``el_rest_days_diff`` is capped home rest minus capped away rest.
    """
    tg = compute_team_el_rest(team_game_df)

    idx = tg.set_index(["Team", "Gamecode", "IsHome"])

    rows: List[dict] = []
    for _, game in games_df.iterrows():
        gc = int(game["Gamecode"])
        ht = str(game["home_team"])
        at = str(game["away_team"])

        try:
            h_row = idx.loc[(ht, gc, 1)]
        except KeyError:
            rows.append(_empty_context(gc))
            continue
        try:
            a_row = idx.loc[(at, gc, 0)]
        except KeyError:
            rows.append(_empty_context(gc))
            continue

        h_rest = float(h_row["el_rest_days"])
        a_rest = float(a_row["el_rest_days"])

        rows.append({
            "Gamecode": gc,
            "el_rest_days_diff": h_rest - a_rest,
        })

    return pd.DataFrame(rows)


def _empty_context(gamecode: int) -> dict:
    return {
        "Gamecode": gamecode,
        "el_rest_days_diff": 0.0,
    }


# ---------------------------------------------------------------------------
# Prediction-time context (upcoming games)
# ---------------------------------------------------------------------------

def build_prediction_context(
    schedule_df: pd.DataFrame,
    team_game_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute rest context for upcoming (unplayed) games.

    Uses the played ``team_game_df`` to find each team's most recent game date,
    then compares calendar gaps to the scheduled game date (capped).
    """
    tg = team_game_df.copy()
    if "game_date" not in tg.columns or tg["game_date"].isna().all():
        return _fallback_prediction_context(schedule_df)

    tg_sorted = tg.sort_values(["Team", "game_date", "Round", "Gamecode"])
    latest = tg_sorted.groupby("Team").last()

    rows: List[dict] = []
    for _, game in schedule_df.iterrows():
        gc = int(game["Gamecode"])
        ht = str(game["home_team"])
        at = str(game["away_team"])
        gd = game.get("game_date", pd.NaT)

        h_rest = _pred_rest(latest, ht, gd)
        a_rest = _pred_rest(latest, at, gd)

        rows.append({
            "Gamecode": gc,
            "el_rest_days_diff": h_rest - a_rest,
        })

    return pd.DataFrame(rows)


def _pred_rest(latest: pd.DataFrame, team: str, game_date) -> float:
    if team not in latest.index or pd.isna(game_date):
        return float(REST_CAP_DAYS)
    prev_date = latest.loc[team, "game_date"]
    if pd.isna(prev_date):
        return float(REST_CAP_DAYS)
    gap = (pd.Timestamp(game_date) - pd.Timestamp(prev_date)).days
    return float(min(gap, REST_CAP_DAYS))


def _fallback_prediction_context(schedule_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Gamecode": schedule_df["Gamecode"].values,
        "el_rest_days_diff": 0.0,
    })
