"""EuroLeague game-context helpers: team coordinates, haversine travel,
schedule-density rest, and venue resolution including neutral-site overrides.

All coordinates are (latitude, longitude) in decimal degrees.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Team home-base coordinates keyed by EuroLeague 3-letter code
# ---------------------------------------------------------------------------

TEAM_COORDS: Dict[str, Tuple[float, float]] = {
    "ASV": (45.7640, 4.8357),      # LDLC ASVEL — Lyon
    "BAR": (41.3809, 2.1228),      # FC Barcelona — Barcelona
    "BAS": (42.8467, -2.6726),     # Baskonia — Vitoria-Gasteiz
    "BER": (52.5200, 13.4050),     # Alba Berlin — Berlin
    "DUB": (25.2048, 55.2708),     # Dubai Basketball — Dubai
    "HTA": (32.1093, 34.8007),     # Hapoel Tel Aviv — Tel Aviv
    "IST": (41.0082, 28.9784),     # Anadolu Efes — Istanbul
    "MAD": (40.4530, -3.6883),     # Real Madrid — Madrid
    "MCO": (43.7384, 7.4246),      # AS Monaco — Monaco
    "MIL": (45.5185, 9.2117),      # EA7 Milan — Milan (Assago/Forum)
    "MUN": (48.1742, 11.5544),     # FC Bayern Munich — Munich
    "OLY": (37.9414, 23.6702),     # Olympiacos — Piraeus
    "PAM": (39.4699, -0.3763),     # Valencia Basket — Valencia
    "PAN": (37.9838, 23.7275),     # Panathinaikos — Athens
    "PAR": (44.8176, 20.4633),     # Partizan — Belgrade
    "PRS": (48.8566, 2.3522),      # Paris Basketball — Paris
    "RED": (44.8176, 20.4633),     # Crvena Zvezda — Belgrade
    "TEL": (32.1093, 34.8007),     # Maccabi Tel Aviv — Tel Aviv
    "ULK": (40.9886, 29.0380),     # Fenerbahce — Istanbul (Ülker Arena)
    "VIR": (44.4949, 11.3426),     # Virtus Bologna — Bologna
    "ZAL": (54.8985, 23.9036),     # Zalgiris — Kaunas
}

# ---------------------------------------------------------------------------
# Neutral-site venue overrides  (season -> team_code -> coordinates)
#
# When a team hosts "home" games at an alternative venue for part or all
# of a season, record the venue coordinates here.  The travel helpers use
# this instead of TEAM_COORDS when the flag is active.
# ---------------------------------------------------------------------------

NEUTRAL_VENUE_OVERRIDES: Dict[int, Dict[str, Tuple[float, float]]] = {
    # 2024-25 season
    2024: {
        "TEL": (44.8176, 20.4633),     # Maccabi Tel Aviv → Belgrade
        "HTA": (42.6977, 23.3219),     # Hapoel Tel Aviv → Sofia
    },
    # 2025-26 season
    2025: {
        "TEL": (44.8176, 20.4633),     # Maccabi Tel Aviv → Belgrade
        "HTA": (42.6977, 23.3219),     # Hapoel Tel Aviv → Sofia
        "DUB": (43.8563, 18.4131),     # Dubai → Sarajevo (Zetra Arena)
    },
}


def get_home_venue(team_code: str, season: int) -> Tuple[float, float]:
    """Return the effective home-venue coordinates for *team_code* in *season*.

    Uses neutral-site overrides when present, otherwise the static home base.
    """
    overrides = NEUTRAL_VENUE_OVERRIDES.get(season, {})
    if team_code in overrides:
        return overrides[team_code]
    if team_code not in TEAM_COORDS:
        raise KeyError(
            f"Unknown team code {team_code!r}. "
            "Add it to TEAM_COORDS in features/context.py."
        )
    return TEAM_COORDS[team_code]


# ---------------------------------------------------------------------------
# Haversine
# ---------------------------------------------------------------------------

_EARTH_RADIUS_KM = 6_371.0


def haversine_km(
    coord1: Tuple[float, float],
    coord2: Tuple[float, float],
) -> float:
    """Great-circle distance in kilometres between two (lat, lon) points."""
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Game-venue resolution for a single game
# ---------------------------------------------------------------------------

def game_venue_coords(
    home_team: str,
    season: int,
    *,
    is_neutral: Optional[bool] = None,
) -> Tuple[float, float]:
    """Return the coordinates of the venue where a game is played.

    For now this always resolves to the home team's effective venue
    (which already accounts for neutral-site overrides).  The optional
    *is_neutral* flag is accepted for forward compatibility but does not
    change behaviour while the neutral venues are season-level overrides.
    """
    return get_home_venue(home_team, season)


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
# Previous-venue travel computation
# ---------------------------------------------------------------------------

TRAVEL_GAP_THRESHOLD_DAYS = 4


def compute_team_travel(
    team_game_df: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    """Add ``venue_lat``, ``venue_lon``, ``recent_travel_km`` per team-game row.

    For each team-game row the venue is resolved as the home-team's effective
    venue for that game.

    ``recent_travel_km`` uses the previous-location rule:

    * If the gap to the previous EuroLeague game is <= ``TRAVEL_GAP_THRESHOLD_DAYS``,
      travel = haversine(previous_game_venue, current_game_venue).
    * Otherwise travel = haversine(team_home_base, current_game_venue).
    * For home games with a long gap, this yields 0 (home base → home venue).
    """
    df = team_game_df.sort_values(["Team", "game_date", "Round", "Gamecode"]).copy()

    # --- resolve venue coordinates for each game row ---
    # The venue is where the game is played = the home team's effective venue.
    # For a home row, the team IS the home team; for an away row, the opponent is.
    home_teams = np.where(df["IsHome"] == 1, df["Team"], df["Opponent"])
    venue_coords = [
        game_venue_coords(str(ht), season) for ht in home_teams
    ]
    df["venue_lat"] = [c[0] for c in venue_coords]
    df["venue_lon"] = [c[1] for c in venue_coords]

    # --- previous venue within the same team ---
    df["_prev_venue_lat"] = df.groupby("Team")["venue_lat"].shift(1)
    df["_prev_venue_lon"] = df.groupby("Team")["venue_lon"].shift(1)
    df["_prev_date"] = df.groupby("Team")["game_date"].shift(1)

    gap_days = (df["game_date"] - df["_prev_date"]).dt.days
    has_prev = df["_prev_venue_lat"].notna()
    short_gap = has_prev & (gap_days <= TRAVEL_GAP_THRESHOLD_DAYS)

    # home base fallback
    home_bases = [TEAM_COORDS.get(str(t), (np.nan, np.nan)) for t in df["Team"]]
    hb_lat = np.array([c[0] for c in home_bases])
    hb_lon = np.array([c[1] for c in home_bases])

    origin_lat = np.where(short_gap, df["_prev_venue_lat"].values, hb_lat)
    origin_lon = np.where(short_gap, df["_prev_venue_lon"].values, hb_lon)

    travel_km = np.array([
        haversine_km((olat, olon), (vlat, vlon))
        if np.isfinite(olat) and np.isfinite(vlat) else 0.0
        for olat, olon, vlat, vlon in zip(
            origin_lat, origin_lon,
            df["venue_lat"].values, df["venue_lon"].values,
        )
    ])
    df["recent_travel_km"] = travel_km

    df.drop(columns=["_prev_venue_lat", "_prev_venue_lon", "_prev_date"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Game-level context features (rest diff + travel for both teams)
# ---------------------------------------------------------------------------

def build_game_context(
    games_df: pd.DataFrame,
    team_game_df: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    """Return a per-game DataFrame with context features ready for ML.

    Columns produced:

    * ``el_rest_days_diff``   – capped home rest minus capped away rest
    * ``home_recent_travel_km`` – home team double-week travel
    * ``away_recent_travel_km`` – away team travel to the current venue
    """
    tg = compute_team_el_rest(team_game_df)
    tg = compute_team_travel(tg, season)

    # index by (Team, Gamecode, IsHome) for fast lookups
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
            "home_recent_travel_km": float(h_row["recent_travel_km"]),
            "away_recent_travel_km": float(a_row["recent_travel_km"]),
        })

    return pd.DataFrame(rows)


def _empty_context(gamecode: int) -> dict:
    return {
        "Gamecode": gamecode,
        "el_rest_days_diff": 0.0,
        "home_recent_travel_km": 0.0,
        "away_recent_travel_km": 0.0,
    }


# ---------------------------------------------------------------------------
# Prediction-time context (upcoming games)
# ---------------------------------------------------------------------------

def build_prediction_context(
    schedule_df: pd.DataFrame,
    team_game_df: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    """Compute rest & travel context for upcoming (unplayed) games.

    Uses the played ``team_game_df`` to find each team's most recent game,
    then applies the same gap/travel logic used for historical games.
    """
    tg = team_game_df.copy()
    if "game_date" not in tg.columns or tg["game_date"].isna().all():
        return _fallback_prediction_context(schedule_df)

    tg = compute_team_travel(tg, season)

    # per-team latest state
    tg_sorted = tg.sort_values(["Team", "game_date", "Round", "Gamecode"])
    latest = tg_sorted.groupby("Team").last()

    rows: List[dict] = []
    for _, game in schedule_df.iterrows():
        gc = int(game["Gamecode"])
        ht = str(game["home_team"])
        at = str(game["away_team"])
        gd = game.get("game_date", pd.NaT)

        h_venue = game_venue_coords(ht, season)
        game_venue = h_venue

        h_rest = _pred_rest(latest, ht, gd)
        a_rest = _pred_rest(latest, at, gd)

        h_travel = _pred_travel(latest, ht, h_venue, gd, season)
        a_travel = _pred_travel(latest, at, game_venue, gd, season)

        rows.append({
            "Gamecode": gc,
            "el_rest_days_diff": h_rest - a_rest,
            "home_recent_travel_km": h_travel,
            "away_recent_travel_km": a_travel,
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


def _pred_travel(
    latest: pd.DataFrame,
    team: str,
    destination: Tuple[float, float],
    game_date,
    season: int,
) -> float:
    hb = TEAM_COORDS.get(team)
    if hb is None:
        return 0.0

    if team not in latest.index or pd.isna(game_date):
        return haversine_km(hb, destination)

    row = latest.loc[team]
    prev_date = row.get("game_date", pd.NaT)
    if pd.isna(prev_date):
        return haversine_km(hb, destination)

    gap = (pd.Timestamp(game_date) - pd.Timestamp(prev_date)).days
    if gap <= TRAVEL_GAP_THRESHOLD_DAYS:
        prev_venue = (float(row["venue_lat"]), float(row["venue_lon"]))
        return haversine_km(prev_venue, destination)

    return haversine_km(hb, destination)


def _fallback_prediction_context(schedule_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Gamecode": schedule_df["Gamecode"].values,
        "el_rest_days_diff": 0.0,
        "home_recent_travel_km": 0.0,
        "away_recent_travel_km": 0.0,
    })
