"""Player-value proxies derived from EuroLeague player-level boxscores.

Two families of metrics are produced:

1. **Game Score (GmSc)** – John Hollinger's single-number box-score summary of a
   player's productivity in one game.
2. **Simplified Box Plus-Minus (BPM)** – a transparent, box-score-only estimate of
   a player's per-100-possession value relative to a league-average player. Real
   BPM (Basketball-Reference) is an NBA-calibrated regression; here we use a
   documented heuristic + a team adjustment anchored to the team's net rating so
   that the minute-weighted player BPMs reconcile with team performance.

The headline integration artefact is the **aggregate BPM of the *available*
roster** for an upcoming game. Because it is built only from players who actually
took the floor, it doubles as a dynamic injury / rotation proxy: when a high-value
player sits out, the team's available BPM drops automatically.

See ``data_engineering_log.md`` for the full derivation of every formula.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Boxscore column mapping (euroleague-api BoxScoreData shape)
# ---------------------------------------------------------------------------
# These are the raw column names returned by
# ``BoxScoreData.get_player_boxscore_stats_single_season``.
COL_MINUTES = "Minutes"
COL_POINTS = "Points"
COL_FGM2 = "FieldGoalsMade2"
COL_FGA2 = "FieldGoalsAttempted2"
COL_FGM3 = "FieldGoalsMade3"
COL_FGA3 = "FieldGoalsAttempted3"
COL_FTM = "FreeThrowsMade"
COL_FTA = "FreeThrowsAttempted"
COL_ORB = "OffensiveRebounds"
COL_DRB = "DefensiveRebounds"
COL_TRB = "TotalRebounds"
COL_AST = "Assistances"
COL_STL = "Steals"
COL_TOV = "Turnovers"
COL_BLK = "BlocksFavour"
COL_PF = "FoulsCommited"

# Simplified raw-BPM coefficients (heuristic; documented in the engineering log).
# Scale intuition: points / 100 possessions above a league-average player.
_BPM_COEFFS = {
    "usg": 0.20,        # x USG%
    "ts": 0.25,         # x (TS% - league_TS) * 100
    "ast": 0.10,        # x AST%
    "trb": 0.07,        # x TRB%
    "stl36": 0.50,      # x steals per 36
    "blk36": 0.30,      # x blocks per 36
    "tov36": -0.45,     # x turnovers per 36
    "pf36": -0.15,      # x fouls per 36
}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_minutes(value: object) -> float:
    """Convert a EuroLeague ``Minutes`` cell to fractional minutes.

    Handles ``"MM:SS"`` strings, plain numbers, and the various non-play
    sentinels (``"DNP"``, empty, ``NaN``) which all map to ``0.0``.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    s = str(value).strip()
    if not s or s.upper() in {"DNP", "DNS", "NP", "-", "NAN"}:
        return 0.0
    if ":" in s:
        parts = s.split(":")
        try:
            mm = int(parts[0])
            ss = int(parts[1]) if len(parts) > 1 else 0
        except ValueError:
            return 0.0
        return mm + ss / 60.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a numeric float Series for *col*, or zeros if the column is absent."""
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)


def _drop_summary_rows(box_df: pd.DataFrame) -> pd.DataFrame:
    """Drop the per-team ``"Team"`` and ``"Total"`` summary rows.

    The boxscore feed appends two non-player rows per team-game; including them
    would double team totals and break every usage/rebound percentage.
    """
    df = box_df.copy()
    if "Player_ID" in df.columns:
        pid = df["Player_ID"].astype(str).str.strip().str.upper()
        df = df[~pid.isin({"TEAM", "TOTAL"})]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Game Score
# ---------------------------------------------------------------------------

def compute_game_score(box_df: pd.DataFrame) -> pd.Series:
    """Hollinger Game Score for each player-game row.

    ``GmSc = PTS + 0.4*FGM - 0.7*FGA - 0.4*(FTA-FTM) + 0.7*ORB + 0.3*DRB
             + STL + 0.7*AST + 0.7*BLK - 0.4*PF - TOV``
    """
    pts = _num(box_df, COL_POINTS)
    fgm = _num(box_df, COL_FGM2) + _num(box_df, COL_FGM3)
    fga = _num(box_df, COL_FGA2) + _num(box_df, COL_FGA3)
    ftm = _num(box_df, COL_FTM)
    fta = _num(box_df, COL_FTA)
    orb = _num(box_df, COL_ORB)
    drb = _num(box_df, COL_DRB)
    stl = _num(box_df, COL_STL)
    ast = _num(box_df, COL_AST)
    blk = _num(box_df, COL_BLK)
    pf = _num(box_df, COL_PF)
    tov = _num(box_df, COL_TOV)

    return (
        pts
        + 0.4 * fgm
        - 0.7 * fga
        - 0.4 * (fta - ftm)
        + 0.7 * orb
        + 0.3 * drb
        + stl
        + 0.7 * ast
        + 0.7 * blk
        - 0.4 * pf
        - tov
    )


# ---------------------------------------------------------------------------
# Per player-game advanced rates + raw BPM
# ---------------------------------------------------------------------------

def compute_player_game_metrics(box_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per player-game GmSc, advanced rates, and a centred raw BPM.

    Advanced rates (USG%, AST%, TRB%, TS%) require team and opponent totals,
    which are derived by aggregating the boxscore itself. Rows where a player
    logged no minutes get ``NaN`` rates and ``NaN`` raw BPM (they cannot have a
    rate) but are kept so downstream availability logic still sees them.

    Returns a copy of the relevant identity columns plus:
    ``minutes, game_score, ts_pct, usg_pct, ast_pct, trb_pct, raw_bpm``.
    """
    df = _drop_summary_rows(box_df)
    df["minutes"] = df[COL_MINUTES].map(parse_minutes) if COL_MINUTES in df.columns else 0.0

    # Player box totals
    pts = _num(df, COL_POINTS)
    fgm = _num(df, COL_FGM2) + _num(df, COL_FGM3)
    fga = _num(df, COL_FGA2) + _num(df, COL_FGA3)
    fta = _num(df, COL_FTA)
    trb = _num(df, COL_TRB)
    ast = _num(df, COL_AST)
    stl = _num(df, COL_STL)
    blk = _num(df, COL_BLK)
    tov = _num(df, COL_TOV)
    pf = _num(df, COL_PF)

    df["_fgm"] = fgm
    df["_fga"] = fga
    df["_fta"] = fta
    df["_tov"] = tov
    df["_trb"] = trb
    df["_pts"] = pts

    df["game_score"] = compute_game_score(df)

    # ---- Team totals per (Season, Gamecode, Team) ----
    keys = ["Season", "Gamecode", "Team"]
    team_tot = df.groupby(keys, as_index=False).agg(
        tm_min=("minutes", "sum"),
        tm_fgm=("_fgm", "sum"),
        tm_fga=("_fga", "sum"),
        tm_fta=("_fta", "sum"),
        tm_tov=("_tov", "sum"),
        tm_trb=("_trb", "sum"),
    )

    # ---- Opponent totals: the *other* team in the same (Season, Gamecode) ----
    opp = team_tot.rename(columns={
        "Team": "_opp_team",
        "tm_trb": "opp_trb",
    })[["Season", "Gamecode", "_opp_team", "opp_trb"]]
    # cross-join within game then drop self-pairs
    merged_opp = team_tot.merge(opp, on=["Season", "Gamecode"])
    merged_opp = merged_opp[merged_opp["Team"] != merged_opp["_opp_team"]]
    opp_trb = merged_opp.groupby(keys, as_index=False).agg(opp_trb=("opp_trb", "sum"))
    team_tot = team_tot.merge(opp_trb, on=keys, how="left")

    df = df.merge(team_tot, on=keys, how="left")

    mp = df["minutes"]
    tm_min = df["tm_min"].replace(0, np.nan)
    tm_poss5 = tm_min / 5.0  # team minutes / 5 ≈ team "lineup games" unit

    played = mp > 0

    # True Shooting %
    ts_denom = 2.0 * (fga + 0.44 * fta)
    df["ts_pct"] = np.where(ts_denom > 0, pts / ts_denom, np.nan)

    # Usage %
    usg_num = (fga + 0.44 * fta + tov) * tm_poss5
    usg_den = mp * (df["tm_fga"] + 0.44 * df["tm_fta"] + df["tm_tov"])
    df["usg_pct"] = np.where((usg_den > 0) & played, 100.0 * usg_num / usg_den, np.nan)

    # Assist %
    ast_den = ((mp / tm_poss5) * df["tm_fgm"]) - fgm
    df["ast_pct"] = np.where((ast_den > 0) & played, 100.0 * ast / ast_den, np.nan)

    # Total rebound %
    trb_den = mp * (df["tm_trb"] + df["opp_trb"].fillna(df["tm_trb"]))
    df["trb_pct"] = np.where((trb_den > 0) & played, 100.0 * (trb * tm_poss5) / trb_den, np.nan)

    # League average TS (minute-weighted) — data-driven centring reference
    lg_ts = _minute_weighted_mean(df["ts_pct"], mp)
    if not np.isfinite(lg_ts):
        lg_ts = 0.56  # EuroLeague-ish fallback

    per36 = np.where(played, 36.0 / mp.replace(0, np.nan), 0.0)
    raw = (
        _BPM_COEFFS["usg"] * df["usg_pct"].fillna(0.0)
        + _BPM_COEFFS["ts"] * (df["ts_pct"].fillna(lg_ts) - lg_ts) * 100.0
        + _BPM_COEFFS["ast"] * df["ast_pct"].fillna(0.0)
        + _BPM_COEFFS["trb"] * df["trb_pct"].fillna(0.0)
        + _BPM_COEFFS["stl36"] * (stl * per36)
        + _BPM_COEFFS["blk36"] * (blk * per36)
        + _BPM_COEFFS["tov36"] * (tov * per36)
        + _BPM_COEFFS["pf36"] * (pf * per36)
    )
    # Clip single-game extremes (small-sample artefacts) before they feed the
    # expanding mean; ±15 brackets realistic per-game value.
    raw = raw.clip(lower=-15.0, upper=15.0)
    raw = pd.Series(np.where(played, raw, np.nan), index=df.index)

    # Centre so the minute-weighted league-average raw BPM is exactly 0.
    centre = _minute_weighted_mean(raw, mp)
    if np.isfinite(centre):
        raw = raw - centre
    df["raw_bpm"] = raw

    out_cols = [
        c for c in ["Season", "Phase", "Round", "Gamecode", "Team", "Player_ID",
                    "Player", "IsPlaying"]
        if c in df.columns
    ]
    out_cols += ["minutes", "game_score", "ts_pct", "usg_pct", "ast_pct",
                 "trb_pct", "raw_bpm"]
    return df[out_cols].copy()


def _minute_weighted_mean(values: pd.Series, minutes: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    m = pd.to_numeric(minutes, errors="coerce")
    mask = np.isfinite(v) & np.isfinite(m) & (m > 0)
    if not mask.any():
        return np.nan
    return float(np.average(v[mask], weights=m[mask]))


# ---------------------------------------------------------------------------
# Team-level aggregation (point-in-time timeline for training)
# ---------------------------------------------------------------------------

def build_team_bpm_timeline(
    box_df: pd.DataFrame,
    team_game_df: pd.DataFrame,
) -> pd.DataFrame:
    """Per team-game *available-roster* BPM with no look-ahead.

    For team T's game G:

    * each player's skill = expanding mean of their **prior** game raw BPM
      (``shift(1)``), i.e. season-to-date form before tip-off;
    * the unit value = minutes-weighted (this game's minutes) mean of those
      skills over the players who actually played → the available-roster
      component / injury proxy;
    * a team anchor = (expanding team ``NetPer100`` before G) / 5 is added so the
      number reflects overall team strength as well as who is on the floor.

    Returns ``DataFrame[Season, Gamecode, Round, Team, team_bpm_available]``.
    """
    pm = compute_player_game_metrics(box_df)
    if pm.empty:
        return pd.DataFrame(columns=["Season", "Gamecode", "Round", "Team", "team_bpm_available"])

    pm = pm.sort_values(["Player_ID", "Round", "Gamecode"]).copy()
    pm["skill"] = (
        pm.groupby("Player_ID")["raw_bpm"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    pm["skill"] = pm["skill"].fillna(0.0)  # first appearance → league average

    pm["_w"] = pm["minutes"].clip(lower=0.0)
    pm["_ws"] = pm["_w"] * pm["skill"]

    grp = pm.groupby(["Season", "Gamecode", "Round", "Team"], as_index=False).agg(
        _wsum=("_w", "sum"),
        _wssum=("_ws", "sum"),
    )
    grp["unit_bpm"] = np.where(grp["_wsum"] > 0, grp["_wssum"] / grp["_wsum"], 0.0)

    # Team anchor: expanding NetPer100 before this game, per team.
    tg = team_game_df.sort_values(["Team", "Round", "Gamecode"]).copy()
    if "NetPer100" in tg.columns:
        tg["net_to_date"] = (
            tg.groupby("Team")["NetPer100"]
            .transform(lambda x: x.shift(1).expanding().mean())
            .fillna(0.0)
        )
    else:
        tg["net_to_date"] = 0.0
    anchor = tg[["Season", "Gamecode", "Team", "net_to_date"]].drop_duplicates(
        ["Season", "Gamecode", "Team"]
    )

    out = grp.merge(anchor, on=["Season", "Gamecode", "Team"], how="left")
    out["net_to_date"] = out["net_to_date"].fillna(0.0)
    out["team_bpm_available"] = out["unit_bpm"] + out["net_to_date"] / 5.0
    return out[["Season", "Gamecode", "Round", "Team", "team_bpm_available"]]


def team_bpm_game_diff(
    games_df: pd.DataFrame,
    bpm_timeline: pd.DataFrame,
) -> pd.DataFrame:
    """Collapse the team-game timeline into a per-game ``net_bpm_diff``.

    ``net_bpm_diff = home_team_bpm_available - away_team_bpm_available``.
    Returns ``DataFrame[Season, Gamecode, net_bpm_diff]``.
    """
    if bpm_timeline.empty or games_df.empty:
        return pd.DataFrame(columns=["Season", "Gamecode", "net_bpm_diff"])

    idx = bpm_timeline.set_index(["Season", "Gamecode", "Team"])["team_bpm_available"]
    rows = []
    for _, g in games_df.iterrows():
        season = int(g["Season"])
        gc = int(g["Gamecode"])
        ht = str(g["home_team"])
        at = str(g["away_team"])
        h = idx.get((season, gc, ht), np.nan)
        a = idx.get((season, gc, at), np.nan)
        if np.isfinite(h) and np.isfinite(a):
            rows.append({"Season": season, "Gamecode": gc, "net_bpm_diff": float(h - a)})
        else:
            rows.append({"Season": season, "Gamecode": gc, "net_bpm_diff": 0.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Prediction-time aggregation (current available roster)
# ---------------------------------------------------------------------------

def compute_current_team_bpm(
    box_df: pd.DataFrame,
    team_game_df: pd.DataFrame,
) -> Dict[str, float]:
    """Current available-roster BPM per team, for upcoming games.

    The "available roster" is taken from each team's most recent played game
    (the players with positive minutes there). Each player is valued at their
    full season raw BPM, minutes-weighted by that latest game, plus the team's
    season ``NetPer100`` / 5 anchor. If a regular sat out the last game (injury,
    rest) they are excluded, so the value reflects who is actually playing.

    Returns ``{team_code: team_bpm_available}``.
    """
    pm = compute_player_game_metrics(box_df)
    if pm.empty:
        return {}

    player_skill = pm.groupby("Player_ID")["raw_bpm"].mean()

    # Season net rating anchor per team
    if "NetPer100" in team_game_df.columns and not team_game_df.empty:
        team_net = team_game_df.groupby("Team")["NetPer100"].mean().to_dict()
    else:
        team_net = {}

    result: Dict[str, float] = {}
    for team, grp in pm.groupby("Team"):
        # latest played game for this team
        played = grp[grp["minutes"] > 0]
        if played.empty:
            continue
        last_key = played.sort_values(["Round", "Gamecode"]).iloc[-1][["Round", "Gamecode"]]
        roster = played[
            (played["Round"] == last_key["Round"])
            & (played["Gamecode"] == last_key["Gamecode"])
        ]
        w = roster["minutes"].to_numpy(dtype=float)
        s = roster["Player_ID"].map(player_skill).fillna(0.0).to_numpy(dtype=float)
        unit = float(np.average(s, weights=w)) if w.sum() > 0 else 0.0
        anchor = float(team_net.get(str(team), 0.0)) / 5.0
        result[str(team)] = unit + anchor
    return result


def current_team_bpm_to_game_diff(
    schedule_df: pd.DataFrame,
    current_bpm: Dict[str, float],
) -> Dict[int, float]:
    """Map current team BPM onto a schedule → ``{gamecode: net_bpm_diff}``."""
    out: Dict[int, float] = {}
    for _, g in schedule_df.iterrows():
        gc = int(g["Gamecode"])
        h = current_bpm.get(str(g["home_team"]))
        a = current_bpm.get(str(g["away_team"]))
        if h is not None and a is not None:
            out[gc] = float(h - a)
        else:
            out[gc] = 0.0
    return out
