"""Per-team season roster cache for the What-If injury simulator.

For every team that has played at least one game in the current season we
persist:

* ``net_anchor`` – team's mean ``NetPer100 / 5`` (matches the anchor term
  used by ``compute_current_team_bpm`` in production).
* ``players[]`` – season-aggregated rows of ``player_id, name, season_bpm,
  avg_minutes, last_game_minutes, games_played``.

The What-If page reconstructs ``net_bpm_diff`` with the *same* formula the
prediction pipeline uses — minutes-weighted mean of season raw_bpm plus the
team anchor — so unchecking all players reproduces the baseline shown on
Daily Predictions, and unchecking one player moves the number exactly as
much as that player's playing-time × skill contribution.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..features.player_metrics import compute_player_game_metrics


def rosters_path(cache_dir: Path, season: int) -> Path:
    return cache_dir / f"rosters_E{season}.json"


def build_rosters(
    box_df: pd.DataFrame,
    team_game_df: pd.DataFrame,
    season: int,
) -> Dict[str, Any]:
    """Aggregate the boxscore feed into one roster row per (team, player)."""
    pm = compute_player_game_metrics(box_df)
    if pm.empty:
        return {"season": int(season), "teams": {}}

    # Season skill = mean raw_bpm across games the player took the floor.
    # Matches the per-player ``player_skill`` used by ``compute_current_team_bpm``.
    pm_played = pm[pm["minutes"] > 0].copy()

    # Season net rating anchor per team (NetPer100 / 5).
    if "NetPer100" in team_game_df.columns and not team_game_df.empty:
        team_net = team_game_df.groupby("Team")["NetPer100"].mean().to_dict()
    else:
        team_net = {}

    teams: Dict[str, Any] = {}
    for team, grp in pm_played.groupby("Team"):
        # Identify each team's most recent played game so we know who suited
        # up and how many minutes they actually got. That's the "default"
        # active roster for the simulator.
        latest = grp.sort_values(["Round", "Gamecode"]).iloc[-1]
        last_round = int(latest["Round"])
        last_gc = int(latest["Gamecode"])
        last_roster = grp[
            (grp["Round"] == last_round) & (grp["Gamecode"] == last_gc)
        ].set_index("Player_ID")["minutes"].to_dict()

        # Season-level per-player aggregates.
        agg = grp.groupby("Player_ID").agg(
            name=("Player", lambda s: next((str(v).strip() for v in s if pd.notna(v) and str(v).strip()), "")),
            season_bpm=("raw_bpm", "mean"),
            avg_minutes=("minutes", "mean"),
            games_played=("minutes", "count"),
        ).reset_index()

        players: List[Dict[str, Any]] = []
        for _, r in agg.iterrows():
            pid = str(r["Player_ID"]).strip()
            players.append({
                "player_id": pid,
                "name": str(r["name"]) or pid,
                "season_bpm": float(r["season_bpm"]) if pd.notna(r["season_bpm"]) else 0.0,
                "avg_minutes": float(r["avg_minutes"]),
                "last_game_minutes": float(last_roster.get(pid, 0.0)),
                "games_played": int(r["games_played"]),
            })

        # Sort by last-game minutes desc, then season average — most relevant
        # players surface at the top of the UI.
        players.sort(
            key=lambda p: (p["last_game_minutes"], p["avg_minutes"]),
            reverse=True,
        )

        teams[str(team)] = {
            "net_anchor": float(team_net.get(str(team), 0.0)) / 5.0,
            "last_game": {"round": last_round, "gamecode": last_gc},
            "players": players,
        }

    return {"season": int(season), "teams": teams}


def write_rosters(
    cache_dir: Path,
    season: int,
    box_df: pd.DataFrame,
    team_game_df: pd.DataFrame,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rosters = build_rosters(box_df, team_game_df, season)
    p = rosters_path(cache_dir, season)
    p.write_text(json.dumps(rosters, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def load_rosters(cache_dir: Path, season: int) -> Optional[Dict[str, Any]]:
    p = rosters_path(cache_dir, season)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Math used by the Streamlit page (kept here so the formula has one home).
# ---------------------------------------------------------------------------

def active_team_bpm(team_entry: Dict[str, Any], active_player_ids: set[str]) -> float:
    """Minutes-weighted mean season BPM for the active set, plus team anchor.

    Mirrors ``features.player_metrics.compute_current_team_bpm``: weights are
    the last-played-game minutes; if a player wasn't in that game but the
    user toggled them on, we fall back to their season-average minutes so
    the roster move still has a meaningful weight.
    """
    weights: List[float] = []
    values: List[float] = []
    for p in team_entry["players"]:
        if p["player_id"] not in active_player_ids:
            continue
        w = p["last_game_minutes"] if p["last_game_minutes"] > 0 else p["avg_minutes"]
        if w <= 0:
            continue
        weights.append(w)
        values.append(p["season_bpm"])

    if not weights:
        return float(team_entry.get("net_anchor", 0.0))

    weighted = sum(v * w for v, w in zip(values, weights)) / sum(weights)
    return float(weighted + team_entry.get("net_anchor", 0.0))


def default_active_ids(team_entry: Dict[str, Any]) -> set[str]:
    """Players who suited up in the most recent played game (the baseline)."""
    return {p["player_id"] for p in team_entry["players"] if p["last_game_minutes"] > 0}
