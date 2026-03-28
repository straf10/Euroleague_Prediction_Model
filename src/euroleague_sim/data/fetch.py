from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List
import time
import requests
import pandas as pd

from euroleague_api.game_stats import GameStats
from euroleague_api.boxscore_data import BoxScoreData


def _normalize_schedule_df(df: pd.DataFrame, round_number: int, season: int) -> pd.DataFrame:
    """Build standardized schedule DataFrame (Round, Gamecode, home_team, away_team, game_date).

    Tries multiple possible column names from v2/v3 API responses.
    """
    cols = _find_schedule_columns(df)
    if not cols:
        return pd.DataFrame()

    out = pd.DataFrame({
        "Round": int(round_number),
        "Gamecode": pd.to_numeric(df[cols["game"]], errors="coerce"),
        "home_team": df[cols["home"]].astype(str).str.strip(),
        "away_team": df[cols["away"]].astype(str).str.strip(),
    })
    out = out.dropna(subset=["Gamecode", "home_team", "away_team"])
    out["Gamecode"] = out["Gamecode"].astype(int)

    _date_parsed = _extract_schedule_date(df)
    if _date_parsed is not None and len(_date_parsed) == len(out):
        out["game_date"] = _date_parsed.values
    else:
        out["game_date"] = pd.NaT

    return out.sort_values("Gamecode").reset_index(drop=True)


def _extract_schedule_date(df: pd.DataFrame) -> pd.Series | None:
    """Try to parse a calendar date from the schedule DataFrame."""
    for col in ("date", "Date", "utcDate", "localDate", "gameDate"):
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce").dt.normalize()
            if parsed.notna().any():
                return parsed
    return None


def _find_schedule_columns(df: pd.DataFrame) -> dict | None:
    """Find column names for game, home team code, away team code. Returns None if not found."""
    col_lower = {c.lower(): c for c in df.columns}
    candidates_game = ["gamecode", "game_code", "gamenumber", "code"]
    candidates_home = [
        # v2: local.club.code (actual Euroleague v2 response)
        "local.club.code", "localclub.code", "localclubcode",
        "localteamcode", "local.code", "local.team.code",
        # v3 / generic
        "hometeam.code", "hometeamcode", "homecode", "hometeam.teamcode",
        "homeclubcode", "homeclub.code", "home.code", "home.teamcode",
    ]
    candidates_away = [
        # v2: road.club.code (actual Euroleague v2 response)
        "road.club.code", "roadclub.code", "roadclubcode",
        "roadteamcode", "road.code", "road.team.code",
        # v3 / generic
        "awayteam.code", "awayteamcode", "awaycode", "awayteam.teamcode",
        "awayclubcode", "awayclub.code", "away.code", "away.teamcode",
    ]
    game_col = None
    for c in candidates_game:
        if c in col_lower:
            game_col = col_lower[c]
            break
    if game_col is None:
        for c in df.columns:
            if "game" in c.lower() and ("code" in c.lower() or "number" in c.lower()):
                game_col = c
                break
    home_col = None
    for c in candidates_home:
        if c in col_lower:
            home_col = col_lower[c]
            break
    if home_col is None:
        for c in df.columns:
            cl = c.lower()
            if ("home" in cl or "local" in cl) and ("code" in cl) and ("club" in cl or "team" in cl):
                home_col = c
                break
    away_col = None
    for c in candidates_away:
        if c in col_lower:
            away_col = col_lower[c]
            break
    if away_col is None:
        for c in df.columns:
            cl = c.lower()
            if ("away" in cl or "road" in cl) and ("code" in cl) and ("club" in cl or "team" in cl):
                away_col = c
                break
    if game_col and home_col and away_col:
        return {"game": game_col, "home": home_col, "away": away_col}
    return None


@dataclass(frozen=True)
class FetchParams:
    competition: str = "E"
    # polite throttling (seconds) between heavy requests
    sleep_s: float = 0.2


class EuroleagueFetcher:
    """Data access layer.

    Uses:
    - `euroleague-api` (giasemidis) wrapper for:
      * v1 results (gamecodes/score/played)
      * boxscore totals
      * v3 game endpoints when needed (report/stats/teamsComparison)
    - direct v3 HTTP calls for `/v3/clubs` (team metadata mapping).
    """

    V3_BASE = "https://api-live.euroleague.net/v3"

    def __init__(self, params: FetchParams = FetchParams(), session: Optional[requests.Session] = None):
        self.params = params
        self.session = session or requests.Session()

        # wrapper instances
        self.gamestats = GameStats(params.competition)
        self.boxscore = BoxScoreData(params.competition)

    # -------------------------
    # Wrapper-based fetchers
    # -------------------------
    def gamecodes_season(self, season_start_year: int) -> pd.DataFrame:
        # v1 results endpoint (XML) via wrapper
        df = self.gamestats.get_gamecodes_season(season_start_year)
        return df

    def gamecodes_round(self, season_start_year: int, round_number: int) -> pd.DataFrame:
        # v2 seasons/{season}/games?roundNumber=... via wrapper
        df = self.gamestats.get_gamecodes_round(season_start_year, round_number)
        return df

    def schedule_round_v3(
        self, season_start_year: int, round_number: int
    ) -> pd.DataFrame:
        """Fetch round schedule from v3 API (fallback when v2 column names differ).

        Note: v3 list-games endpoint may return 405 Method Not Allowed; we catch and return empty.
        """
        season_code = f"{self.params.competition}{season_start_year}"
        url = (
            f"{self.V3_BASE}/competitions/{self.params.competition}"
            f"/seasons/{season_code}/games"
        )
        params = {"roundNumber": round_number}
        try:
            r = self.session.get(url, params=params, timeout=30)
            r.raise_for_status()
        except requests.exceptions.HTTPError:
            # v3 often returns 405 for GET list; rely on v2 only
            return pd.DataFrame()
        data = r.json()
        rows = data.get("games") or data.get("data") or (data if isinstance(data, list) else [])
        if not rows:
            return pd.DataFrame()

        df = pd.json_normalize(rows)
        return _normalize_schedule_df(df, round_number, season_start_year)

    def player_boxscore_stats_season(self, season_start_year: int) -> pd.DataFrame:
        # live.euroleague.net Boxscore endpoint via wrapper
        df = self.boxscore.get_player_boxscore_stats_single_season(season_start_year)
        return df

    # -------------------------
    # Direct v3 fetchers
    # -------------------------
    def clubs_v3(self, limit: int = 400) -> pd.DataFrame:
        """Fetches clubs from v3.

        Swagger sometimes paginates; we handle common response shapes:
        - {"total":..., "clubs": [...]}
        - {"data": [...], ...}
        - plain list
        """
        url = f"{self.V3_BASE}/clubs"
        params = {"limit": limit, "offset": 0}
        all_rows: List[Dict[str, Any]] = []
        while True:
            r = self.session.get(url, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list):
                rows = data
                total = len(rows)
            elif isinstance(data, dict):
                rows = data.get("clubs") or data.get("data") or data.get("items") or []
                total = data.get("total") or data.get("count") or len(rows)
            else:
                rows = []
                total = 0

            if not rows:
                break

            all_rows.extend(rows)

            # stop if no pagination signals
            if isinstance(data, list):
                break

            offset = params.get("offset", 0)
            if offset + len(rows) >= total:
                break

            params["offset"] = offset + len(rows)
            time.sleep(self.params.sleep_s)

        return pd.json_normalize(all_rows)

    # -------------------------
    # Convenience
    # -------------------------
    def fetch_all_for_season(self, season_start_year: int) -> Dict[str, pd.DataFrame]:
        gamecodes = self.gamecodes_season(season_start_year)
        time.sleep(self.params.sleep_s)
        boxscore = self.player_boxscore_stats_season(season_start_year)
        time.sleep(self.params.sleep_s)
        clubs = self.clubs_v3()
        return {
            "gamecodes": gamecodes,
            "boxscore": boxscore,
            "clubs": clubs
        }
