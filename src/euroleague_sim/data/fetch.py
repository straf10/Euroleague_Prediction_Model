from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List
import time
import requests
import pandas as pd

from euroleague_api.game_stats import GameStats
from euroleague_api.boxscore_data import BoxScoreData


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
