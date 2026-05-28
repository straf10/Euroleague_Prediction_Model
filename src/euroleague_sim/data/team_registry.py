"""Team registry + season metadata.

Two JSON artefacts are persisted under ``data_cache/`` and consumed by both
the feature pipeline and the Streamlit UI:

* ``teams_registry_E{season}.json`` — 3-letter code → display name / city /
  country / crest URL.  Built from the cached ``raw_clubs_v3.pkl`` filtered
  to the teams that actually appear in the season's gamecodes.
* ``season_meta_E{season}.json``    — round ranges per competition phase,
  derived from the cached ``raw_gamecodes_E{season}.pkl``.  Replaces the
  hardcoded ``max_rounds=34`` used by ``round_progress``.

Both files are safe to regenerate: the pipeline only reads them; missing
files fall back to sensible defaults so older snapshots keep working.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


# Curated short names for the 20 active Euroleague clubs.  The full
# ``raw_clubs_v3`` names are screaming-uppercase ("EA7 EMPORIO ARMANI
# MILAN") which reads poorly in tables; these are the human-friendly forms
# used across the dashboard.  Codes outside this map fall back to a
# title-cased version of the official name.
SHORT_NAME_OVERRIDES: Dict[str, str] = {
    "ASV": "ASVEL Villeurbanne",
    "BAR": "FC Barcelona",
    "BAS": "Baskonia",
    "DUB": "Dubai BC",
    "HTA": "Hapoel Tel Aviv",
    "IST": "Anadolu Efes",
    "MAD": "Real Madrid",
    "MCO": "AS Monaco",
    "MIL": "Olimpia Milano",
    "MUN": "Bayern Munich",
    "OLY": "Olympiacos",
    "PAM": "Valencia Basket",
    "PAN": "Panathinaikos",
    "PAR": "Partizan Belgrade",
    "PRS": "Paris Basketball",
    "RED": "Crvena Zvezda",
    "TEL": "Maccabi Tel Aviv",
    "ULK": "Fenerbahce",
    "VIR": "Virtus Bologna",
    "ZAL": "Zalgiris Kaunas",
}


# Map gamecodes "Phase" values to canonical phase keys used downstream.
_PHASE_KEY_MAP = {
    "RS": "regular_season",
    "PI": "playin",
    "PO": "playoff",
    "FF": "final_four",
}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def teams_registry_path(cache_dir: Path, season: int) -> Path:
    return cache_dir / f"teams_registry_E{season}.json"


def season_meta_path(cache_dir: Path, season: int) -> Path:
    return cache_dir / f"season_meta_E{season}.json"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _title_case(name: str) -> str:
    if not name:
        return ""
    parts = []
    for word in name.split():
        if word.isupper() and len(word) <= 3:
            parts.append(word)  # keep "FC", "AS" as-is
        else:
            parts.append(word.title())
    return " ".join(parts)


def build_teams_registry(
    gamecodes_df: pd.DataFrame,
    clubs_df: Optional[pd.DataFrame],
    season: int,
) -> Dict[str, Any]:
    """Return ``{season, teams: {CODE: {name, short_name, city, country, crest_url}}}``."""
    codes: List[str] = []
    long_names: Dict[str, str] = {}
    for col_code, col_name in [("homecode", "hometeam"), ("awaycode", "awayteam")]:
        if col_code in gamecodes_df.columns:
            for code, name in zip(gamecodes_df[col_code], gamecodes_df.get(col_name, [])):
                code = str(code).strip()
                if not code or code == "nan":
                    continue
                if code not in long_names and isinstance(name, str) and name.strip():
                    long_names[code] = name.strip()
                codes.append(code)

    unique_codes = sorted(set(codes))

    clubs_lookup: Dict[str, Dict[str, Any]] = {}
    if clubs_df is not None and not clubs_df.empty and "code" in clubs_df.columns:
        for _, row in clubs_df.iterrows():
            c = str(row.get("code", "")).strip()
            if not c:
                continue
            clubs_lookup[c] = row.to_dict()

    teams: Dict[str, Dict[str, Any]] = {}
    for code in unique_codes:
        club = clubs_lookup.get(code, {})
        full_name = club.get("name") or long_names.get(code) or code
        short = SHORT_NAME_OVERRIDES.get(code) or _title_case(full_name)
        teams[code] = {
            "name": full_name,
            "short_name": short,
            "city": club.get("city") or "",
            "country": club.get("country.name") or club.get("country") or "",
            "crest_url": club.get("images.crest") or "",
        }

    return {"season": int(season), "teams": teams}


def build_season_meta(gamecodes_df: pd.DataFrame, season: int) -> Dict[str, Any]:
    """Derive round-range metadata from the cached gamecodes table."""
    df = gamecodes_df.copy()
    df["Round"] = pd.to_numeric(df["Round"], errors="coerce")
    df = df.dropna(subset=["Round"])

    phases: Dict[str, List[int]] = {}
    if "Phase" in df.columns:
        for phase_code, sub in df.groupby("Phase"):
            key = _PHASE_KEY_MAP.get(str(phase_code).strip(), str(phase_code).strip().lower())
            rounds = sorted(int(r) for r in sub["Round"].unique())
            phases[key] = rounds

    all_rounds = sorted(int(r) for r in df["Round"].unique())
    rs_rounds = phases.get("regular_season", [])
    regular_season_end = max(rs_rounds) if rs_rounds else (max(all_rounds) if all_rounds else 34)
    n_rounds = max(all_rounds) if all_rounds else regular_season_end

    teams = set()
    for col in ("homecode", "awaycode"):
        if col in df.columns:
            teams.update(str(c).strip() for c in df[col].dropna() if str(c).strip())

    return {
        "season": int(season),
        "n_rounds": int(n_rounds),
        "n_teams": int(len(teams)),
        "regular_season_end_round": int(regular_season_end),
        "phases": {k: [int(min(v)), int(max(v))] for k, v in phases.items() if v},
    }


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def write_registry_and_meta(
    cache_dir: Path,
    season: int,
    gamecodes_df: pd.DataFrame,
    clubs_df: Optional[pd.DataFrame],
) -> Dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    reg = build_teams_registry(gamecodes_df, clubs_df, season)
    meta = build_season_meta(gamecodes_df, season)
    reg_p = teams_registry_path(cache_dir, season)
    meta_p = season_meta_path(cache_dir, season)
    reg_p.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")
    meta_p.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"registry": reg_p, "season_meta": meta_p}


def load_teams_registry(cache_dir: Path, season: int) -> Optional[Dict[str, Any]]:
    p = teams_registry_path(cache_dir, season)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def load_season_meta(cache_dir: Path, season: int) -> Optional[Dict[str, Any]]:
    p = season_meta_path(cache_dir, season)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Display helpers — used by Streamlit
# ---------------------------------------------------------------------------

def team_label(code: str, registry: Optional[Dict[str, Any]] = None, kind: str = "short") -> str:
    """Return a human-friendly name for ``code``.

    ``kind`` is ``"short"`` (default) for table-friendly labels, or ``"full"``
    for the official club name. Falls back to the code itself if no registry
    entry exists, so callers never need to guard.
    """
    if not code:
        return ""
    code = str(code).strip()
    if registry is None:
        return SHORT_NAME_OVERRIDES.get(code, code)
    team = registry.get("teams", {}).get(code)
    if not team:
        return SHORT_NAME_OVERRIDES.get(code, code)
    if kind == "full":
        return team.get("name") or team.get("short_name") or code
    return team.get("short_name") or team.get("name") or code


def max_rounds_from_meta(meta: Optional[Dict[str, Any]], default: int = 34) -> int:
    """Prefer ``regular_season_end_round`` (the right normaliser for
    ``round_progress``); fall back to ``n_rounds`` then ``default``."""
    if not meta:
        return default
    return int(
        meta.get("regular_season_end_round")
        or meta.get("n_rounds")
        or default
    )
