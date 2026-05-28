"""Generate ``teams_registry_E{season}.json`` and ``season_meta_E{season}.json``.

Reads the cached gamecodes + clubs DataFrames already produced by the
nightly pipeline, so this script never hits the EuroLeague API. Run after
``euroleague-sim update-data`` (or as a final step in the nightly Action).

Usage:
    python scripts/build_team_registry.py            # current season from config
    python scripts/build_team_registry.py --season 2024
"""
from __future__ import annotations

import argparse
from pathlib import Path

from euroleague_sim.config import ProjectConfig
from euroleague_sim.data.cache import Cache
from euroleague_sim.data.team_registry import write_registry_and_meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    args = ap.parse_args()

    cfg = ProjectConfig.default()
    season = args.season or cfg.season.season_start_year

    cache = Cache(args.cache_dir)
    gc_key = f"raw_gamecodes_E{season}"
    if not cache.has_df(gc_key):
        raise SystemExit(
            f"Missing cache: {gc_key}.pkl — run `euroleague-sim update-data "
            f"--season {season}` first."
        )

    gamecodes = cache.load_df(gc_key)
    clubs = cache.load_df("raw_clubs_v3") if cache.has_df("raw_clubs_v3") else None

    paths = write_registry_and_meta(args.cache_dir, season, gamecodes, clubs)
    for label, p in paths.items():
        print(f"  wrote {label}: {p}")


if __name__ == "__main__":
    main()
