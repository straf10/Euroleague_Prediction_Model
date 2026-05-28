"""Run the historical domestic scrape and save to data_cache/domestic_matches.pkl.

Usage:
    python scripts/scrape_domestic_history.py
    python scripts/scrape_domestic_history.py --seasons 2024,2025

The default scrapes seasons 2023, 2024, 2025 (i.e. 2023-24 → 2025-26) across
all 21 teams currently in TEAM_DOMESTIC_MAP. Throttled with jitter; expect
~10–25 minutes wall time depending on network.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from euroleague_sim.data.cache import Cache  # noqa: E402
from euroleague_sim.data.domestic_scraper import (  # noqa: E402
    ScrapeConfig,
    TEAM_DOMESTIC_MAP,
    run_domestic_scrape,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--seasons",
        default="2023,2024,2025",
        help="Comma-separated season START years (e.g. 2023,2024,2025).",
    )
    ap.add_argument("--cache-dir", default=str(ROOT / "data_cache"))
    ap.add_argument("--cache-key", default="domestic_matches")
    args = ap.parse_args()

    seasons = tuple(int(s) for s in args.seasons.split(",") if s.strip())
    cfg = ScrapeConfig(seasons=seasons)
    print(f"Scraping {len(TEAM_DOMESTIC_MAP)} teams × {len(seasons)} seasons = "
          f"{len(TEAM_DOMESTIC_MAP) * len(seasons)} page loads…")
    df = run_domestic_scrape(cfg=cfg)
    print(f"\nScrape complete. Total rows: {len(df)}")
    if df.empty:
        print("WARNING: empty result; not writing cache.")
        return 1

    cache = Cache(Path(args.cache_dir))
    path = cache.save_df(args.cache_key, df)
    print(f"Saved cache -> {path}")

    print("\nPer-team row counts:")
    print(df.groupby("el_code").size().sort_values(ascending=False).to_string())
    print("\nPer-season row counts:")
    print(df.assign(season=df["match_date"].dt.year)
          .groupby("season").size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
