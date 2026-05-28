"""Zero-cost domestic-league fatigue scraper.

EuroLeague clubs also play in their national leagues (ACB, Lega A, HEBA/GBL, BSL,
LKL, ABA, Betclic ÉLITE, …). Those midweek/weekend domestic games add physical
load that the EuroLeague-only schedule cannot see. This module scrapes recent
domestic fixtures from a free aggregator and turns them into a
``rolling_7d_domestic_minutes`` load feature per team.

Design principles
-----------------
* **Polite, not evasive.** We throttle requests with jitter, cap concurrency,
  retry with exponential backoff, send a real ``User-Agent`` and cache results
  on disk. We do **not** rotate proxies, spoof fingerprints or solve captchas.
  Check each target site's ``robots.txt`` / Terms of Service before running, and
  keep the request rate low. This is intended for personal, low-volume research.
* **Crash-proof.** Every network/parse step is wrapped so a site redesign or a
  single failing team degrades to "no data for that team" rather than taking
  down the whole prediction pipeline. Missing data ⇒ neutral fatigue (0).
* **Dependency-light at import time.** ``playwright`` is imported lazily so the
  rest of the package works even when it is not installed.

A domestic game contributes a fixed physical-load budget (``MINUTES_PER_GAME``,
default 40 — one regulation game per team). Overtime games can be scored higher
by passing a richer parser, but the default is robust to missing score data.

See ``data_engineering_log.md`` for the scraping strategy and Playwright setup.
"""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd


# ---------------------------------------------------------------------------
# Constants & configuration
# ---------------------------------------------------------------------------

MINUTES_PER_GAME: float = 40.0  # regulation team minutes for one domestic game
DEFAULT_WINDOW_DAYS: int = 7

# A realistic desktop UA. Not an evasion tactic — just avoids being treated as a
# misconfigured client. Update occasionally to a current browser string.
DEFAULT_USER_AGENT: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class DomesticTeam:
    """Maps a EuroLeague team code to its domestic-league identity.

    ``url`` should point at the club's fixtures/results page on the chosen
    aggregator. Because the EuroLeague team *code* (e.g. ``MAD``) never matches a
    scraper's club slug, this table is the bridge — and it must be validated
    against ``data_cache/raw_clubs_v3.pkl`` whenever the EuroLeague roster
    changes.
    """

    el_code: str
    competition: str          # ACB, LegaA, HEBA, BSL, LKL, ABA, Betclic, …
    domestic_name: str
    url: str = ""             # team ROOT URL on the aggregator (no /schedule suffix);
                              # the parser appends /schedule/{season} per season.


@dataclass(frozen=True)
class ScrapeConfig:
    """Politeness / robustness knobs for the scraper."""

    provider: str = "proballers"
    user_agent: str = DEFAULT_USER_AGENT
    headless: bool = True
    nav_timeout_ms: int = 30_000          # explicit page-load / selector wait
    selector_timeout_ms: int = 15_000
    min_delay_s: float = 1.5              # polite throttle between requests …
    max_delay_s: float = 3.5              # … with jitter in [min, max]
    max_concurrency: int = 2              # cap simultaneous pages
    max_retries: int = 3
    backoff_base_s: float = 2.0
    # Proballers uses the start year of the season as the schedule path suffix:
    # /schedule/2024  -> 2024-25 season, /schedule/2023 -> 2023-24, etc.
    # Default = current + previous two seasons, matching the training window.
    seasons: tuple = (2023, 2024, 2025)
    lookback_days: Optional[int] = None   # if set, drop rows older than N days (current-only use)


# Curated map for the 2025-26 EuroLeague field. URLs point at the Proballers
# *team root* (no ``/schedule`` suffix); the parser appends ``/schedule/{year}``
# per season. IDs were verified against league/177/euroleague/teams on
# 2026-05-28 — re-validate whenever the EuroLeague field changes.
#
# Note: ``PAN`` (Panathinaikos) was NOT in the legacy 19-team map but plays in
# HEBA so we add it here for completeness. ``DUB`` / ``HAP`` joined in 2025-26.
# ``BER`` (ALBA Berlin) is no longer in EuroLeague but stays for historical
# 2023-24 / 2024-25 training data.
TEAM_DOMESTIC_MAP: Dict[str, DomesticTeam] = {
    # Spain – Liga Endesa (ACB)
    "MAD": DomesticTeam("MAD", "ACB", "Real Madrid", "https://www.proballers.com/basketball/team/160/real-madrid"),
    "BAR": DomesticTeam("BAR", "ACB", "FC Barcelona", "https://www.proballers.com/basketball/team/148/fc-barcelona"),
    "BAS": DomesticTeam("BAS", "ACB", "Baskonia", "https://www.proballers.com/basketball/team/155/baskonia-vitoria-gasteiz"),
    "PAM": DomesticTeam("PAM", "ACB", "Valencia Basket", "https://www.proballers.com/basketball/team/153/valencia-basket"),
    # Greece – HEBA/GBL
    "OLY": DomesticTeam("OLY", "HEBA", "Olympiacos", "https://www.proballers.com/basketball/team/188/olympiacos-piraeus"),
    "PAN": DomesticTeam("PAN", "HEBA", "Panathinaikos", "https://www.proballers.com/basketball/team/185/panathinaikos-athens"),
    # Turkey – BSL
    "IST": DomesticTeam("IST", "BSL", "Anadolu Efes", "https://www.proballers.com/basketball/team/543/anadolu-efes-istanbul"),
    "ULK": DomesticTeam("ULK", "BSL", "Fenerbahce Beko", "https://www.proballers.com/basketball/team/552/fenerbahce-istanbul"),
    # Israel – Ligat Winner
    "TEL": DomesticTeam("TEL", "Ligat-Winner", "Maccabi Tel Aviv", "https://www.proballers.com/basketball/team/609/maccabi-playtika-tel-aviv"),
    "HAP": DomesticTeam("HAP", "Ligat-Winner", "Hapoel Tel Aviv", "https://www.proballers.com/basketball/team/776/hapoel-tel-aviv"),
    # Italy – LegaA
    "MIL": DomesticTeam("MIL", "LegaA", "Olimpia Milano", "https://www.proballers.com/basketball/team/178/ea7-emporio-armani-milan"),
    "VIR": DomesticTeam("VIR", "LegaA", "Virtus Bologna", "https://www.proballers.com/basketball/team/165/virtus-segafredo-bologna"),
    # Germany – BBL
    "MUN": DomesticTeam("MUN", "BBL", "Bayern Munich", "https://www.proballers.com/basketball/team/2020/fc-bayern-munich"),
    "BER": DomesticTeam("BER", "BBL", "ALBA Berlin", "https://www.proballers.com/basketball/team/384/alba-berlin"),
    # France – Betclic ÉLITE
    "MCO": DomesticTeam("MCO", "Betclic", "AS Monaco", "https://www.proballers.com/basketball/team/283/as-monaco"),
    "ASV": DomesticTeam("ASV", "Betclic", "LDLC ASVEL", "https://www.proballers.com/basketball/team/1/ldlc-asvel"),
    "PRS": DomesticTeam("PRS", "Betclic", "Paris Basketball", "https://www.proballers.com/basketball/team/13219/paris"),
    # Adriatic – ABA League
    "PAR": DomesticTeam("PAR", "ABA", "Partizan", "https://www.proballers.com/basketball/team/565/partizan-belgrade"),
    "RED": DomesticTeam("RED", "ABA", "Crvena Zvezda", "https://www.proballers.com/basketball/team/561/crvena-zvezda-belgrade"),
    # Lithuania – LKL
    "ZAL": DomesticTeam("ZAL", "LKL", "Zalgiris Kaunas", "https://www.proballers.com/basketball/team/649/zalgiris-kaunas"),
    # UAE – Dubai (new 2025-26 EL entry, no historical domestic data)
    "DUB": DomesticTeam("DUB", "ABA", "Dubai", "https://www.proballers.com/basketball/team/15350/dubai"),
}


@dataclass
class DomesticMatch:
    """A single scraped domestic fixture."""

    el_code: str
    competition: str
    match_date: pd.Timestamp
    minutes: float = MINUTES_PER_GAME
    opponent: str = ""


# ---------------------------------------------------------------------------
# Lazy Playwright import
# ---------------------------------------------------------------------------

def _import_async_playwright():
    """Import ``playwright.async_api`` lazily with a helpful error message."""
    try:
        from playwright.async_api import async_playwright  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "Playwright is not installed. Install it with:\n"
            "    pip install playwright\n"
            "    python -m playwright install chromium\n"
            "See data_engineering_log.md for details."
        ) from exc
    return async_playwright


# ---------------------------------------------------------------------------
# Provider parsers
# ---------------------------------------------------------------------------
# A provider is just an async function (page, team, cfg) -> list[DomesticMatch].
# Selectors WILL drift as sites change; each parser must therefore be defensive
# and is always called from within a try/except in the orchestrator.

# Tab names we exclude — these are NOT domestic-league fatigue contributors.
# Anything else (Liga Endesa, BSL, LegaA, BBL, ABA, Betclic ÉLITE, LKL, HEBA,
# domestic cups, domestic playoffs, …) is counted toward domestic load.
_EXCLUDE_TABS = ("euroleague", "eurocup", "champions league", "nba", "fiba")


async def _parse_proballers(
    page,
    team: DomesticTeam,
    cfg: ScrapeConfig,
    season: int,
) -> List[DomesticMatch]:
    """Parser for one team × one season on Proballers.

    URL pattern: ``{team.url}/schedule/{season}`` where *season* is the START
    year (e.g. ``2024`` ⇒ 2024-25 season).

    Proballers renders the page as Bootstrap tabs: each competition the team
    played in has an ``<a class="list-group-item" data-toggle="list"
    href="#league-NNN">`` plus a matching ``<div id="league-NNN">`` pane with a
    ``<table>`` inside. Crucially:

    * The server only fully populates the **EuroLeague** tab's table (opponent
      names appear); other tabs ship as date-only stubs that need a tab click
      to hydrate — *but the dates are already there*. Since the fatigue
      feature only uses dates (we sum minutes per team per day), date-stub
      rows are sufficient.
    * We therefore enumerate every tab whose label is *not* an international
      competition (Euroleague/Eurocup/NBA/FIBA) and read dates from each
      corresponding pane. This naturally captures regular-season + domestic
      playoffs + domestic cup, which all add training load.
    """
    if not team.url:
        return []

    url = f"{team.url.rstrip('/')}/schedule/{season}"
    try:
        resp = await page.goto(url, timeout=cfg.nav_timeout_ms, wait_until="domcontentloaded")
    except Exception:
        return []
    if resp is None or resp.status >= 400:
        return []

    try:
        await page.wait_for_selector("a.list-group-item[data-toggle='list']",
                                     timeout=cfg.selector_timeout_ms)
    except Exception:
        return []

    tabs = await page.query_selector_all("a.list-group-item[data-toggle='list']")
    if not tabs:
        return []

    today = pd.Timestamp.now().normalize()
    cutoff = None
    if cfg.lookback_days is not None and season >= today.year:
        cutoff = today - pd.Timedelta(days=cfg.lookback_days)

    matches: List[DomesticMatch] = []
    for tab in tabs:
        try:
            label = (await tab.inner_text()).strip()
            href = await tab.get_attribute("href")
            if not label or not href or not href.startswith("#"):
                continue
            low = label.lower()
            if any(token in low for token in _EXCLUDE_TABS):
                continue

            pane = await page.query_selector(href)
            if pane is None:
                continue
            tables = await pane.query_selector_all("table")
            if not tables:
                continue
            rows = await tables[0].query_selector_all("tbody tr")
            for row in rows:
                tds = await row.query_selector_all("td")
                if not tds:
                    continue
                date_str = (await tds[0].inner_text()).strip()
                match_date = _parse_date(date_str)
                if match_date is None or match_date >= today:
                    continue
                if cutoff is not None and match_date < cutoff:
                    continue
                # Opponent best-effort — usually empty for non-EuroLeague tabs;
                # we keep it for downstream debugging only.
                opponent = ""
                links = await row.query_selector_all("a")
                if len(links) >= 2:
                    opp_raw = (await links[1].inner_text()).strip()
                    for prefix in ("vs\xa0", "@\xa0", "vs ", "@ "):
                        if opp_raw.startswith(prefix):
                            opp_raw = opp_raw[len(prefix):]
                            break
                    opponent = opp_raw.split("\n")[0].strip()
                matches.append(
                    DomesticMatch(
                        el_code=team.el_code,
                        competition=team.competition,
                        match_date=match_date,
                        minutes=MINUTES_PER_GAME,
                        opponent=opponent,
                    )
                )
        except Exception:
            continue

    return matches


PROVIDERS = {
    "proballers": _parse_proballers,
}


def _parse_date(raw: Optional[str]) -> Optional[pd.Timestamp]:
    """Parse heterogeneous date strings to a normalised UTC ``Timestamp``."""
    if not raw:
        return None
    raw = str(raw).strip()
    ts = pd.to_datetime(raw, errors="coerce", utc=True)
    if pd.isna(ts):
        # try day-first European formats
        ts = pd.to_datetime(raw, errors="coerce", dayfirst=True, utc=True)
    if pd.isna(ts):
        return None
    return ts.normalize().tz_localize(None)


# ---------------------------------------------------------------------------
# Async orchestration
# ---------------------------------------------------------------------------

async def _scrape_team_season(
    browser,
    team: DomesticTeam,
    season: int,
    cfg: ScrapeConfig,
    semaphore: asyncio.Semaphore,
) -> List[DomesticMatch]:
    """Scrape one (team, season) pair with retries/backoff; never raises."""
    parser = PROVIDERS.get(cfg.provider)
    if parser is None:
        return []

    async with semaphore:
        for attempt in range(1, cfg.max_retries + 1):
            context = None
            try:
                context = await browser.new_context(user_agent=cfg.user_agent)
                page = await context.new_page()
                page.set_default_timeout(cfg.selector_timeout_ms)
                matches = await parser(page, team, cfg, season)
                await context.close()
                await asyncio.sleep(random.uniform(cfg.min_delay_s, cfg.max_delay_s))
                print(f"  [domestic] {team.el_code} {season}: {len(matches)} games")
                return matches
            except Exception as exc:  # noqa: BLE001 - defensive by design
                if context is not None:
                    try:
                        await context.close()
                    except Exception:
                        pass
                if attempt >= cfg.max_retries:
                    print(f"  [domestic] {team.el_code} {season}: giving up after "
                          f"{attempt} attempts ({type(exc).__name__}: {exc})")
                    return []
                backoff = cfg.backoff_base_s * (2 ** (attempt - 1))
                backoff += random.uniform(0, 1.0)
                print(f"  [domestic] {team.el_code} {season}: attempt {attempt} failed "
                      f"({type(exc).__name__}); retrying in {backoff:.1f}s")
                await asyncio.sleep(backoff)
    return []


async def scrape_domestic_matches_async(
    teams: Sequence[DomesticTeam],
    cfg: ScrapeConfig = ScrapeConfig(),
) -> pd.DataFrame:
    """Scrape domestic fixtures for *teams* across every season in ``cfg.seasons``.

    Returns a tidy DataFrame with columns
    ``el_code, competition, match_date, minutes, opponent``. Any (team, season)
    pair that fails contributes no rows; the function never raises.
    """
    async_playwright = _import_async_playwright()

    semaphore = asyncio.Semaphore(cfg.max_concurrency)
    rows: List[DomesticMatch] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=cfg.headless)
        try:
            tasks = [
                _scrape_team_season(browser, t, s, cfg, semaphore)
                for t in teams
                for s in cfg.seasons
            ]
            print(f"[domestic] launching {len(tasks)} (team×season) scrapes "
                  f"(teams={len(teams)} × seasons={len(cfg.seasons)})")
            results = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            await browser.close()

    for res in results:
        if isinstance(res, list):
            rows.extend(res)

    if not rows:
        return pd.DataFrame(
            columns=["el_code", "competition", "match_date", "minutes", "opponent"]
        )
    df = pd.DataFrame([r.__dict__ for r in rows])
    # Deduplicate: a row could in principle appear in two seasons if pagination
    # overlaps; (el_code, match_date, opponent) is the natural unique key.
    df = df.drop_duplicates(subset=["el_code", "match_date", "opponent"]).reset_index(drop=True)
    return df


def run_domestic_scrape(
    teams: Optional[Iterable[DomesticTeam]] = None,
    cfg: ScrapeConfig = ScrapeConfig(),
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Synchronous entry point — runs the async scraper and optionally caches.

    Parameters
    ----------
    teams : iterable of :class:`DomesticTeam`; defaults to ``TEAM_DOMESTIC_MAP``.
    cfg : scraping configuration.
    cache_path : if given, the resulting DataFrame is written as parquet/pickle
        and reused; this is the polite default — scrape once per matchday.
    """
    team_list = list(teams) if teams is not None else list(TEAM_DOMESTIC_MAP.values())

    try:
        df = asyncio.run(scrape_domestic_matches_async(team_list, cfg))
    except RuntimeError as exc:
        # e.g. already-running event loop (Jupyter) or Playwright missing
        print(f"  [domestic] scrape unavailable: {exc}")
        return pd.DataFrame(
            columns=["el_code", "competition", "match_date", "minutes", "opponent"]
        )

    if cache_path is not None and not df.empty:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_pickle(cache_path)
        except Exception as exc:  # noqa: BLE001
            print(f"  [domestic] could not cache results: {exc}")
    return df


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_rolling_domestic_minutes(
    matches_df: pd.DataFrame,
    as_of: pd.Timestamp,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> Dict[str, float]:
    """Sum domestic minutes in the trailing ``window_days`` before ``as_of``.

    Returns ``{el_code: minutes}`` for the window ``[as_of - window_days, as_of)``.
    The window is half-open on the right so a same-day domestic game (rare before
    a EuroLeague tip) does not double-count.
    """
    if matches_df is None or matches_df.empty:
        return {}

    df = matches_df.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    as_of = pd.Timestamp(as_of).normalize()
    start = as_of - pd.Timedelta(days=window_days)

    window = df[(df["match_date"] >= start) & (df["match_date"] < as_of)]
    if window.empty:
        return {}
    grouped = window.groupby("el_code")["minutes"].sum()
    return {str(k): float(v) for k, v in grouped.items()}


def domestic_fatigue_diff_for_schedule(
    schedule_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    window_days: int = DEFAULT_WINDOW_DAYS,
    default_as_of: Optional[pd.Timestamp] = None,
) -> Dict[int, float]:
    """Per-game ``domestic_fatigue_diff`` = home load − away load.

    Uses each game's ``game_date`` as the ``as_of`` reference; falls back to
    ``default_as_of`` (or today) when a game has no date. A positive value means
    the home team carried more recent domestic load into the game.
    """
    out: Dict[int, float] = {}
    if schedule_df is None or schedule_df.empty:
        return out
    default_as_of = pd.Timestamp(default_as_of or pd.Timestamp.now().normalize())

    for _, g in schedule_df.iterrows():
        gc = int(g["Gamecode"])
        gd = g.get("game_date", pd.NaT)
        as_of = pd.Timestamp(gd) if pd.notna(gd) else default_as_of
        loads = compute_rolling_domestic_minutes(matches_df, as_of, window_days)
        h = loads.get(str(g["home_team"]), 0.0)
        a = loads.get(str(g["away_team"]), 0.0)
        out[gc] = float(h - a)
    return out
