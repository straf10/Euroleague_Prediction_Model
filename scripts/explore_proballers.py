"""Exploration script: find Proballers EuroLeague team URLs + historical season pattern.

Phase A — try several likely competition-listing URLs on Proballers, dump every
team link found so we can map el_code -> proballers_url.
Phase B — inspect Real Madrid's schedule page for a season selector (dropdown,
query param, or path segment) so we can build historical URLs.

Run:
    python scripts/explore_proballers.py
"""
from __future__ import annotations

import asyncio
import re
from playwright.async_api import async_playwright

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Candidate listing pages — we try each, take the first that returns >=15 team links
CANDIDATE_LISTINGS = [
    "https://www.proballers.com/basketball/league/2/euroleague",
    "https://www.proballers.com/basketball/league/2/euroleague/teams",
    "https://www.proballers.com/basketball/league/2/euroleague/standings",
    "https://www.proballers.com/basketball/euroleague",
    "https://www.proballers.com/basketball/competition/euroleague",
]

MAD_SCHEDULE = "https://www.proballers.com/basketball/team/160/real-madrid/schedule"


async def explore():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(user_agent=UA)
        page = await context.new_page()

        # ---- Phase A: find competition listing ----
        print("=" * 70)
        print("PHASE A: Finding EuroLeague team listing on Proballers")
        print("=" * 70)
        best_url = None
        best_links: dict[str, str] = {}
        for url in CANDIDATE_LISTINGS:
            try:
                resp = await page.goto(url, timeout=30000, wait_until="domcontentloaded")
                status = resp.status if resp else "?"
                # Wait briefly for any client-side render
                await page.wait_for_timeout(1500)
                anchors = await page.query_selector_all("a[href*='/basketball/team/']")
                seen: dict[str, str] = {}
                for a in anchors:
                    href = await a.get_attribute("href")
                    if not href:
                        continue
                    # normalise to numeric-id team pages only
                    m = re.search(r"/basketball/team/(\d+)/([^/]+)", href)
                    if not m:
                        continue
                    team_id, slug = m.group(1), m.group(2)
                    key = f"{team_id}/{slug}"
                    if key in seen:
                        continue
                    txt = (await a.inner_text()).strip().replace("\n", " ")[:60]
                    seen[key] = txt
                print(f"  {url}\n    status={status}  unique team links={len(seen)}")
                if len(seen) > len(best_links):
                    best_links = seen
                    best_url = url
            except Exception as e:
                print(f"  {url}\n    ERROR {type(e).__name__}: {e}")

        print(f"\nBest listing: {best_url} -> {len(best_links)} teams")
        for key, txt in sorted(best_links.items(), key=lambda kv: kv[1]):
            print(f"  https://www.proballers.com/basketball/team/{key}  --  {txt}")

        # ---- Phase B: discover historical-season URL pattern ----
        print("\n" + "=" * 70)
        print("PHASE B: Real Madrid schedule — looking for season selector")
        print("=" * 70)
        await page.goto(MAD_SCHEDULE, timeout=30000, wait_until="domcontentloaded")
        await page.wait_for_timeout(2000)

        # Look for <select> elements (season dropdowns usually)
        selects = await page.query_selector_all("select")
        print(f"\nFound {len(selects)} <select> elements:")
        for i, sel in enumerate(selects):
            name = await sel.get_attribute("name")
            sid = await sel.get_attribute("id")
            opts = await sel.query_selector_all("option")
            opt_data = []
            for o in opts[:8]:
                v = await o.get_attribute("value")
                t = (await o.inner_text()).strip()
                opt_data.append(f"{v!r}={t!r}")
            print(f"  [{i}] name={name!r} id={sid!r}  options[:8]={opt_data}")

        # Look for links containing 'season' or year ranges anywhere on the page
        all_links = await page.query_selector_all("a[href]")
        season_pat = re.compile(r"(season|year|/20\d{2})", re.I)
        season_links = []
        for a in all_links:
            href = await a.get_attribute("href")
            if not href:
                continue
            if season_pat.search(href):
                txt = (await a.inner_text()).strip().replace("\n", " ")[:40]
                season_links.append((href, txt))
        # Deduplicate
        season_links = list(dict.fromkeys(season_links))[:20]
        print(f"\nLinks containing 'season'/'year'/'/20xx' (first 20 unique):")
        for href, txt in season_links:
            print(f"  {href}  --  {txt}")

        # Try the most common URL conventions directly
        print("\nProbing common historical-URL conventions:")
        candidates = [
            "https://www.proballers.com/basketball/team/160/real-madrid/schedule?season=2023-2024",
            "https://www.proballers.com/basketball/team/160/real-madrid/schedule/2023-2024",
            "https://www.proballers.com/basketball/team/160/real-madrid/schedule?year=2024",
            "https://www.proballers.com/basketball/team/160/real-madrid/2023-2024/schedule",
        ]
        for url in candidates:
            try:
                resp = await page.goto(url, timeout=20000, wait_until="domcontentloaded")
                await page.wait_for_timeout(800)
                final = page.url
                status = resp.status if resp else "?"
                tables = await page.query_selector_all("table")
                rows = 0
                if len(tables) >= 2:
                    body_rows = await tables[1].query_selector_all("tbody tr")
                    rows = len(body_rows)
                print(f"  {url}\n    -> status={status}  final={final}  table[1] rows={rows}")
            except Exception as e:
                print(f"  {url}\n    ERROR {type(e).__name__}: {e}")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(explore())
