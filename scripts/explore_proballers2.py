"""Phase 2 exploration: find the real EuroLeague league ID + map all 19 EL teams.

Approach:
A) Visit Real Madrid's team root page and extract competition/league links — one
   of them will be EuroLeague with the correct league ID.
B) If that fails, sweep a small range of /basketball/league/{id} pages looking
   for one whose listed teams include Real Madrid + Barcelona + Olympiacos.
C) Once found, dump every team link on the EL listing for the current season.
"""
from __future__ import annotations

import asyncio
import re
from playwright.async_api import async_playwright

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

EL_MARKERS = {"real madrid", "fc barcelona", "olympiacos", "panathinaikos", "fenerbahce"}


async def explore():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(user_agent=UA)
        page = await context.new_page()

        # ---- A: scan Real Madrid root page for competition/league links ----
        print("=" * 70)
        print("PHASE A: Real Madrid root page — search for league/competition links")
        print("=" * 70)
        await page.goto("https://www.proballers.com/basketball/team/160/real-madrid",
                        timeout=30000, wait_until="domcontentloaded")
        await page.wait_for_timeout(1500)
        anchors = await page.query_selector_all("a[href*='/basketball/league/']")
        seen: dict[str, str] = {}
        for a in anchors:
            href = await a.get_attribute("href")
            if not href:
                continue
            txt = (await a.inner_text()).strip().replace("\n", " ")[:60]
            seen[href] = txt
        for href, txt in seen.items():
            print(f"  {href}  --  {txt!r}")

        # ---- B: brute-force scan league IDs 1..40 for EuroLeague markers ----
        print("\n" + "=" * 70)
        print("PHASE B: Scanning league IDs 1..40 for EuroLeague team markers")
        print("=" * 70)
        candidates = []
        for lid in range(1, 41):
            url = f"https://www.proballers.com/basketball/league/{lid}/-"
            try:
                resp = await page.goto(url, timeout=15000, wait_until="domcontentloaded")
                if not resp or resp.status >= 400:
                    continue
                await page.wait_for_timeout(400)
                # Read page text once and search for markers
                body = (await page.inner_text("body")).lower()
                hits = sum(1 for m in EL_MARKERS if m in body)
                title = await page.title()
                if hits >= 2:
                    print(f"  league/{lid:<3}  hits={hits}/5  title={title[:60]!r}  final={page.url}")
                    candidates.append((lid, hits, page.url, title))
            except Exception:
                continue

        if not candidates:
            print("\nNo league ID matched. Trying a few keyword-named slugs:")
            for slug in ["euroleague", "euro-league", "eurolega",
                        "-european-league", "european-basketball-league"]:
                for lid in [1, 7, 12, 24, 100, 101]:
                    url = f"https://www.proballers.com/basketball/league/{lid}/{slug}"
                    try:
                        resp = await page.goto(url, timeout=15000, wait_until="domcontentloaded")
                        if resp and resp.status == 200:
                            body = (await page.inner_text("body")).lower()
                            hits = sum(1 for m in EL_MARKERS if m in body)
                            print(f"  {url}  hits={hits}/5")
                    except Exception:
                        pass

        # ---- C: if we found a winner, dump all team links from that listing ----
        if candidates:
            # pick the league with the most hits
            candidates.sort(key=lambda c: -c[1])
            lid, hits, final_url, title = candidates[0]
            print(f"\nBest match: league/{lid} ({title}) hits={hits}")
            url = f"https://www.proballers.com/basketball/league/{lid}/-/teams"
            try:
                await page.goto(url, timeout=20000, wait_until="domcontentloaded")
                await page.wait_for_timeout(1500)
            except Exception:
                pass
            # fall back to the league root
            anchors = await page.query_selector_all("a[href*='/basketball/team/']")
            seen2: dict[str, str] = {}
            for a in anchors:
                href = await a.get_attribute("href")
                if not href:
                    continue
                m = re.search(r"/basketball/team/(\d+)/([^/]+)", href)
                if not m:
                    continue
                key = f"{m.group(1)}/{m.group(2)}"
                if key in seen2:
                    continue
                txt = (await a.inner_text()).strip().replace("\n", " ")[:60]
                seen2[key] = txt
            print(f"\nEuroLeague team links from league {lid} listing ({len(seen2)}):")
            for key, txt in sorted(seen2.items(), key=lambda kv: kv[1].lower()):
                print(f"  https://www.proballers.com/basketball/team/{key}  --  {txt}")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(explore())
