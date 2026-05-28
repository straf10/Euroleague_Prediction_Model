"""Dump all EuroLeague team URLs from Proballers league 177."""
import asyncio
import re
from playwright.async_api import async_playwright

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")


async def main():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(user_agent=UA)
        page = await ctx.new_page()
        # Try the /teams sub-page first (full listing); fallback to root
        for url in [
            "https://www.proballers.com/basketball/league/177/euroleague/teams",
            "https://www.proballers.com/basketball/league/177/euroleague",
        ]:
            await page.goto(url, timeout=30000, wait_until="domcontentloaded")
            await page.wait_for_timeout(2000)
            anchors = await page.query_selector_all("a[href*='/basketball/team/']")
            seen: dict[str, str] = {}
            for a in anchors:
                href = await a.get_attribute("href")
                if not href:
                    continue
                m = re.search(r"/basketball/team/(\d+)/([^/]+)", href)
                if not m:
                    continue
                key = f"{m.group(1)}/{m.group(2)}"
                if key in seen:
                    continue
                txt = (await a.inner_text()).strip().replace("\n", " ")[:80]
                seen[key] = txt
            print(f"\n=== {url}  ({len(seen)} teams) ===")
            for key, txt in sorted(seen.items(), key=lambda kv: kv[1].lower()):
                tid, slug = key.split("/")
                print(f"  id={tid:>6}  slug={slug:<45}  text={txt!r}")
            if len(seen) >= 18:
                break
        await browser.close()

asyncio.run(main())
