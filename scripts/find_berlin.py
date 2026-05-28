"""Find ALBA Berlin's Proballers ID by scanning the BBL league listing."""
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
        # Visit Bayern (team 2020) to find BBL league link
        await page.goto("https://www.proballers.com/basketball/team/2020/fc-bayern-munich",
                        timeout=30000, wait_until="domcontentloaded")
        await page.wait_for_timeout(1500)
        anchors = await page.query_selector_all("a[href*='/basketball/league/']")
        print("League links on Bayern page:")
        bbl_url = None
        for a in anchors:
            href = await a.get_attribute("href")
            txt = (await a.inner_text()).strip().replace("\n", " ")[:60]
            if not href:
                continue
            print(f"  {href}  --  {txt!r}")
            if "bbl" in href.lower() or "germany" in href.lower() or "bundesliga" in href.lower():
                bbl_url = href
        if bbl_url and not bbl_url.startswith("http"):
            bbl_url = f"https://www.proballers.com{bbl_url}"
        if not bbl_url:
            print("\nNo BBL link found from Bayern page; trying guesses…")
        else:
            print(f"\nVisiting BBL listing: {bbl_url}")
            target = bbl_url.rstrip("/")
            if not target.endswith("/teams"):
                target = target + "/teams"
            await page.goto(target, timeout=30000, wait_until="domcontentloaded")
            await page.wait_for_timeout(1500)
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
                txt = (await a.inner_text()).strip().replace("\n", " ")[:60]
                seen[key] = txt
            print(f"\nBBL teams ({len(seen)}):")
            for key, txt in sorted(seen.items(), key=lambda kv: kv[1].lower()):
                tid, slug = key.split("/")
                print(f"  id={tid:>6}  slug={slug:<40}  text={txt!r}")
        await browser.close()


asyncio.run(main())
