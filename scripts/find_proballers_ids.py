"""One-shot script to find correct Proballers team IDs for all EuroLeague clubs."""
import asyncio
from playwright.async_api import async_playwright

SEARCHES = {
    "BAR": "fc barcelona basketball",
    "OLY": "olympiacos basketball",
    "MIL": "olimpia milano basketball",
    "ULK": "fenerbahce beko basketball",
    "TEL": "maccabi tel aviv basketball",
    "MUN": "fc bayern munich basketball",
    "BER": "alba berlin basketball",
    "MCO": "as monaco basketball",
    "ASV": "ldlc asvel basketball",
    "PRS": "paris basketball",
    "PAR": "partizan basketball",
    "RED": "crvena zvezda basketball",
    "ZAL": "zalgiris basketball",
    "IST": "anadolu efes basketball",
    "MAD": "real madrid basketball",
    "VIR": "virtus bologna basketball",
    "BAS": "baskonia basketball",
    "PAM": "valencia basket basketball",
}

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")


async def find_ids():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        for code, query in SEARCHES.items():
            context = await browser.new_context(user_agent=UA)
            page = await context.new_page()
            url = f"https://www.proballers.com/basketball/search?q={query.replace(' ', '+')}"
            try:
                await page.goto(url, timeout=30000, wait_until="domcontentloaded")
                links = await page.query_selector_all("a[href*='/basketball/team/']")
                seen = set()
                results = []
                for lnk in links:
                    href = await lnk.get_attribute("href")
                    txt = (await lnk.inner_text()).strip().replace("\n", " ")[:50]
                    if href and href not in seen:
                        seen.add(href)
                        results.append(f"  {txt} -> {href}")
                print(f"\n{code} ({query}):")
                for r in results[:4]:
                    print(r)
            except Exception as e:
                print(f"{code}: ERROR {e}")
            finally:
                await context.close()
        await browser.close()

asyncio.run(find_ids())
