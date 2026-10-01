"""Direct web scraping: httpx -> (Wayback fallback) -> optional Playwright render.
Text via Trafilatura, links via BeautifulSoup, both with regex fallbacks if not installed."""
import asyncio
import html
import re
from dataclasses import dataclass, field
from urllib.parse import urljoin

from . import config

try:
    import trafilatura
except ImportError:  # pragma: no cover
    trafilatura = None
try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None
try:
    from googlenewsdecoder import gnews_decoder_async
except ImportError:  # pragma: no cover
    gnews_decoder_async = None

_pw_sem = asyncio.Semaphore(2)
_BAD_LINK = ("sharer", "intent/tweet", "/login", "/subscribe", "mailto:", "javascript:")
_WAYBACK = re.compile(r"^https?://web\.archive\.org/web/\d+[a-z_]*/")
_redirect_cache: dict[str, str] = {}
_page_cache: dict[str, "Page"] = {}


@dataclass
class Page:
    url: str
    html: str
    text: str
    title: str
    links: list = field(default_factory=list)
    via: str = "live"  # live | wayback | playwright


def extract_text(raw, url=None, limit=6000):
    if raw.startswith("%PDF"):
        return "(PDF document; text extraction not attempted)"
    t = None
    if trafilatura:
        try:
            t = trafilatura.extract(raw, url=url, favor_recall=True)
        except Exception:
            t = None
    if not t:
        t = re.sub(r"(?is)<(script|style|nav|footer|header|aside).*?</\1>", " ", raw)
        t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    return re.sub(r"\s+", " ", t).strip()[:limit]


def extract_title(raw):
    if BeautifulSoup:
        t = BeautifulSoup(raw, "html.parser").find("title")
        return t.get_text(strip=True)[:200] if t else ""
    m = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw)
    return html.unescape(m.group(1)).strip()[:200] if m else ""


def extract_links(raw, base):
    """Outbound links from the main content area (article/main), Wayback rewriting unwrapped."""
    hrefs = []
    if BeautifulSoup:
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["nav", "footer", "header", "aside", "script", "style", "form"]):
            tag.decompose()
        root = soup.find("article") or soup.find("main") or soup.body or soup
        hrefs = [a["href"] for a in root.find_all("a", href=True)]
    else:
        seg = " ".join(re.findall(r"(?is)<(?:p|li|blockquote|figure)\b.*?</(?:p|li|blockquote|figure)>", raw)) or raw
        hrefs = [m.group(1) for m in re.finditer(r'(?is)<a\b[^>]*?href=["\']([^"\'#]+)', seg)]
    out = []
    for h in hrefs:
        u = _WAYBACK.sub("", urljoin(base, html.unescape(h).split("#")[0]))
        if u.startswith("http") and not any(x in u for x in _BAD_LINK) and u not in out:
            out.append(u)
    return out


async def render_js(url):
    """Render with headless Chromium. Needs `pip install playwright && playwright install chromium`."""
    from playwright.async_api import async_playwright
    async with _pw_sem, async_playwright() as p:
        b = await p.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"])
        try:
            pg = await b.new_page(user_agent=config.BROWSER_UA)
            await pg.goto(url, wait_until="domcontentloaded", timeout=8000)
            await pg.wait_for_timeout(300)
            return pg.url, await pg.content()
        finally:
            await b.close()


async def resolve_redirect(url, c=None):
    """Resolve JS/RSS redirects (e.g. Google News links) to publisher URL (~200ms-1s via batchexecute)."""
    if not url:
        return None
    if url in _redirect_cache:
        return _redirect_cache[url]
    if "news.google.com" not in url:
        return url

    # 1. Fast Google News decode via batchexecute RPC (~1s vs 17s Playwright)
    if gnews_decoder_async:
        try:
            res = await gnews_decoder_async(url)
            if isinstance(res, dict) and (res.get("status") or res.get("success")):
                dec = res.get("decoded_url")
                if dec:
                    _redirect_cache[url] = dec
                    return dec
        except Exception:
            pass

    # 2. Fast Playwright fallback if enabled
    if not config.USE_PLAYWRIGHT:
        return None
    from playwright.async_api import async_playwright
    try:
        async with _pw_sem, async_playwright() as p:
            b = await p.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"])
            try:
                pg = await b.new_page(user_agent=config.BROWSER_UA)
                await pg.goto(url, wait_until="domcontentloaded", timeout=8000)
                await pg.wait_for_url(lambda u: "news.google.com" not in u, timeout=4000)
                final_url = pg.url
                if final_url and "news.google.com" not in final_url:
                    _redirect_cache[url] = final_url
                    return final_url
                return None
            finally:
                await b.close()
    except Exception:
        return None


async def fetch_page(c, url) -> Page:
    """Live fetch, falling back to the Wayback Machine; renders with Playwright when the text is thin."""
    if url in _page_cache:
        return _page_cache[url]
    via = "live"
    try:
        r = await c.get(url, headers={"User-Agent": config.BROWSER_UA, "Accept-Language": "en-US,en;q=0.9"})
        r.raise_for_status()
    except Exception:
        try:
            # Wayback fallback with tight 4s timeout so it doesn't freeze the pipeline
            r = await c.get("https://web.archive.org/web/2/" + url, timeout=4.0)
            r.raise_for_status()
            via = "wayback"
        except Exception:
            raise
    raw = r.text[:600_000]
    text = extract_text(raw, url)
    if config.USE_PLAYWRIGHT and via == "live" and len(text) < 300:
        try:
            final, raw2 = await render_js(url)
            t2 = extract_text(raw2, final)
            if len(t2) > len(text):
                raw, text, via = raw2, t2, "playwright"
        except Exception:
            pass
    page = Page(url=url, html=raw, text=text, title=extract_title(raw), links=extract_links(raw, url), via=via)
    _page_cache[url] = page
    return page
