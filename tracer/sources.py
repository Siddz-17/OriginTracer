"""Data-source clients: GDELT, Reddit, Bluesky, RSS/Google News, DuckDuckGo, Wikipedia, Wayback CDX.

DuckDuckGo: uses the `duckduckgo_search` (ddgs) package – text and news searches.
Falls back to a self-hosted SearXNG instance (SEARXNG_URL env var) if ddgs throttles.
"""
import asyncio
import html
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

from .classify import UTC, parse_iso
from .config import BROWSER_UA, DEFAULT_FEEDS, UA
from .outlets import FACT_CHECK, MAJOR, WIRE, site_groups

logger = logging.getLogger(__name__)


_gdelt_lock = asyncio.Lock()
_gdelt_last = [0.0]
GDELT_INTERVAL = float(os.getenv("GDELT_INTERVAL", "5.5"))  # GDELT rejects >1 request per 5 s


async def gdelt(c, q):
    """GDELT DOC 2.0 API – sorted by relevance, most-recent 3-month window.

    Calls are serialised behind a lock: GDELT answers parallel requests with HTTP 429
    "Please limit requests to one every 5 seconds", which used to fail every query."""
    try:
        for attempt in range(2):
            async with _gdelt_lock:
                wait = _gdelt_last[0] + GDELT_INTERVAL * (attempt + 1) - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
                try:
                    r = await c.get("https://api.gdeltproject.org/api/v2/doc/doc", params={
                        "query": q, "mode": "artlist", "format": "json", "maxrecords": 75,
                        # DateAsc returned the oldest 75 items, not the best matches.  Use relevance.
                        "sort": "HybridRel", "timespan": "3months"}, timeout=20.0)
                finally:
                    _gdelt_last[0] = time.monotonic()
            if not (r.status_code == 429 or "limit requests" in r.text[:200]):
                break
        else:
            logger.warning("gdelt: throttled twice for %r; skipping", q)
            return []
        r.raise_for_status()
        try:
            data = r.json()
        except ValueError:  # GDELT returns plain-text errors (bad query syntax, phrase too short...)
            logger.warning("gdelt: non-JSON reply for %r: %s", q, r.text[:150])
            return []
        articles = data.get("articles") or []
        if not articles:
            logger.warning("gdelt: query %r returned 0 articles (status %s, body preview: %s)",
                           q, r.status_code, r.text[:200])
        out = []
        for a in articles:
            try:
                ts = datetime.strptime(a["seendate"], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
            except (KeyError, ValueError) as e:
                logger.debug("gdelt: skipping article with bad seendate: %s", e)
                continue
            out.append(dict(source="gdelt", url=a["url"], author=a.get("domain", ""), ts=ts,
                            text=a.get("title", ""), engagement=0,
                            meta={"lang": a.get("language"), "country": a.get("sourcecountry"),
                                  "outlet_domain": a.get("domain", "")}))
        return out
    except Exception as e:
        logger.error("gdelt: query %r failed: %s: %s", q, type(e).__name__, e)
        raise


async def reddit(c, q):
    r = await c.get("https://www.reddit.com/search.json",
                    params={"q": q, "sort": "relevance", "t": "all", "limit": 100},
                    headers={"User-Agent": UA})
    r.raise_for_status()
    out = []
    for ch in r.json()["data"]["children"]:
        d = ch["data"]
        links = []
        dest = d.get("url_overridden_by_dest") or d.get("url") or ""
        if dest and "reddit.com" not in dest:
            links.append(dest)
        for cp in d.get("crosspost_parent_list", []) or []:
            links.append("https://reddit.com" + cp["permalink"])
        out.append(dict(source="reddit", url="https://reddit.com" + d["permalink"],
                        author=d.get("author", ""),
                        ts=datetime.fromtimestamp(d["created_utc"], UTC),
                        text=(d.get("title", "") + " " + (d.get("selftext") or ""))[:600],
                        engagement=int(d.get("score", 0)) + int(d.get("num_comments", 0)),
                        meta={"subreddit": d.get("subreddit"), "links": links}))
    return out


_bsky_tok = {"v": None}


async def _bsky_token(c, force=False):
    if _bsky_tok["v"] and not force:
        return _bsky_tok["v"]
    h, p = os.getenv("BSKY_HANDLE"), os.getenv("BSKY_APP_PASSWORD")
    if h and p:
        try:
            r = await c.post("https://bsky.social/xrpc/com.atproto.server.createSession",
                             json={"identifier": h.strip(), "password": p.strip()})
            if r.status_code == 200:
                _bsky_tok["v"] = r.json().get("accessJwt")
        except Exception:
            pass
    return _bsky_tok["v"]


async def bluesky(c, q):
    tok = await _bsky_token(c)
    base = "https://bsky.social" if tok else "https://public.api.bsky.app"
    headers = {"Authorization": f"Bearer {tok}"} if tok else {}
    out = []
    for sort in ("latest", "top"):
        try:
            r = await c.get(f"{base}/xrpc/app.bsky.feed.searchPosts",
                            params={"q": q, "limit": 100, "sort": sort}, headers=headers)
            if r.status_code in (401, 403) and tok:
                tok = await _bsky_token(c, force=True)
                headers = {"Authorization": f"Bearer {tok}"} if tok else {}
                r = await c.get(f"{base}/xrpc/app.bsky.feed.searchPosts",
                                params={"q": q, "limit": 100, "sort": sort}, headers=headers)
            r.raise_for_status()
        except Exception:
            continue
        for p in r.json().get("posts", []):
            rec = p.get("record", {})
            ts = parse_iso(rec.get("createdAt", ""))
            if not ts:
                continue
            handle = p["author"]["handle"]
            links, quotes = [], []
            for f in rec.get("facets", []) or []:
                for ft in f.get("features", []):
                    if ft.get("uri"):
                        links.append(ft["uri"])
            emb = p.get("embed") or {}
            if (emb.get("external") or {}).get("uri"):
                links.append(emb["external"]["uri"])
            rq = emb.get("record") or {}
            qu = rq.get("uri") or (rq.get("record") or {}).get("uri")
            if qu:
                quotes.append(qu)
            out.append(dict(source="bluesky",
                            url=f"https://bsky.app/profile/{handle}/post/{p['uri'].split('/')[-1]}",
                            author=handle, ts=ts, text=rec.get("text", ""),
                            engagement=p.get("likeCount", 0) + p.get("repostCount", 0) * 2 + p.get("replyCount", 0),
                            meta={"at_uri": p["uri"], "links": links, "quotes": quotes}))
    return out


def _rss_items(content, default_author=""):
    root = ET.fromstring(content)
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        desc = re.sub(r"<[^>]+>", " ", html.unescape(it.findtext("description") or ""))
        try:
            ts = parsedate_to_datetime(it.findtext("pubDate"))
            ts = ts if ts.tzinfo else ts.replace(tzinfo=UTC)
        except Exception:
            continue
        src = it.find("source")
        outlet = urlparse(src.get("url") or "").netloc.lower().removeprefix("www.") if src is not None else ""
        name = ((src.text if src is not None else "") or "").strip()
        if name and title.endswith(" - " + name):  # Google News appends " - Publisher" to titles
            title = title[: -len(name) - 3]
        desc = re.sub(r"\s+", " ", desc).strip()
        if name and desc.endswith(name):
            desc = desc[: -len(name)].strip()
        text = title if desc.startswith(title[:40]) else f"{title}. {desc}"
        yield dict(source="news", url=(it.findtext("link") or "").strip(),
                   author=name or default_author,
                   ts=ts, text=text.strip()[:500], engagement=0,
                   meta={"outlet_domain": outlet or default_author})


_gnews_sem = asyncio.Semaphore(4)


async def google_news(c, q):
    async with _gnews_sem:
        r = await c.get("https://news.google.com/rss/search",
                        params={"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"},
                        headers={"User-Agent": BROWSER_UA})
    r.raise_for_status()
    return list(_rss_items(r.content))


async def outlets(c, q):
    """Google News restricted to wire services and major newsrooms (searched before anything else)."""
    res = await asyncio.gather(*(google_news(c, f"{q} {g}") for g in site_groups((WIRE, MAJOR))),
                               return_exceptions=True)
    return [it for r in res if not isinstance(r, Exception) for it in r]


async def factcheckers(c, q):
    """Google News restricted to fact-checking organisations, plus Google's ClaimReview index if keyed."""
    res = await asyncio.gather(*(google_news(c, f"{q} {g}") for g in site_groups((FACT_CHECK,))),
                               return_exceptions=True)
    out = [it for r in res if not isinstance(r, Exception) for it in r]
    key = os.getenv("GOOGLE_FACTCHECK_API_KEY")
    if key:
        try:
            r = await c.get("https://factchecktools.googleapis.com/v1alpha1/claims:search",
                            params={"query": q, "languageCode": "en", "pageSize": 20, "key": key})
            r.raise_for_status()
            for cl in r.json().get("claims", []):
                for rv in cl.get("claimReview", []):
                    ts = parse_iso(rv.get("reviewDate") or cl.get("claimDate") or "")
                    if not rv.get("url") or not ts:
                        continue
                    pub = rv.get("publisher", {})
                    out.append(dict(source="news", url=rv["url"], author=pub.get("name") or pub.get("site", ""),
                                    ts=ts, engagement=0,
                                    text=f"Fact check: {cl.get('text', '')}. Rating: {rv.get('textualRating', '')}. "
                                         f"{rv.get('title', '')}"[:500],
                                    meta={"outlet_domain": pub.get("site", ""), "rating": rv.get("textualRating"),
                                          "claimant": cl.get("claimant")}))
        except Exception as e:
            logger.warning("factcheck api: %s: %s", type(e).__name__, e)
    return out


async def news(c, q):
    out = []
    try:
        out += await google_news(c, q)
    except Exception as e:
        logger.error("news/google: query %r failed: %s: %s", q, type(e).__name__, e)
    toks = [t.lower() for t in re.findall(r"\w{4,}", q)]
    need = min(2, len(toks))
    for f in os.getenv("TRACER_FEEDS", ",".join(DEFAULT_FEEDS)).split(","):
        try:
            rr = await c.get(f.strip(), headers={"User-Agent": BROWSER_UA})
            for it in _rss_items(rr.content, urlparse(f.strip()).netloc.removeprefix("www.").removeprefix("feeds.")):
                if sum(t in it["text"].lower() for t in toks) >= need:
                    out.append(it)
        except Exception as e:
            logger.debug("news/feed %s failed: %s", f.strip(), e)
    return out


# ------------------------------------------------------------------ DuckDuckGo
_DDGS_AVAILABLE = None  # cached after first import attempt


def _ddgs_module():
    global _DDGS_AVAILABLE
    if _DDGS_AVAILABLE is None:
        try:
            from ddgs import DDGS  # noqa: F401
            _DDGS_AVAILABLE = True
        except ImportError:
            _DDGS_AVAILABLE = False
    return _DDGS_AVAILABLE


async def _searxng_text(c, q, max_results=20):
    """SearXNG fallback (self-hosted).  Set SEARXNG_URL=http://localhost:8888 in .env."""
    base = os.getenv("SEARXNG_URL", "").rstrip("/")
    if not base:
        return []
    try:
        r = await c.get(f"{base}/search", params={"q": q, "format": "json", "language": "en", "pageno": 1},
                        headers={"User-Agent": UA})
        r.raise_for_status()
        results = r.json().get("results", [])[:max_results]
        now = datetime.now(UTC)
        return [dict(source="duckduckgo", url=res["url"],
                     author=urlparse(res["url"]).netloc.removeprefix("www."),
                     ts=now, text=(res.get("title", "") + ". " + res.get("content", ""))[:500],
                     engagement=0, meta={"engine": "searxng"})
                for res in results if res.get("url")]
    except Exception as e:
        logger.error("searxng: query %r failed: %s: %s", q, type(e).__name__, e)
        return []


async def duckduckgo(c, q):
    """DuckDuckGo text + news via the ddgs package; falls back to SearXNG on throttle."""
    if not _ddgs_module():
        logger.warning("duckduckgo: duckduckgo_search package not installed; falling back to SearXNG")
        return await _searxng_text(c, q)

    import asyncio as _asyncio
    from ddgs import DDGS

    now = datetime.now(UTC)
    out = []
    try:
        # ddgs is synchronous internally; run in executor to avoid blocking the event loop
        loop = _asyncio.get_running_loop()

        def _fetch():
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(q, max_results=20):
                    results.append(("text", r))
                for r in ddgs.news(q, max_results=20):
                    results.append(("news", r))
            return results

        raw = await loop.run_in_executor(None, _fetch)
        for kind, r in raw:
            url = r.get("href") or r.get("url") or ""
            if not url:
                continue
            ts_str = r.get("date") or r.get("published")
            ts = parse_iso(ts_str) if ts_str else None
            text = ((r.get("title") or "") + ". " + (r.get("body") or r.get("excerpt") or ""))[:500]
            dom = urlparse(url).netloc.lower().removeprefix("www.")
            out.append(dict(source="duckduckgo", url=url, author=dom,
                            ts=ts or now, text=text.strip(), engagement=0,
                            # text results carry no date: flag them so origin scoring ignores their timestamp
                            meta={"engine": f"ddgs_{kind}", "outlet_domain": dom, "undated": not ts}))
    except Exception as e:
        err_str = str(e).lower()
        if "ratelimit" in err_str or "202" in err_str or "429" in err_str or "blocked" in err_str:
            logger.warning("duckduckgo: throttled (%s); falling back to SearXNG", e)
            return await _searxng_text(c, q)
        logger.error("duckduckgo: query %r failed: %s: %s", q, type(e).__name__, e)
        raise
    return out


async def wikipedia(c, q):
    r = await c.get("https://en.wikipedia.org/w/api.php", params={
        "action": "query", "generator": "search", "gsrsearch": q, "gsrlimit": 4,
        "prop": "extracts|info", "exintro": 1, "explaintext": 1, "inprop": "url", "format": "json"})
    pages = r.json().get("query", {}).get("pages", {})
    return [dict(url=p["fullurl"], title=p["title"], text=p.get("extract", "")[:2500])
            for p in pages.values()]


async def wayback_first(c, url):
    r = await c.get("https://web.archive.org/cdx/search/cdx", params={
        "url": url, "output": "json", "limit": 1, "fl": "timestamp", "filter": "statuscode:200"})
    rows = r.json()
    if len(rows) > 1:
        return datetime.strptime(rows[1][0], "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    return None


ENABLE_REDDIT = os.getenv("TRACER_ENABLE_REDDIT", "0").lower() in ("1", "true")

# Discovery runs every source concurrently; reputable outlets and fact-checkers get their own
# site-restricted searches and are prioritised by matching and evidence ranking.
SOURCES = {"outlets": outlets, "factcheck": factcheckers, "news": news, "gdelt": gdelt,
           "duckduckgo": duckduckgo, "bluesky": bluesky}
if ENABLE_REDDIT:
    SOURCES["reddit"] = reddit

CONTRA = {"news": ["fact check", "debunked false"],
          "gdelt": ['(debunked OR "fact check" OR retracted OR denied OR correction)'],
          "bluesky": ["debunked fact check"],
          "duckduckgo": ["debunked fact check"]}
if ENABLE_REDDIT:
    CONTRA["reddit"] = ["debunked fact check"]
