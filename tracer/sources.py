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
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

import httpx

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
    """Reddit search. With REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET (a free "script" app) it uses OAuth, which
    is not blocked; the anonymous JSON endpoint usually answers 403."""
    params = {"q": q, "sort": "relevance", "t": "all", "limit": 100}
    tok = await _reddit_token(c)
    if tok:
        r = await c.get("https://oauth.reddit.com/search", params=params,
                        headers={"User-Agent": UA, "Authorization": f"bearer {tok}"})
    else:
        r = await c.get("https://www.reddit.com/search.json", params=params, headers={"User-Agent": UA})
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
    """Google News restricted to fact-checking organisations."""
    res = await asyncio.gather(*(google_news(c, f"{q} {g}") for g in site_groups((FACT_CHECK,))),
                               return_exceptions=True)
    return [it for r in res if not isinstance(r, Exception) for it in r]


async def claimreview(c, q):
    """Google Fact Check Tools API (free key: GOOGLE_FACTCHECK_API_KEY): ClaimReview ratings published by
    fact-checkers worldwide, with the date the claim was made and who made it (direct origin evidence)."""
    key = os.getenv("GOOGLE_FACTCHECK_API_KEY")
    if not key:
        return []
    r = await c.get("https://factchecktools.googleapis.com/v1alpha1/claims:search",
                    params={"query": q[:200], "languageCode": "en", "pageSize": 25, "key": key})
    r.raise_for_status()
    out = []
    for cl in r.json().get("claims", []):
        claim_ts = parse_iso(cl.get("claimDate") or "")
        for rv in cl.get("claimReview", []):
            ts = parse_iso(rv.get("reviewDate") or "") or claim_ts
            if not rv.get("url") or not ts:
                continue
            pub = rv.get("publisher", {})
            site = (pub.get("site") or urlparse(rv["url"]).netloc).lower().removeprefix("www.")
            out.append(dict(source="news", url=rv["url"], author=pub.get("name") or site, ts=ts, engagement=0,
                            text=f"Fact check: {cl.get('text', '')}. Rating: {rv.get('textualRating', '')}. "
                                 f"{rv.get('title', '')}"[:500],
                            meta={"outlet_domain": site, "rating": rv.get("textualRating"),
                                  "claimant": cl.get("claimant"), "claim_date": cl.get("claimDate"),
                                  "engine": "claimreview"}))
    return out


# ------------------------------------------------------------------ keyed web search APIs
_REL_DATE = re.compile(r"(\d+)\s+(minute|hour|day|week|month|year)s?\s+ago", re.I)


def parse_loose_date(s):
    """ISO dates, 'Jan 5, 2024', '5 Jan 2024' or '3 days ago' -> aware datetime (None if unparseable)."""
    if not s:
        return None
    d = parse_iso(s)
    if d:
        return d
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%d %B %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s.strip(), fmt).replace(tzinfo=UTC)
        except ValueError:
            pass
    m = _REL_DATE.search(s)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        days = {"minute": 1 / 1440, "hour": 1 / 24, "day": 1, "week": 7, "month": 30, "year": 365}[unit] * n
        return datetime.now(UTC) - timedelta(days=days)
    return None


def _web_item(url, title, snippet, date, engine):
    ts = parse_loose_date(date)
    dom = urlparse(url).netloc.lower().removeprefix("www.")
    return dict(source="websearch", url=url, author=dom, ts=ts or datetime.now(UTC), engagement=0,
                text=f"{title}. {snippet or ''}".strip()[:500],
                meta={"engine": engine, "outlet_domain": dom, "undated": ts is None})


async def _brave(c, q, key):
    r = await c.get("https://api.search.brave.com/res/v1/web/search",
                    params={"q": q, "count": 20}, headers={"X-Subscription-Token": key, "Accept": "application/json"})
    r.raise_for_status()
    return [_web_item(x["url"], x.get("title", ""), re.sub(r"<[^>]+>", "", x.get("description", "")),
                      x.get("page_age") or x.get("age"), "brave")
            for x in (r.json().get("web") or {}).get("results", []) if x.get("url")]


async def _tavily(c, q, key):
    r = await c.post("https://api.tavily.com/search", headers={"Authorization": f"Bearer {key}"},
                     json={"query": q, "max_results": 20, "search_depth": "basic"})
    r.raise_for_status()
    return [_web_item(x["url"], x.get("title", ""), x.get("content", ""), x.get("published_date"), "tavily")
            for x in r.json().get("results", []) if x.get("url")]


async def _serper(c, q, key):
    out = []
    for kind in ("search", "news"):
        r = await c.post(f"https://google.serper.dev/{kind}", headers={"X-API-KEY": key},
                         json={"q": q, "num": 20})
        r.raise_for_status()
        for x in r.json().get("organic" if kind == "search" else "news", []):
            if x.get("link"):
                out.append(_web_item(x["link"], x.get("title", ""), x.get("snippet", ""), x.get("date"),
                                     f"serper_{kind}"))
    return out


WEBSEARCH_KEYS = (("BRAVE_API_KEY", _brave), ("TAVILY_API_KEY", _tavily), ("SERPER_API_KEY", _serper))


async def websearch(c, q):
    """First configured search API (Brave, Tavily or Serper, all with free tiers): dated, reliable web results,
    unlike scraped DuckDuckGo, which throttles and carries no dates."""
    for env, fn in WEBSEARCH_KEYS:
        key = os.getenv(env)
        if key:
            return await fn(c, q, key)
    return []


# ------------------------------------------------------------------ Reddit (OAuth app-only)
_reddit_tok = {"v": None, "exp": 0.0}


async def _reddit_token(c):
    cid, sec = os.getenv("REDDIT_CLIENT_ID"), os.getenv("REDDIT_CLIENT_SECRET")
    if not (cid and sec):
        return None
    if _reddit_tok["v"] and time.time() < _reddit_tok["exp"] - 60:
        return _reddit_tok["v"]
    r = await c.post("https://www.reddit.com/api/v1/access_token", auth=(cid, sec),
                     data={"grant_type": "client_credentials"}, headers={"User-Agent": UA})
    r.raise_for_status()
    js = r.json()
    _reddit_tok.update(v=js.get("access_token"), exp=time.time() + float(js.get("expires_in", 3600)))
    return _reddit_tok["v"]


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


_ATODAY_FIRST = re.compile(r'rel="first memento"; datetime="([^"]+)"')
ARCHIVE_TODAY = os.getenv("TRACER_ARCHIVE_TODAY", "1") == "1"


async def archive_today_first(c, url):
    """First archive.today (archive.ph) snapshot of a URL from its Memento TimeMap, or None. Captures pages the
    Wayback Machine misses (paywalled news, deleted social posts people archived as proof)."""
    if not ARCHIVE_TODAY:
        return None
    # plain HTTP: archive.today's TLS endpoints time out for non-browser clients; the TimeMap is public metadata
    r = await c.get(f"http://archive.ph/timemap/{url}", headers={"User-Agent": BROWSER_UA},
                    timeout=httpx.Timeout(15.0, connect=8.0))
    if r.status_code != 200:
        return None
    m = _ATODAY_FIRST.search(r.text)
    if not m:
        return None
    try:
        return parsedate_to_datetime(m.group(1)).astimezone(UTC)
    except (TypeError, ValueError):
        return None


async def first_capture(c, url):
    """Earliest archive capture across the Wayback Machine and archive.today -> (datetime, archive) or None."""
    res = await asyncio.gather(wayback_first(c, url), archive_today_first(c, url), return_exceptions=True)
    hits = [(d, name) for d, name in zip(res, ("wayback", "archive.today")) if isinstance(d, datetime)]
    return min(hits) if hits else None


# ------------------------------------------------------------------ recency + extra free sources (no keys)
RECENT_DAYS = int(os.getenv("TRACER_RECENT_DAYS", "14"))


def _bing_items(content):
    """Bing News RSS items; the real article URL is the `url` parameter of Bing's click-tracking link."""
    from urllib.parse import parse_qs
    root = ET.fromstring(content)
    ns = "{http://www.bing.com/schema/news}"
    for it in root.iter("item"):
        link = (it.findtext("link") or "").strip()
        real = (parse_qs(urlparse(link).query).get("url") or [link])[0]
        try:
            ts = parsedate_to_datetime(it.findtext("pubDate"))
            ts = ts if ts.tzinfo else ts.replace(tzinfo=UTC)
        except Exception:
            continue
        title = html.unescape((it.findtext("title") or "").strip())
        desc = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(it.findtext("description") or ""))).strip()
        dom = urlparse(real).netloc.lower().removeprefix("www.")
        name = (it.findtext(ns + "Source") or dom).strip()
        text = title if desc.startswith(title[:40]) else f"{title}. {desc}"
        yield dict(source="news", url=real, author=name, ts=ts, text=text[:500], engagement=0,
                   meta={"outlet_domain": dom, "engine": "bing_news"})


async def bing_news(c, q):
    """Bing News RSS sorted newest first (free, no key)."""
    r = await c.get("https://www.bing.com/news/search", headers={"User-Agent": BROWSER_UA},
                    params={"q": q, "format": "rss", "qft": 'sortbydate="1"', "setlang": "en"})
    r.raise_for_status()
    return list(_bing_items(r.content))


async def recent_news(c, q):
    """The latest coverage: Google News limited to the last TRACER_RECENT_DAYS days + Bing News by date.
    Breaking claims ("X retires", "Y has died") are settled by the newest reports, which relevance-sorted
    searches often bury under older stories."""
    res = await asyncio.gather(google_news(c, f"{q} when:{RECENT_DAYS}d"), bing_news(c, q), return_exceptions=True)
    out = []
    for r in res:
        if isinstance(r, Exception):
            logger.warning("recent_news: %s: %s", type(r).__name__, r)
            continue
        for it in r:
            it["meta"]["recent_search"] = True
            out.append(it)
    return out


async def hackernews(c, q):
    """Hacker News via the free Algolia API (tech / science claims; links to the original source)."""
    r = await c.get("https://hn.algolia.com/api/v1/search", params={"query": q, "tags": "story", "hitsPerPage": 40})
    r.raise_for_status()
    out = []
    for h in r.json().get("hits", []):
        ts = parse_iso(h.get("created_at") or "")
        if not ts:
            continue
        item = f"https://news.ycombinator.com/item?id={h['objectID']}"
        links = [h["url"]] if h.get("url") else []
        text = (h.get("title") or "") + (". " + re.sub(r"<[^>]+>", " ", h["story_text"])[:300] if h.get("story_text") else "")
        out.append(dict(source="hackernews", url=item, author=h.get("author", ""), ts=ts, text=text[:500],
                        engagement=int(h.get("points") or 0) + int(h.get("num_comments") or 0),
                        meta={"links": links}))
    return out


MASTODON_INSTANCE = os.getenv("MASTODON_INSTANCE", "https://mastodon.social").rstrip("/")


async def mastodon(c, tag):
    """Public Mastodon hashtag timeline (no key). `tag` is a single word such as a name in the claim."""
    tag = re.sub(r"[^0-9A-Za-z_]", "", tag)
    if len(tag) < 3:
        return []
    r = await c.get(f"{MASTODON_INSTANCE}/api/v1/timelines/tag/{tag}", params={"limit": 40})
    r.raise_for_status()
    out = []
    for p in r.json():
        ts = parse_iso(p.get("created_at") or "")
        if not ts or not p.get("url"):
            continue
        text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", p.get("content") or ""))).strip()
        links = [m.get("url") for m in [p.get("card") or {}] if m.get("url")]
        acct = (p.get("account") or {}).get("acct", "")
        out.append(dict(source="mastodon", url=p["url"], author=acct, ts=ts, text=text[:500],
                        engagement=int(p.get("favourites_count") or 0) + 2 * int(p.get("reblogs_count") or 0)
                        + int(p.get("replies_count") or 0), meta={"links": links}))
    return out


ENABLE_REDDIT = (os.getenv("TRACER_ENABLE_REDDIT", "0").lower() in ("1", "true")
                 or bool(os.getenv("REDDIT_CLIENT_ID") and os.getenv("REDDIT_CLIENT_SECRET")))

# Discovery runs every source concurrently; reputable outlets and fact-checkers get their own
# site-restricted searches and are prioritised by matching and evidence ranking.
SOURCES = {"recent": recent_news, "hackernews": hackernews, "mastodon": mastodon,
           "outlets": outlets, "factcheck": factcheckers, "news": news, "gdelt": gdelt,
           "duckduckgo": duckduckgo, "bluesky": bluesky}
if ENABLE_REDDIT:
    SOURCES["reddit"] = reddit
if os.getenv("GOOGLE_FACTCHECK_API_KEY"):
    SOURCES["claimreview"] = claimreview
if any(os.getenv(k) for k, _ in WEBSEARCH_KEYS):
    SOURCES["websearch"] = websearch

CONTRA = {"news": ["fact check", "debunked false"],
          "gdelt": ['(debunked OR "fact check" OR retracted OR denied OR correction)'],
          "bluesky": ["debunked fact check"],
          "duckduckgo": ["debunked fact check"]}
if ENABLE_REDDIT:
    CONTRA["reddit"] = ["debunked fact check"]
