"""Data-source clients: GDELT, Reddit, Bluesky, RSS/Google News, Wikipedia, Wayback CDX."""
import html
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

from .classify import UTC, parse_iso
from .config import DEFAULT_FEEDS, UA


async def gdelt(c, q):
    r = await c.get("https://api.gdeltproject.org/api/v2/doc/doc", params={
        "query": q, "mode": "artlist", "format": "json", "maxrecords": 75,
        "sort": "DateAsc", "timespan": "3months"})
    out = []
    for a in r.json().get("articles", []):
        ts = datetime.strptime(a["seendate"], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        out.append(dict(source="gdelt", url=a["url"], author=a.get("domain", ""), ts=ts,
                        text=a.get("title", ""), engagement=0,
                        meta={"lang": a.get("language"), "country": a.get("sourcecountry")}))
    return out


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
        yield dict(source="news", url=(it.findtext("link") or "").strip(),
                   author=(src.text if src is not None else default_author) or default_author,
                   ts=ts, text=f"{title}. {desc}".strip()[:500], engagement=0, meta={})


async def news(c, q):
    out = []
    r = await c.get("https://news.google.com/rss/search",
                    params={"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"})
    out += list(_rss_items(r.content))
    toks = [t.lower() for t in re.findall(r"\w{4,}", q)]
    need = min(2, len(toks))
    for f in os.getenv("TRACER_FEEDS", ",".join(DEFAULT_FEEDS)).split(","):
        try:
            rr = await c.get(f.strip())
            for it in _rss_items(rr.content, urlparse(f).netloc):
                if sum(t in it["text"].lower() for t in toks) >= need:
                    out.append(it)
        except Exception:
            pass
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

SOURCES = {"gdelt": gdelt, "bluesky": bluesky, "news": news}
if ENABLE_REDDIT:
    SOURCES["reddit"] = reddit

CONTRA = {"news": ["fact check", "debunked false", "correction retracted denied"],
          "gdelt": ['(debunked OR "fact check" OR retracted OR denied OR correction)'],
          "bluesky": ["debunked fact check"]}
if ENABLE_REDDIT:
    CONTRA["reddit"] = ["debunked fact check"]
