"""Image provenance: where else has this picture appeared, and since when?

    pip install pillow imagehash          (OCR also needs: pip install pytesseract + the tesseract binary)

For an image URL or local file this module reports:
  * fingerprint   SHA-256 and a perceptual hash (pHash), which survives resizing and recompression
  * metadata      EXIF capture date, camera, editing software, GPS presence (social sites strip EXIF, so a
                  missing EXIF block is normal and says nothing by itself)
  * OCR text      words in the image (memes, screenshots), if pytesseract + tesseract are installed
  * reverse search
      SERPAPI_API_KEY   Google Lens visual matches (via SerpAPI, free tier), then the earliest archive capture
                        (Wayback / archive.today) of each matching page as its first-seen date
      TINEYE_API_KEY    TinEye matches with TinEye's own crawl dates, oldest first
    Both need a public image URL. Without a key, ready-made manual search links are returned.
  * near-duplicates matched thumbnails are pHash-compared: distance <= 8 is the same picture, 9-22 a crop /
                    edit of it ("variant"), more is a different picture that merely looks alike.
The key misinformation signal is an image whose earliest appearance predates the event it is said to show.
"""
import asyncio
import hashlib
import io
import logging
import os
from datetime import datetime
from urllib.parse import quote, urlparse

from .classify import UTC, parse_iso
from .config import BROWSER_UA, UA

logger = logging.getLogger(__name__)

MAX_BYTES = 15 * 1024 * 1024
SAME_MAX, VARIANT_MAX = 8, 22  # pHash Hamming distance bands (64-bit hash)
EDITORS = ("photoshop", "gimp", "lightroom", "snapseed", "picsart", "canva", "facetune", "affinity", "pixlr")


def _libs():
    try:
        from PIL import Image, ExifTags
        import imagehash
        return Image, ExifTags, imagehash
    except ImportError:
        return None


async def load_bytes(c, src):
    if src.startswith(("http://", "https://")):
        r = await c.get(src, headers={"User-Agent": BROWSER_UA}, timeout=30.0)
        if r.status_code == 403:  # e.g. Wikimedia refuses browser-imitating scripts; it wants a descriptive UA
            r = await c.get(src, headers={"User-Agent": UA}, timeout=30.0)
        r.raise_for_status()
        if len(r.content) > MAX_BYTES:
            raise ValueError("image larger than 15 MB")
        return r.content
    with open(src, "rb") as f:
        data = f.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("image larger than 15 MB")
    return data


def _exif(img, ExifTags):
    out, flags = {}, []
    try:
        raw = img.getexif()
    except Exception:
        raw = {}
    if not raw:
        flags.append("no EXIF metadata (normal for images re-shared via social media or messaging apps)")
        return out, flags
    tags = {ExifTags.TAGS.get(k, k): v for k, v in raw.items()}
    try:
        tags.update({ExifTags.TAGS.get(k, k): v for k, v in raw.get_ifd(0x8769).items()})  # Exif sub-IFD
    except Exception:
        pass
    for k in ("DateTimeOriginal", "DateTime", "Make", "Model", "Software", "Artist", "Copyright"):
        if tags.get(k):
            out[k] = str(tags[k]).strip("\x00 ")[:120]
    try:
        out["has_gps"] = bool(raw.get_ifd(0x8825))
    except Exception:
        out["has_gps"] = False
    sw = out.get("Software", "").lower()
    if any(e in sw for e in EDITORS):
        flags.append(f"EXIF says it was saved by editing software ({out['Software']})")
    if out.get("DateTimeOriginal"):
        flags.append(f"EXIF capture time {out['DateTimeOriginal']} (camera clock; can be wrong or edited)")
    if out.get("has_gps"):
        flags.append("EXIF contains GPS coordinates (location can be checked against the claim)")
    return out, flags


def exif_date(exif):
    s = exif.get("DateTimeOriginal") or ""
    try:
        return datetime.strptime(s[:19], "%Y:%m:%d %H:%M:%S").replace(tzinfo=UTC)
    except ValueError:
        return None


def _ocr(img):
    try:
        import pytesseract
        txt = pytesseract.image_to_string(img.convert("RGB"))
        return " ".join(txt.split())[:1000] or None
    except Exception:
        return None  # pytesseract or the tesseract binary not installed


def fingerprint(data):
    """(info dict, PIL image, pHash) for raw image bytes."""
    libs = _libs()
    info = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    if not libs:
        info["notes"] = ["pillow / imagehash not installed: pip install pillow imagehash"]
        return info, None, None
    Image, ExifTags, imagehash = libs
    img = Image.open(io.BytesIO(data))
    img.load()
    ph = imagehash.phash(img)
    exif, flags = _exif(img, ExifTags)
    info.update(format=img.format, width=img.width, height=img.height, phash=str(ph), exif=exif, flags=flags)
    return info, img, ph


def manual_links(url):
    if not url or not url.startswith("http"):
        return {"note": "reverse search needs a public image URL; upload the file manually to the engines below",
                "google_lens": "https://lens.google.com/", "tineye": "https://tineye.com/",
                "yandex": "https://yandex.com/images/"}
    u = quote(url, safe="")
    return {"google_lens": f"https://lens.google.com/uploadbyurl?url={u}",
            "tineye": f"https://tineye.com/search?url={u}",
            "yandex": f"https://yandex.com/images/search?rpt=imageview&url={u}",
            "bing": f"https://www.bing.com/images/search?view=detailv2&iss=sbi&q=imgurl:{u}"}


async def _lens(c, url, key):
    r = await c.get("https://serpapi.com/search.json", params={"engine": "google_lens", "url": url, "api_key": key},
                    timeout=60.0)
    r.raise_for_status()
    return [{"url": m.get("link"), "title": m.get("title", ""), "site": m.get("source") or urlparse(m.get("link", "")).netloc,
             "thumbnail": m.get("thumbnail"), "first_seen": None, "engine": "google_lens"}
            for m in r.json().get("visual_matches", [])[:20] if m.get("link")]


async def _tineye(c, url, key):
    r = await c.get("https://api.tineye.com/rest/search/", headers={"x-api-key": key},
                    params={"image_url": url, "sort": "crawl_date", "order": "asc", "limit": 50}, timeout=60.0)
    r.raise_for_status()
    out = []
    for m in (r.json().get("results") or {}).get("matches", []):
        for b in m.get("backlinks", [])[:3]:
            out.append({"url": b.get("backlink"), "title": "", "site": m.get("domain", ""),
                        "thumbnail": m.get("image_url"), "first_seen": b.get("crawl_date"), "engine": "tineye"})
    return [x for x in out if x["url"]]


async def _thumb_distance(c, thumb, ph):
    libs = _libs()
    if not (thumb and ph is not None and libs):
        return None
    Image, _, imagehash = libs
    try:
        r = await c.get(thumb, headers={"User-Agent": BROWSER_UA}, timeout=15.0)
        r.raise_for_status()
        return int(ph - imagehash.phash(Image.open(io.BytesIO(r.content))))
    except Exception:
        return None


def relation(d):
    return None if d is None else "same" if d <= SAME_MAX else "variant" if d <= VARIANT_MAX else "different"


async def reverse_search(c, url, ph):
    """Reverse image search with whichever key is configured; dates every match. Returns (engine, matches)."""
    from . import sources  # archive first-capture lookups
    engine, matches = None, []
    for env, fn, name in (("TINEYE_API_KEY", _tineye, "tineye"), ("SERPAPI_API_KEY", _lens, "google_lens")):
        key = os.getenv(env)
        if key and url and url.startswith("http"):
            try:
                matches, engine = await fn(c, url, key), name
                break
            except Exception as e:
                logger.warning("reverse image search %s failed: %s", name, e)
    if not matches:
        return engine, []
    sem = asyncio.Semaphore(5)

    async def enrich(m):
        async with sem:
            m["phash_distance"] = await _thumb_distance(c, m.pop("thumbnail", None), ph)
            m["relation"] = relation(m["phash_distance"])
            if not m["first_seen"]:
                try:
                    hit = await sources.first_capture(c, m["url"])
                    m["first_seen"], m["first_seen_by"] = (hit[0].isoformat(), hit[1]) if hit else (None, None)
                except Exception:
                    pass
            else:
                m["first_seen_by"] = engine

    await asyncio.gather(*(enrich(m) for m in matches[:20]))
    matches.sort(key=lambda m: m.get("first_seen") or "9999")
    return engine, matches


async def analyze_image(c, src, claim=None):
    """Full provenance report for one image (URL or local path)."""
    out = {"source": src, "manual_search": manual_links(src)}
    try:
        data = await load_bytes(c, src)
    except Exception as e:
        out["error"] = f"could not load image: {type(e).__name__}: {e}"
        return out
    info, img, ph = fingerprint(data)
    out.update(info)
    if img is not None:
        out["ocr_text"] = _ocr(img)
    engine, matches = await reverse_search(c, src, ph)
    out["reverse_engine"] = engine
    out["matches"] = matches
    dated = [m for m in matches if m.get("first_seen") and m.get("relation") != "different"]
    notes = list(out.get("flags", []))
    if dated:
        e = dated[0]
        out["earliest"] = {"url": e["url"], "site": e["site"], "first_seen": e["first_seen"],
                           "relation": e.get("relation")}
        notes.append(f"Earliest known appearance: {e['first_seen'][:10]} on {e['site']} ({e.get('relation') or 'unverified'} image)")
    elif not engine:
        notes.append("No reverse-search key (TINEYE_API_KEY or SERPAPI_API_KEY): use the manual search links")
    variants = [m for m in matches if m.get("relation") == "variant"]
    if variants:
        notes.append(f"{len(variants)} near-duplicate(s) differ from this image (cropped or edited versions exist)")
    ed = exif_date(out.get("exif") or {})
    if ed:
        out["exif_date"] = ed.isoformat()
    out["notes"] = notes
    return out


def earliest_date(img_report):
    """Earliest date this image is known to have existed (reverse-search first-seen or EXIF capture)."""
    ds = [parse_iso((img_report.get("earliest") or {}).get("first_seen") or ""), parse_iso(img_report.get("exif_date") or "")]
    ds = [d for d in ds if d]
    return min(ds) if ds else None
