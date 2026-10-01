"""Reputable-outlet registry. Discovery queries these first; evidence ranking boosts them.

Tiers (higher = searched first, ranked higher):
  fact_check  IFCN-style fact-checkers and the fact-check desks of wire services
  wire        news agencies (original reporting most outlets copy)
  major       large national / international newsrooms with corrections policies
"""
from urllib.parse import urlparse

FACT_CHECK = ("snopes.com", "politifact.com", "factcheck.org", "fullfact.org", "leadstories.com",
              "factcheck.afp.com", "checkyourfact.com", "altnews.in", "boomlive.in", "factly.in",
              "vishvasnews.com", "newschecker.in", "thequint.com/news/webqoof", "logically.ai",
              "healthfeedback.org", "climatefeedback.org", "reuters.com/fact-check", "apnews.com/hub/ap-fact-check",
              "usatoday.com/news/factcheck", "bbc.co.uk/news/reality_check", "africacheck.org", "maldita.es")
WIRE = ("reuters.com", "apnews.com", "afp.com", "bloomberg.com", "pti.in", "ani.in", "upi.com")
MAJOR = ("bbc.com", "bbc.co.uk", "nytimes.com", "washingtonpost.com", "theguardian.com", "wsj.com", "ft.com",
         "npr.org", "cnn.com", "cbsnews.com", "nbcnews.com", "abcnews.go.com", "aljazeera.com", "economist.com",
         "axios.com", "politico.com", "theatlantic.com", "time.com", "usatoday.com", "latimes.com",
         "cnbc.com", "pbs.org", "dw.com", "france24.com", "lemonde.fr", "theverge.com", "techcrunch.com",
         "arstechnica.com", "wired.com", "nature.com", "science.org", "statnews.com",
         "thehindu.com", "indianexpress.com", "hindustantimes.com", "ndtv.com", "timesofindia.indiatimes.com",
         "indiatoday.in", "livemint.com", "scroll.in", "theprint.in", "abc.net.au", "cbc.ca", "smh.com.au",
         "independent.co.uk", "telegraph.co.uk", "skynews.com", "japantimes.co.jp", "scmp.com")

TIER_SCORE = {"fact_check": 1.0, "wire": 0.9, "major": 0.75}


def _match(host, path, lst):
    for d in lst:
        dom, _, sub = d.partition("/")
        if (host == dom or host.endswith("." + dom)) and (not sub or path.startswith("/" + sub)):
            return True
    return False


def outlet_tier(url_or_domain):
    """Return 'fact_check' | 'wire' | 'major' | None for a URL or bare domain."""
    s = url_or_domain if "://" in (url_or_domain or "") else f"https://{url_or_domain or ''}"
    p = urlparse(s)
    host, path = (p.netloc or "").lower().removeprefix("www."), (p.path or "").lower()
    if not host:
        return None
    if _match(host, path, FACT_CHECK) or "fact-check" in path or "factcheck" in path:
        return "fact_check"
    if _match(host, path, WIRE):
        return "wire"
    if _match(host, path, MAJOR):
        return "major"
    return None


def site_groups(tier_lists=(FACT_CHECK, WIRE, MAJOR), size=8):
    """Chunk outlet domains into `(site:a OR site:b ...)` filters short enough for Google News search."""
    groups = []
    for lst in tier_lists:
        doms = sorted({d.split("/")[0] for d in lst})
        for i in range(0, len(doms), size):
            groups.append("(" + " OR ".join(f"site:{d}" for d in doms[i:i + size]) + ")")
    return groups
