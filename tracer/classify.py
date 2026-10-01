"""Source-type classification, weights and small URL/time helpers."""
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

UTC = timezone.utc

# ------------------------------------------------------- source types & weights
# news weight deliberately below PRIMARY_MIN_W (0.75) so news items never count
# as primary sources and the confidence cap (line 42 scoring.py) always applies when
# no court/government/peer-reviewed source is cited.  Was 0.75 – that equalled the
# threshold, causing primary_sources: 8 and suppressing the cap.
SOURCE_WEIGHTS = {"court": 1.0, "government": 0.95, "dataset": 0.95, "peer_reviewed": 0.9,
                  "press_release": 0.75, "news": 0.5, "institutional": 0.7, "preprint": 0.6,
                  "reference": 0.5, "web": 0.4, "social": 0.2}
TERMINAL = {"court", "government", "dataset", "peer_reviewed", "press_release", "institutional"}
PRIMARY_MIN_W = 0.75  # >= this counts as a primary-type source in confidence stats

SOCIAL = ("reddit.com", "bsky.app", "twitter.com", "x.com", "facebook.com", "instagram.com",
          "tiktok.com", "youtube.com", "youtu.be", "t.me", "threads.net")
NEWS = ("news.google.com", "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
        "cnn.com", "nytimes.com", "theguardian.com", "washingtonpost.com",
        "aljazeera.com", "ndtv.com", "indiatoday.in", "thehindu.com",
        "timesofindia.indiatimes.com", "cbsnews.com", "nbcnews.com", "foxnews.com",
        "bloomberg.com", "wsj.com", "afp.com", "agenzianova.com", "jpost.com")
COURT = ("courtlistener.com", "supremecourt.gov", "uscourts.gov", "pacer.gov", "law.cornell.edu",
         "justia.com", "casetext.com", "curia.europa.eu", "echr.coe.int", "bailii.org")
DATASET = ("data.gov", "kaggle.com", "zenodo.org", "ourworldindata.org", "census.gov",
           "data.worldbank.org", "figshare.com", "dataverse.harvard.edu", "fred.stlouisfed.org",
           "ec.europa.eu/eurostat", "data.un.org")
PAPER = ("doi.org", "nature.com", "science.org", "thelancet.com", "nejm.org", "pubmed.ncbi.nlm.nih.gov",
         "ncbi.nlm.nih.gov", "sciencedirect.com", "springer.com", "wiley.com", "jstor.org", "cell.com",
         "bmj.com", "pnas.org", "plos.org")
PREPRINT = ("arxiv.org", "biorxiv.org", "medrxiv.org", "ssrn.com")
PRESS = ("prnewswire.com", "businesswire.com", "globenewswire.com", "accesswire.com")
GOV_RE = re.compile(r"((^|\.)gov(\.[a-z]{2})?|\.mil|\.int|\.gc\.ca|\.europa\.eu|(^|\.)un\.org)$")


def host_in(h, lst):
    return any(h == d or h.endswith("." + d) or (("/" in d) and d in h) for d in lst)


def classify_url(u):
    p = urlparse(u)
    h, path = (p.netloc or "").lower().removeprefix("www."), (p.path or "").lower()
    if host_in(h, SOCIAL):
        return "social"
    if host_in(h, NEWS) or "news" in h:
        return "news"
    if host_in(h, COURT) or "/docket" in path:
        return "court"
    if GOV_RE.search(h):
        return "government"
    if host_in(h, DATASET) or "/dataset" in path:
        return "dataset"
    if host_in(h, PAPER) or "/doi/" in path:
        return "peer_reviewed"
    if host_in(h, PREPRINT):
        return "preprint"
    if host_in(h, PRESS) or any(s in path for s in ("/press-release", "/newsroom", "/media/press")):
        return "press_release"
    if h.endswith(".edu") or h.endswith(".ac.uk"):
        return "institutional"
    if h.endswith("wikipedia.org"):
        return "reference"
    return "web"



def parse_iso(s):
    try:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=UTC)
    except Exception:
        return None


def urls_in(text):
    return [u.rstrip(".,);]") for u in re.findall(r"https?://[^\s<>\"']+", text or "")]


def norm_url(u):
    p = urlparse(u)
    return p.netloc.lower().removeprefix("www.") + p.path.rstrip("/")
