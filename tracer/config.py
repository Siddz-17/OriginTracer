"""Environment-driven configuration."""
import os

for _b in [os.getcwd(), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]:
    _p = os.path.join(_b, ".env")
    if os.path.exists(_p):
        for _l in open(_p, encoding="utf-8"):
            _l = _l.split("#", 1)[0].strip()
            if "=" in _l:
                _k, _v = _l.split("=", 1)
                if _v.strip():
                    os.environ.setdefault(_k.strip(), _v.strip())
        break

CONTACT = os.getenv("TRACER_CONTACT", "tracer@example.com")  # SEC/Wikipedia ask for a contact in the UA
UA = f"NarrativeOriginTracer/0.3 (research tool; {CONTACT})"
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///tracer.db")  # postgresql+psycopg://user:pw@host/db
USE_PLAYWRIGHT = os.getenv("USE_PLAYWRIGHT", "0") == "1"  # JS rendering + Google News redirect resolution
MAX_CONCURRENT_RUNS = int(os.getenv("MAX_CONCURRENT_RUNS", "2"))

# --- LLM: Gemini and Groq only -------------------------------------------------
# Each tier = (provider, model). Override with <TIER>_PROVIDER / <TIER>_MODEL env vars.
#   fast       query expansion, claim matcher          (small/medium prompts)
#   smart      fact checker, judge                     (big evidence prompts -> needs big context)
#   adversary  devil's advocate                        (default: other model family than 'smart')
# Groq's free tier is ~6k tokens/min, so the large-prompt tiers default to Gemini. Model IDs change often:
# check your account's list (GET https://api.groq.com/openai/v1/models) and override if a default 404s.
_DEFAULTS = {
    "fast": ("gemini", "gemini-3.8-flash"),
    "smart": ("gemini", "gemini-3.8-flash"),
    "adversary": ("groq", "openai/gpt-oss-120b"),
}
PROVIDERS = ("gemini", "groq")
TIERS = {}
for _t, (_p, _m) in _DEFAULTS.items():
    _prov = os.getenv(f"{_t.upper()}_PROVIDER", _p).lower()
    if _prov not in PROVIDERS:
        raise ValueError(f"{_t.upper()}_PROVIDER must be one of {PROVIDERS}, got {_prov!r}")
    _model = os.getenv(f"{_t.upper()}_MODEL") or (_m if _prov == _p else
             {"gemini": "gemini-3.8-flash", "groq": "openai/gpt-oss-120b"}[_prov])
    TIERS[_t] = (_prov, _model)
GROQ_MAX_INPUT_CHARS = int(os.getenv("GROQ_MAX_INPUT_CHARS", "14000"))  # bigger prompts go to Gemini instead
LLM_FALLBACK = os.getenv("LLM_FALLBACK", "1") == "1"                      # on repeated failure, try the other provider

DEFAULT_FEEDS = [
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.theguardian.com/world/rss",
    "https://feeds.npr.org/1001/rss.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
    "https://feeds.feedburner.com/ndtvnews-world-news",
    "https://www.france24.com/en/rss",
]
