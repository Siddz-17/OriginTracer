"""Environment-driven configuration."""
import os
import re

for _b in [os.getcwd(), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]:
    _p = os.path.join(_b, ".env")
    if os.path.exists(_p):
        for _l in open(_p, encoding="utf-8-sig"):
            _l = _l.strip()
            if not _l or _l.startswith("#") or "=" not in _l:
                continue
            _k, _v = _l.split("=", 1)
            _v = _v.strip()
            if _v[:1] in "\"'" and _v[:1] and _v.count(_v[0]) >= 2:  # quoted value: keep '#' inside quotes
                _v = _v[1:_v.index(_v[0], 1)]
            else:  # inline comment needs whitespace before '#', so passwords containing '#' survive
                _v = re.split(r"\s#", _v, maxsplit=1)[0].strip()
            if _v:
                os.environ.setdefault(_k.strip().removeprefix("export "), _v)
        break

CONTACT = os.getenv("TRACER_CONTACT", "tracer@example.com")  # SEC/Wikipedia ask for a contact in the UA
UA = f"NarrativeOriginTracer/0.3 (research tool; {CONTACT})"
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///tracer.db")  # postgresql+psycopg://user:pw@host/db
USE_PLAYWRIGHT = os.getenv("USE_PLAYWRIGHT", "0") == "1"  # JS rendering + Google News redirect resolution; keep 0 (launches Chromium per thin page)
MAX_CONCURRENT_RUNS = int(os.getenv("MAX_CONCURRENT_RUNS", "2"))

# Browser-like UA for news pages / RSS (many publishers 403 unknown bots); UA above is for APIs.
BROWSER_UA = os.getenv("BROWSER_UA", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                     "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36")

# Pipeline mode:
#   single  (default) gather + rank all evidence with zero LLM calls, then ONE consolidated LLM analysis
#   agents  legacy 8-agent loop (6+ LLM calls per round; burns free-tier rate limits quickly)
MODE = os.getenv("TRACER_MODE", "single").lower()

# --- LLM: OpenRouter (default), Groq, Gemini ----------------------------------------
# Each tier = (provider, model). Override with <TIER>_PROVIDER / <TIER>_MODEL env vars.
#   smart      consolidated analysis (single mode); fact checker + judge (agents mode)
#   fast       query expansion + claim matcher (agents mode only)
#   adversary  devil's advocate (agents mode only)
# OpenRouter free models are rate limited (~20 req/min, 50 req/day without credits) and come and go,
# so a call walks OPENROUTER_MODELS in order, moving to the next model on 404 / 429 / empty output.
# Current list: https://openrouter.ai/models?q=free  (python check_setup.py prints the live list)
OPENROUTER_MODELS = [m.strip() for m in os.getenv(
    "OPENROUTER_MODELS",
    "nvidia/nemotron-3-super-120b-a12b:free,google/gemma-4-31b-it:free,qwen/qwen3.8-27b:free,"
    "nvidia/nemotron-3-ultra-550b-a55b:free,google/gemma-4-26b-a4b-it:free").split(",") if m.strip()]
_DEFAULT_MODEL = {"gemini": "gemini-2.5-flash", "groq": "openai/gpt-oss-120b",
                  "openrouter": OPENROUTER_MODELS[0] if OPENROUTER_MODELS else "openrouter/auto"}
_DEFAULTS = {"fast": "openrouter", "smart": "openrouter", "adversary": "openrouter"}
PROVIDERS = ("openrouter", "groq", "gemini")
TIERS = {}
for _t, _p in _DEFAULTS.items():
    _prov = os.getenv(f"{_t.upper()}_PROVIDER", _p).lower()
    if _prov not in PROVIDERS:
        raise ValueError(f"{_t.upper()}_PROVIDER must be one of {PROVIDERS}, got {_prov!r}")
    TIERS[_t] = (_prov, os.getenv(f"{_t.upper()}_MODEL") or _DEFAULT_MODEL[_prov])
GROQ_MAX_INPUT_CHARS = int(os.getenv("GROQ_MAX_INPUT_CHARS", "10000"))  # bigger prompts skip Groq
LLM_FALLBACK = os.getenv("LLM_FALLBACK", "1") == "1"  # on repeated failure, try the next provider
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "180"))  # free reasoning models can be slow on big prompts

# Evidence budget for the single analysis prompt (characters, ~4 chars/token).
ANALYSIS_MAX_CHARS = int(os.getenv("ANALYSIS_MAX_CHARS", "10000"))
ANALYSIS_MAX_ITEMS = int(os.getenv("ANALYSIS_MAX_ITEMS", "20"))

DEFAULT_FEEDS = [
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.theguardian.com/world/rss",
    "https://feeds.npr.org/1001/rss.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
    "https://feeds.feedburner.com/ndtvnews-world-news",
    "https://www.france24.com/en/rss",
]
