"""Evidence record and the shared mutable run state passed between agents."""
import sys
from dataclasses import dataclass, field
from datetime import datetime

from .classify import SOURCE_WEIGHTS, UTC, classify_url, urls_in


@dataclass
class Ev:
    id: str
    source: str  # gdelt | reddit | bluesky | news
    url: str
    author: str
    ts: datetime
    text: str
    engagement: int = 0
    meta: dict = field(default_factory=dict)
    first_capture: datetime | None = None  # Wayback first capture

    @property
    def eff_ts(self):
        return min(self.ts, self.first_capture) if self.first_capture else self.ts

    def brief(self, n=280):
        return {"id": self.id, "t": self.eff_ts.strftime("%Y-%m-%d %H:%M"), "src": self.source,
                "by": self.author, "text": self.text[:n].replace("\n", " "), "eng": self.engagement}


class State:
    """Everything one run knows. Agents read and mutate it; the pipeline passes it along."""

    def __init__(self, claim, primary_urls=(), on_event=None):
        self.claim = claim
        self.pool: dict[str, Ev] = {}
        self.n = 0
        self.plan: dict = {}
        self.base_query = ""  # round-1 keyword query, reused by official-domain / SEC lookups
        self.paraphrases: list[str] = []
        self.orgs: list[str] = []
        self.official_domains: list[str] = []
        self.sec_relevant = False
        self.follow_up = None
        self.discovery: dict = {}
        self.variants: list[dict] = []
        self.assign: dict = {}
        self.origin = None
        self.timeline: list[dict] = []
        self.amplifiers: list[dict] = []
        self.refs: list[dict] = []
        self.chains: list[dict] = []
        self.cite_edges: list[tuple] = []
        self.fetched: set = set()
        self.archive_tried: set = set()
        self.queries_used: set = set()
        self.user_primary = list(primary_urls)
        self.fact: dict = {}
        self.critiques: list[dict] = []
        self.judge: dict = {}
        self.evidence_pack: list[dict] = []  # ranked evidence sent to the consolidated analysis
        self.result: dict = {}
        self.events: list[dict] = []
        self.on_event = on_event

    def emit(self, agent, msg):
        rec = {"ts": datetime.now(UTC).isoformat(), "agent": agent, "message": msg}
        self.events.append(rec)
        print(f"[{agent}] {msg}", file=sys.stderr, flush=True)
        if self.on_event:
            try:
                self.on_event(agent, msg)
            except Exception:
                pass

    def add(self, raw):
        if raw["url"] and raw["url"] not in self.pool:
            self.n += 1
            e = Ev(id=f"E{self.n}", **raw)
            e.meta["links"] = list(dict.fromkeys(urls_in(e.text) + e.meta.get("links", [])))
            self.pool[raw["url"]] = e

    def sorted(self):
        return sorted(self.pool.values(), key=lambda e: e.eff_ts)

    def by_id(self):
        return {e.id: e for e in self.pool.values()}

    def add_ref(self, kind, url, title, text, typ=None, **extra):
        if any(r["url"] == url for r in self.refs):
            return
        typ = typ or classify_url(url)
        self.refs.append({"kind": kind, "url": url, "title": title, "text": text, "type": typ,
                          "weight": SOURCE_WEIGHTS.get(typ, 0.4), **extra})
