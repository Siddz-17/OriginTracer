"""Offline end-to-end smoke test: fake LLM + fake sources + fake HTTP. Run: python tests/test_smoke.py"""
import asyncio
import json
import os
import sys
import types
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    import httpx  # noqa
except ImportError:  # stub just enough for tracer.pipeline to import
    sys.modules["httpx"] = types.ModuleType("httpx")
    import httpx

from tracer import agents, llm, pipeline, scrape, sources  # noqa: E402

_REAL_ASK = llm.ask_json  # setup() swaps in a fake; the routing test needs the real one
from tracer.mutation import diff_entities, refine_mutations  # noqa: E402
from tracer.report import render_report  # noqa: E402

UTC = timezone.utc
T0 = datetime.now(UTC) - timedelta(days=10)
CLAIM = "Apple is shutting down iCloud"
calls = {"da": 0, "expansion": 0}


# ------------------------------------------------------------------ fake sources
def raw(src, url, author, hours, text, eng=0, **meta):
    return dict(source=src, url=url, author=author, ts=T0 + timedelta(hours=hours), text=text, engagement=eng, meta=meta)


async def f_reddit(c, q):
    return [raw("reddit", "https://reddit.com/r/apple/1", "rumor_guy", 0, "Apple is shutting down iCloud, 1000000 accounts affected", 900)]


async def f_bluesky(c, q):
    return [raw("bluesky", "https://bsky.app/profile/amp.bsky.social/post/aaa", "amp.bsky.social", 6,
                "wow via u/rumor_guy: Apple is shutting down iCloud, 5 million accounts affected", 80,
                at_uri="at://did:x/app.bsky.feed.post/aaa", links=["https://reddit.com/r/apple/1"])]


async def f_gdelt(c, q):
    if "domain:apple.com" in q:
        return [raw("gdelt", "https://www.apple.com/newsroom/2026/09/icloud-update/", "apple.com", 100, "iCloud update")]
    return [raw("gdelt", "https://news.example.com/apple-icloud", "news.example.com", 30,
                "Apple to discontinue iCloud, sources say")]


async def f_news(c, q):
    return [raw("news", "https://factcheck.example.org/icloud", "factcheck.example.org", 50,
                "Fact check: Apple is not shutting down iCloud, claim is false")]


async def f_wiki(c, q):
    return [dict(url="https://en.wikipedia.org/wiki/ICloud", title="iCloud", text="iCloud is a cloud storage service by Apple.")]


async def f_wayback(c, url):
    return T0 - timedelta(days=2) if "news.example.com" in url else None


# ---------------------------------------------------------------- fake HTTP pages
PAGES = {
    "https://news.example.com/apple-icloud":
        '<html><title>Apple to discontinue iCloud</title><article><p>Per <a href="https://www.apple.com/newsroom/2026/09/icloud-update/">Apple\'s statement</a> and '
        '<a href="https://www.sec.gov/Archives/edgar/data/320193/x.htm">filing</a>.</p></article></html>',
    "https://www.apple.com/newsroom/2026/09/icloud-update/":
        "<html><title>iCloud update</title><article><p>iCloud is not going away; storage plans are changing.</p></article></html>",
    "https://www.sec.gov/Archives/edgar/data/320193/x.htm": "<html><title>8-K</title><p>Apple Inc. 8-K, no iCloud discontinuation.</p></html>",
    "https://www.sec.gov/Archives/edgar/data/320193/000032019326000100/aapl-8k.htm": "<html><title>8-K</title><p>Apple 8-K.</p></html>",
}


class Resp:
    def __init__(self, text="", data=None):
        self.text, self._d = text, data

    def json(self):
        return self._d

    def raise_for_status(self):
        pass


class FakeClient:
    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, **kw):
        if "efts.sec.gov" in url:
            return Resp(data={"hits": {"hits": [{"_id": "0000320193-26-000100:aapl-8k.htm", "_source": {
                "adsh": "0000320193-26-000100", "ciks": ["0000320193"], "form": "8-K", "display_names": ["Apple Inc."]}}]}})
        if url in PAGES:
            return Resp(PAGES[url])
        raise RuntimeError("404")


# --------------------------------------------------------------------- fake LLM
async def fake_llm(system, user, tier="fast", max_tokens=4000):
    if "Query Expansion" in system:
        calls["expansion"] += 1
        return {"paraphrases": ["Apple discontinues iCloud", "Apple ends iCloud service"],
                "keywords": {"gdelt": ["apple icloud shutdown"], "news": ["apple icloud discontinued"]},
                "orgs": ["Apple"], "official_domains": ["apple.com", "not a domain"], "sec_relevant": True}
    if "Claim Matcher" in system:
        items = json.loads(user.split("Items:\n")[1])
        a = {}
        for it in items:
            if "Fact check" in it["text"]:
                a[it["id"]] = {"variant": "V0", "stance": "debunk"}
            elif "5 million" in it["text"]:
                a[it["id"]] = {"variant": "V1", "stance": "assert"}
            elif it["src"] in ("reddit", "gdelt"):
                a[it["id"]] = {"variant": "V0", "stance": "assert" if it["src"] == "reddit" else "report"}
        return {"variants": [
            {"id": "V0", "text": "Apple is shutting down iCloud, 1000000 accounts affected", "mutation": "original", "parent": None},
            {"id": "V1", "text": "Apple is shutting down iCloud, 5 million accounts affected", "mutation": "weird_label", "parent": "V0"}],
            "assignments": a}
    if "Break the claim" in system:
        return {"subclaims": [{"id": "S1", "text": "Apple announced an iCloud shutdown", "depends_on": []},
                              {"id": "S2", "text": "1M accounts affected", "depends_on": ["S1"]}],
                "wikipedia": ["iCloud"], "primary_hint": "Apple newsroom / 8-K"}
    if "Judge each sub-claim" in system:
        docs = json.loads(user)["reference_docs"]
        prim = [d["id"] for d in docs if d["weight"] >= 0.75]
        return {"subclaims": [{"id": "S1", "verdict": "refuted", "reasoning": "Apple says it is not going away.",
                               "supporting": prim[:2], "dissenting": []},
                              {"id": "S2", "verdict": "unverifiable", "reasoning": "no data", "supporting": [], "dissenting": []}],
                "missing_context": ["Storage plan changes were announced"],
                "overall": {"verdict": "false", "summary": "Apple denies it."}}
    if "You are the Judge" in system:
        return {"verdict": "false", "rationale": "Primary sources contradict it.", "origin_statement": "Reddit, unconfirmed.",
                "changed_from_factchecker": False, "caveats": ["Search coverage is partial"]}
    if "Devil's Advocate" in system:
        calls["da"] += 1
        if calls["da"] == 1:
            return {"verdict_holds": False, "attacks": [{"target": "source", "argument": "Could be outdated", "severity": "high"}],
                    "follow_up": {"news": ["apple icloud storage plans"], "gdelt": [], "reddit": [], "bluesky": []}}
        return {"verdict_holds": True, "attacks": [{"target": "origin", "argument": "coverage gaps", "severity": "low"}], "follow_up": {}}
    raise AssertionError("unexpected prompt: " + system[:60])


def setup():
    llm.ask_json = fake_llm
    sources.SOURCES.update(reddit=f_reddit, bluesky=f_bluesky, gdelt=f_gdelt, news=f_news)
    sources.gdelt, sources.wikipedia, sources.wayback_first = f_gdelt, f_wiki, f_wayback
    httpx.AsyncClient = FakeClient


def setup_module():
    setup()


# ------------------------------------------------------------------------ tests
def test_mutation_labels():
    vs = [{"id": "V0", "text": "5000 migrants crossed", "parent": None, "mutation": "original"},
          {"id": "V1", "text": "50,000 migrants crossed", "parent": "V0", "mutation": "zzz"},
          {"id": "V2", "text": "5000 migrants crossed in 2019", "parent": "V0", "mutation": "other"},
          {"id": "V3", "text": "x", "parent": "NOPE", "mutation": "fabrication"}]
    refine_mutations(vs)
    assert vs[1]["mutation"] == "exaggeration" and vs[1]["diff"]["numbers"]["ratio"] == 10
    assert vs[2]["mutation"] == "date_shift"
    assert vs[3]["parent"] is None


def test_scrape_links():
    page = ('<nav><a href="/home">h</a></nav><article><p>See <a href="https://web.archive.org/web/2026/https://www.justice.gov/p/1">DOJ</a>'
            '<a href="https://twitter.com/intent/tweet?x">share</a></p></article>')
    assert scrape.extract_links(page, "https://n.example.com/a") == ["https://www.justice.gov/p/1"]


def test_pipeline(use_langgraph=False):
    calls.update(da=0, expansion=0)
    st = asyncio.run(pipeline.run_pipeline(CLAIM, rounds=3, use_langgraph=use_langgraph))
    r = st.result
    assert calls["da"] == 2 and calls["expansion"] == 1, calls  # round 2 used follow-up queries, then verdict held
    assert r["verdict"] == "false"
    assert r["origin"]["likely"]["source"] == "reddit" and r["origin"]["likely"]["author"] == "rumor_guy"
    assert r["origin"]["probability"] > 0.5
    assert r["origin"]["evidence"], r["origin"]
    labels = {n["id"]: n["label"] for n in r["mutation_graph"]["nodes"]}
    assert labels == {"V0": "original", "V1": "exaggeration"}, labels
    assert r["mutation_graph"]["edges"][0]["diff"]["numbers"]["ratio"] == 5
    types = {d["type"] for d in r["fact_check"]["supporting_evidence"]}
    assert types & {"press_release", "government"}, types  # primary-type evidence was found and cited
    assert r["scores"]["primary_sources"] >= 1 and r["confidence"] > 0.3
    assert any(ch["resolved"] for ch in r["citation_chains"])
    assert st.refs and all(x["weight"] >= 0.4 for x in st.refs)
    assert any(x.get("first_capture") is None for x in st.refs)  # wayback lookups ran
    # archive pass pushed the news article earlier than GDELT's seen-date
    news = next(e for e in st.pool.values() if "news.example.com" in e.url)
    assert news.eff_ts < news.ts
    assert r["scores"]["after_critique"] <= r["scores"]["confidence"]
    md = render_report(st)
    assert "Likely origin" in md and "Mutation lineage" in md and "Citation chains" in md
    json.dumps(r)  # result is JSON-serialisable
    return st


def test_langgraph_wiring():
    """Verify node/edge topology against a minimal stand-in for langgraph (real LangGraph is not installed here)."""
    mod = types.ModuleType("langgraph.graph")
    mod.END = "__end__"

    class SG:
        def __init__(self, schema):
            self.nodes, self.edges, self.cond, self.entry = {}, {}, {}, None

        def add_node(self, n, fn):
            self.nodes[n] = fn

        def set_entry_point(self, n):
            self.entry = n

        def add_edge(self, a, b):
            self.edges[a] = b

        def add_conditional_edges(self, a, fn, mapping):
            self.cond[a] = (fn, mapping)

        def compile(self):
            g = self

            class App:
                async def ainvoke(self, state, config=None):
                    n = g.entry
                    while n != "__end__":
                        state.update(await g.nodes[n](state) or {})
                        if n in g.cond:
                            fn, m = g.cond[n]
                            n = m[fn(state)]
                        else:
                            n = g.edges[n]
                    return state
            return App()

    mod.StateGraph = SG
    pkg = types.ModuleType("langgraph")
    sys.modules.update({"langgraph": pkg, "langgraph.graph": mod})
    try:
        test_pipeline(use_langgraph=True)
    finally:
        del sys.modules["langgraph"], sys.modules["langgraph.graph"]


def test_llm_routing():
    """Tier -> provider mapping, Groq-size fallback to Gemini, and Groq REST call shape (no network)."""
    import asyncio, importlib, os
    from tracer import config, llm
    orig_tiers = dict(config.TIERS)
    config.TIERS["smart"] = ("gemini", "gemini-3.8-flash")
    config.TIERS["adversary"] = ("groq", "openai/gpt-oss-120b")
    try:
        assert set(p for p, _ in config.TIERS.values()) <= {"gemini", "groq"}
        assert config.TIERS["adversary"][0] != config.TIERS["smart"][0], "adversary should be a different family"
        os.environ["GEMINI_API_KEY"] = "x"; os.environ["GROQ_API_KEY"] = "y"
        plan = llm._plan("adversary", 100)
        assert plan[0][0] == "groq" and plan[1][0] == "gemini", plan
        big = llm._plan("adversary", config.GROQ_MAX_INPUT_CHARS + 1)
        assert big[0][0] == "gemini", big            # oversized prompt avoids Groq's small TPM budget
        seen = {}
        async def fake_groq(system, user, model, max_tokens):
            seen["model"] = model; return 'noise {"ok": true} noise'
        llm._CALLS["groq"] = fake_groq
        fake_ask, llm.ask_json = llm.ask_json, _REAL_ASK
        out = asyncio.run(llm.ask_json("s", "u", "adversary"))
        assert out == {"ok": True} and seen["model"] == config.TIERS["adversary"][1]
        async def bad_groq(*a):
            raise llm.RateLimited(0)
        async def ok_gemini(*a): return '{"via": "gemini"}'
        llm._CALLS["groq"], llm._CALLS["gemini"] = bad_groq, ok_gemini
        assert asyncio.run(llm.ask_json("s", "u", "adversary")) == {"via": "gemini"}   # fallback on 429
    finally:
        llm.ask_json = fake_ask
        config.TIERS.clear()
        config.TIERS.update(orig_tiers)


if __name__ == "__main__":
    setup()
    for t in (test_mutation_labels, test_scrape_links, test_pipeline, test_langgraph_wiring, test_llm_routing):
        t()
        print("PASS", t.__name__)
