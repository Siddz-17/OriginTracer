"""Offline end-to-end smoke test: fake LLM + fake sources + fake HTTP. Run: python tests/test_smoke.py"""
import asyncio
import json
import os
import sys
import types
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("TRACER_LOCAL_MODELS", "0")  # deterministic heuristics; test_local_models opts in
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
calls = {"da": 0, "expansion": 0, "analysis": 0}


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


async def f_outlets(c, q):
    return [raw("news", "https://news.google.com/rss/articles/reuters1", "Reuters", 40,
                "Apple says it is not shutting down iCloud", outlet_domain="reuters.com")]


async def f_factcheck(c, q):
    return [raw("news", "https://www.snopes.com/fact-check/apple-icloud-shutdown/", "Snopes", 60,
                "Fact Check: Is Apple shutting down iCloud? No, the rumor is false", outlet_domain="snopes.com")]


async def f_empty(c, q):
    return []


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


async def fake_resolve(url, c=None):
    return None  # Google News links stay unresolved offline


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
            if "Fact check" in it["text"] or "Fact Check" in it["text"]:
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
    if "Judge each sub-claim ONLY" in system:
        docs = json.loads(user)["reference_docs"]
        prim = [d["id"] for d in docs if d["weight"] >= 0.75]
        return {"subclaims": [{"id": "S1", "verdict": "refuted", "reasoning": "Apple says it is not going away.",
                               "supporting": prim[:2], "dissenting": []},
                              {"id": "S2", "verdict": "unverifiable", "reasoning": "no data", "supporting": [], "dissenting": []}],
                "missing_context": ["Storage plan changes were announced"],
                "overall": {"verdict": "false", "summary": "Apple denies it."}}
    if "meticulous fact-checker" in system:  # the single consolidated analysis call
        calls["analysis"] += 1
        pack = json.loads(user)
        assert pack["evidence"][0]["type"] in ("fact_check", "press_release", "government"), pack["evidence"][0]
        strong = [e["id"] for e in pack["evidence"] if e["weight"] >= 0.7]
        return {"subclaims": [{"id": "S1", "text": "Apple announced an iCloud shutdown", "verdict": "refuted",
                               "reasoning": "Apple and fact-checkers say no.", "supporting": strong[:3] + ["BOGUS"],
                               "dissenting": []}],
                "verdict": "false", "rationale": "Fact-checkers and Apple's own statement refute it.",
                "missing_context": ["Storage plans changed"], "origin_statement": "Reddit post, unconfirmed.",
                "self_critique": {"attacks": [{"target": "origin", "argument": "coverage gaps", "severity": "medium"}],
                                  "verdict_holds": True},
                "debunk_ids": [], "irrelevant_ids": [],
                "caveats": ["Search coverage is partial"], "primary_hint": "Apple newsroom"}
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
    sources.SOURCES.clear()
    sources.SOURCES.update(reddit=f_reddit, bluesky=f_bluesky, gdelt=f_gdelt, news=f_news,
                           outlets=f_outlets, factcheck=f_factcheck, duckduckgo=f_empty)
    sources.gdelt, sources.wikipedia, sources.wayback_first = f_gdelt, f_wiki, f_wayback
    sources.ARCHIVE_TODAY = False  # offline: no archive.today lookups
    scrape.resolve_redirect = fake_resolve
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


def test_local_matching():
    from tracer import local
    from tracer.outlets import outlet_tier
    p = local.claim_profile("NASA confirmed Earth will experience 15 days of darkness in November")
    assert local.relevance(p, "No, NASA did not say Earth will go dark for 15 days of darkness") >= 0.5
    assert local.relevance(p, "November 10, 1967: NASA's First Color Picture of the Earth") < 0.5
    assert local.stance("No, Earth will not experience 15 days of darkness") == "debunk"
    assert local.stance("NASA confirms 15 days of darkness") == "report"
    p2 = local.claim_profile(CLAIM)
    assert local.relevance(p2, "Apple to discontinue iCloud, sources say") > 0.8  # synonym match
    assert local.relevance(p2, "Google shuts down Stadia") < 0.3
    assert outlet_tier("https://www.reuters.com/fact-check/x") == "fact_check"
    assert outlet_tier("apnews.com") == "wire" and outlet_tier("https://www.bbc.co.uk/news/x") == "major"
    assert outlet_tier("randomblog.example") is None
    plan, _, _, base = local.expand_queries("NASA confirmed Earth will experience 15 days of darkness in November")
    assert "darkness" in base and "confirmed" not in base, base


def test_json_extraction():
    txt = '<think>maybe {"a": 1}</think>```json\n{"verdict": "false", "x": {"y": 2}}\n```'
    assert llm._parse(txt) == {"verdict": "false", "x": {"y": 2}}
    assert llm._parse('Sure! {"a": 1} and then {"verdict": "true", "subclaims": []}') == {"verdict": "true", "subclaims": []}
    assert llm._parse("no json here") == {}


def test_pipeline(use_langgraph=False):
    """Legacy multi-agent mode."""
    calls.update(da=0, expansion=0, analysis=0)
    st = asyncio.run(pipeline.run_pipeline(CLAIM, rounds=3, use_langgraph=use_langgraph, mode="agents"))
    r = st.result
    assert calls["da"] == 2 and calls["expansion"] == 1, calls  # round 2 used follow-up queries, then verdict held
    assert r["verdict"] == "false"
    assert r["origin"]["likely"]["source"] == "reddit" and r["origin"]["likely"]["author"] == "rumor_guy"
    assert r["origin"]["probability"] > 0.5
    assert r["origin"]["evidence"], r["origin"]
    labels = {n["id"]: n["label"] for n in r["mutation_graph"]["nodes"]}
    assert labels == {"V0": "original", "V1": "exaggeration"}, labels
    assert r["mutation_graph"]["edges"][0]["diff"]["numbers"]["ratio"] == 5
    types_ = {d["type"] for d in r["fact_check"]["supporting_evidence"]}
    assert types_ & {"press_release", "government"}, types_  # primary-type evidence was found and cited
    assert all(s.get("text") for s in r["fact_check"]["subclaims"])  # sub-claim text merged from the plan
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


def test_single_call_pipeline():
    """Default mode: retrieval / matching / ranking are local and exactly ONE LLM call is made."""
    calls.update(da=0, expansion=0, analysis=0)
    st = asyncio.run(pipeline.run_pipeline(CLAIM, rounds=2, mode="single"))
    r = st.result
    assert calls == {"da": 0, "expansion": 0, "analysis": 1}, calls
    assert r["verdict"] == "false"
    sub = r["fact_check"]["subclaims"][0]
    assert sub["text"] and "BOGUS" not in sub["supporting"], sub  # invented ids are dropped
    assert r["scores"]["after_critique"] < r["scores"]["confidence"]  # medium self-attack lowers confidence
    ranked = r["evidence_ranked"]
    assert ranked and any(x["tier"] == "fact_check" for x in ranked), ranked
    assert any(x["outlet"] == "reuters.com" for x in ranked), ranked  # reputable outlet item kept
    assert r["origin"]["likely"]["source"] == "reddit", r["origin"]["likely"]
    ev = st.by_id()
    assert all(a["stance"] == "debunk" for i, a in st.assign.items() if "snopes" in ev[i].url)
    labels = {n["id"]: n["label"] for n in r["mutation_graph"]["nodes"]}
    assert labels.get("V0") == "original" and "exaggeration" in labels.values(), labels  # 1M -> 5M
    assert "Likely origin" in render_report(st)
    json.dumps(r)


def test_langgraph_wiring():
    """Verify node/edge topology against a minimal stand-in for langgraph."""
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
    saved = {k: sys.modules.get(k) for k in ("langgraph", "langgraph.graph")}
    sys.modules.update({"langgraph": pkg, "langgraph.graph": mod})
    try:
        test_pipeline(use_langgraph=True)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_llm_routing():
    """Tier -> provider mapping, model walking, rate-limit / quota handling (no network)."""
    from tracer import config
    orig_tiers = dict(config.TIERS)
    orig_calls = dict(llm._CALLS)
    orig_interval = dict(llm._MIN_INTERVAL)
    orig_env = {k: os.environ.get(k) for k in ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "GOOGLE_API_KEY")}
    llm._MIN_INTERVAL.update(openrouter=0, groq=0, gemini=0)
    current_ask = llm.ask_json
    config.TIERS["fast"] = ("openrouter", config.OPENROUTER_MODELS[0])
    config.TIERS["smart"] = ("gemini", "gemini-2.5-flash")
    config.TIERS["adversary"] = ("groq", "openai/gpt-oss-120b")
    llm.ask_json = _REAL_ASK
    try:
        os.environ.update(GEMINI_API_KEY="x", GROQ_API_KEY="y", OPENROUTER_API_KEY="z")

        plan = llm._plan("adversary", 100)
        assert plan[0][0] == "groq", f"Expected groq first in plan, got: {plan}"
        big = llm._plan("adversary", config.GROQ_MAX_INPUT_CHARS + 1)
        assert big[0][0] != "groq", f"Groq should be deprioritised for big prompts, got: {big}"

        seen = {}

        async def fake_groq(system, user, model, max_tokens):
            seen["model"] = model
            return 'noise {"ok": true} noise'
        llm._CALLS["groq"] = fake_groq
        out = asyncio.run(llm.ask_json("s", "u", "adversary"))
        assert out == {"ok": True}, f"Expected {{ok: True}}, got {out}"
        assert seen["model"] == config.TIERS["adversary"][1]

        # groq 429 -> every retired OpenRouter model tried once -> gemini answers
        async def bad_groq(*a):
            raise llm.RateLimited(0)

        async def ok_gemini(*a):
            return '{"via": "gemini"}'

        async def or_404(system, user, model, max_tokens):
            seen.setdefault("or_models", []).append(model)
            raise llm.ModelUnavailable("404 retired")
        llm._CALLS.update(groq=bad_groq, gemini=ok_gemini, openrouter=or_404)
        result = asyncio.run(llm.ask_json("s", "u", "adversary"))
        assert result.get("via") == "gemini", f"Expected gemini fallback, got: {result}"
        assert seen["or_models"] == config.OPENROUTER_MODELS, seen

        # OpenRouter: a retired model, then a rate-limited one, fall through to the next free model (no sleep)
        config.TIERS["smart"] = ("openrouter", config.OPENROUTER_MODELS[0])
        tried = []

        async def or_flaky(system, user, model, max_tokens):
            tried.append(model)
            if len(tried) == 1:
                raise llm.ModelUnavailable("404")
            if len(tried) == 2:
                raise llm.RateLimited(30)
            return '{"model": "%s"}' % model
        llm._CALLS["openrouter"] = or_flaky
        out = asyncio.run(llm.ask_json("s", "u", "smart"))
        assert out["model"] == config.OPENROUTER_MODELS[2] and len(tried) == 3, (out, tried)

        # Daily free quota exhausted: stop using OpenRouter after one request
        tried.clear()

        async def or_daily(system, user, model, max_tokens):
            tried.append(model)
            raise llm.ProviderUnavailable("daily free quota exhausted")
        llm._CALLS["openrouter"] = or_daily
        for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
            os.environ.pop(k, None)
        try:
            asyncio.run(llm.ask_json("s", "u", "smart"))
            raise AssertionError("expected NoJsonReturned")
        except llm.NoJsonReturned:
            pass
        assert len(tried) == 1, tried
    finally:
        llm.ask_json = current_ask
        llm._CALLS.clear()
        llm._CALLS.update(orig_calls)
        llm._MIN_INTERVAL.update(orig_interval)
        config.TIERS.clear()
        config.TIERS.update(orig_tiers)
        for k, v in orig_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_stance_and_matching_accuracy():
    """Regression cases for the local matcher: claim-aware stance, fact-check ratings, stemmed synonyms."""
    from tracer import local
    moon = local.claim_profile("Chandrayaan-3 landed near the Moon's south pole")
    icloud = local.claim_profile("Apple is shutting down iCloud")
    nasa = local.claim_profile("NASA confirmed Earth will experience 6 days of total darkness")
    gates = local.claim_profile("Bill Gates is the largest private owner of farmland in the United States")
    bleach = local.claim_profile("Drinking bleach cures COVID-19")
    cases = [
        (moon, "ISRO says Chandrayaan-3 lander did not wake up after lunar night", "report"),  # unrelated negation
        (moon, "No, Chandrayaan-3 did not land on the Moon", "debunk"),
        (icloud, "Apple denies report it will discontinue iCloud", "debunk"),
        (nasa, "Did NASA say Earth will go dark for six days? No", "debunk"),
        (gates, "Fact check: Bill Gates is the largest private farmland owner in US - True", "support"),
        (gates, "Fact check: Is Bill Gates the largest farmland owner? Rating: Mostly True", "support"),
        (bleach, "Fact Check: Drinking bleach does not cure COVID-19", "debunk"),
        (bleach, "Bleach poisoning calls rise during pandemic", "report"),
    ]
    for prof, text, want in cases:
        assert local.stance(text, prof) == want, (text, local.stance(text, prof), want)
    assert local.fact_check_rating("x", {"rating": "Mostly False"}) == "misleading"
    assert local.fact_check_rating("x", {"rating": "Pants on Fire!"}) == "false"
    assert local.relevance(icloud, "Apple discontinues iCloud") > 0.8  # stemmed synonym ("discontinues")
    assert local.relevance(icloud, "Pineapple farms shutting down in Hawaii cloud") < 0.3  # no substring names
    assert local.relevance(nasa, "NASA says Earth will go dark for six days") > 0.7  # number words
    assert local.is_preview("Chandrayaan-3 landed near the Moon's south pole", "India plans to land Chandrayaan-3")
    assert not local.is_preview("Bill Gates is the largest private owner of farmland in the United States",
                                "Gates could become the largest owner")
    assert local.predicate_coverage(moon, "Chandrayaan-3 mission: ISRO shares new images") < 0.5


def test_verdict_checks():
    """Deterministic verdict reconciliation, fact-check consensus and evidence re-orientation."""
    from types import SimpleNamespace

    from tracer import analysis, scoring
    subs = [{"verdict": "refuted"}, {"verdict": "refuted"}]
    v, notes = scoring.reconcile_verdict("true", subs, {"majority": None})
    assert v == "false" and notes, (v, notes)  # contradiction with its own sub-claims is fixed
    v, notes = scoring.reconcile_verdict("false", [{"verdict": "refuted"}], {"majority": "true", "publishers": {}})
    assert v == "false" and notes  # disagreement with fact-checkers is flagged, not overridden
    assert scoring.implied_verdict([{"verdict": "supported"}, {"verdict": "refuted"}]) == "misleading"
    assert scoring.confidence_cap(0, 2) == 0.8 and scoring.confidence_cap(1, 0) == 1.0
    # a 'refuted' sub-claim whose debunks the model filed under "dissenting" is re-oriented
    st = SimpleNamespace(assign={"E1": {"stance": "debunk", "relevance": 0.9},
                                 "E2": {"stance": "report", "relevance": 0.9},
                                 "E3": {"stance": "debunk", "relevance": 0.55}}, emit=lambda *a: None)
    sc = {"id": "S1", "verdict": "refuted", "supporting": [], "dissenting": ["E1", "E2"]}
    analysis._orient_evidence(st, [sc])
    assert sc["supporting"] == ["E1"] and sc["dissenting"] == ["E2"], sc
    # a loosely related debunk (a different rumour about the same event) keeps the model's placement
    sc = {"id": "S1", "verdict": "supported", "supporting": ["E2", "E3"], "dissenting": []}
    analysis._orient_evidence(st, [sc])
    assert sc["supporting"] == ["E2", "E3"] and sc["dissenting"] == [], sc


def test_local_models():
    """Embedding veto / rescue and NLI debunks (skipped when sentence-transformers is not installed)."""
    from tracer import nlp
    old = os.environ.get("TRACER_LOCAL_MODELS")
    os.environ["TRACER_LOCAL_MODELS"] = "auto"
    try:
        if not nlp.enabled():
            print("SKIP test_local_models (sentence-transformers not installed)")
            return
        claim = "Bill Gates is the largest private owner of farmland in the United States"
        sims = nlp.similarities(claim, ["Bill Gates is the biggest private owner of farmland in the US",
                                        "Fact check: Bill Gates did not promote water reserve cryptocoin"])
        assert sims[0] >= nlp.RESCUE_SIM and sims[1] < nlp.MATCH_MIN_SIM, sims
        con = nlp.contradiction("Drinking bleach cures COVID-19",
                                ["Fact Check: Drinking bleach does not cure COVID-19"])
        assert con[0] >= nlp.NLI_CONTRA, con
    finally:
        if old is None:
            os.environ.pop("TRACER_LOCAL_MODELS", None)
        else:
            os.environ["TRACER_LOCAL_MODELS"] = old


def test_image_provenance():
    """Fingerprint, EXIF flags, reverse-search dating and pHash relation, all offline."""
    from tracer import image
    if not image._libs():
        print("SKIP test_image_provenance (pillow / imagehash not installed)")
        return
    import io as _io
    import tempfile
    from PIL import Image
    img = Image.new("RGB", (64, 64), (200, 30, 30))
    for x in range(32):
        img.putpixel((x, x), (0, 0, 0))
    exif = img.getexif()
    exif[0x0131] = "Adobe Photoshop 25.0"  # Software
    exif[0x0132] = "2019:05:01 10:00:00"    # DateTime
    buf = _io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    path = os.path.join(tempfile.mkdtemp(), "x.jpg")
    open(path, "wb").write(buf.getvalue())

    info, pil, ph = image.fingerprint(buf.getvalue())
    assert info["format"] == "JPEG" and info["phash"] and len(info["sha256"]) == 64, info
    assert any("editing software" in f for f in info["flags"]), info["flags"]
    assert image.relation(3) == "same" and image.relation(15) == "variant" and image.relation(40) == "different"
    assert "tineye.com/search?url=" in image.manual_links("https://ex.com/a.jpg")["tineye"]

    async def fake_lens(c, url, key):
        return [{"url": "https://old.example/2015", "title": "old", "site": "old.example", "thumbnail": "t1",
                 "first_seen": None, "engine": "google_lens"},
                {"url": "https://new.example/2024", "title": "new", "site": "new.example", "thumbnail": "t2",
                 "first_seen": None, "engine": "google_lens"}]

    async def fake_dist(c, thumb, ph_):
        return 2 if thumb == "t1" else 30

    async def fake_capture(c, url):
        d = datetime(2015, 3, 1, tzinfo=UTC) if "old" in url else datetime(2024, 1, 1, tzinfo=UTC)
        return d, "wayback"

    orig = (image._lens, image._thumb_distance, sources.first_capture, os.environ.get("SERPAPI_API_KEY"))
    image._lens, image._thumb_distance, sources.first_capture = fake_lens, fake_dist, fake_capture
    os.environ["SERPAPI_API_KEY"] = "test"
    try:
        engine, ms = asyncio.run(image.reverse_search(None, "https://ex.com/a.jpg", ph))
        assert engine == "google_lens" and ms[0]["site"] == "old.example", ms
        assert ms[0]["relation"] == "same" and ms[1]["relation"] == "different", ms
        assert ms[0]["first_seen"].startswith("2015-03-01"), ms[0]
        rep = asyncio.run(image.analyze_image(None, path))  # local file: no reverse search, manual links instead
        assert rep["phash"] == info["phash"] and rep["manual_search"]["note"], rep
    finally:
        image._lens, image._thumb_distance, sources.first_capture = orig[:3]
        if orig[3] is None:
            os.environ.pop("SERPAPI_API_KEY", None)
        else:
            os.environ["SERPAPI_API_KEY"] = orig[3]


def test_recent_sources():
    """Bing News RSS parsing (real URL unwrapped from the click link) and the recency ranking bonus."""
    from datetime import timedelta as _td
    from tracer import local
    from tracer.models import Ev
    rss = (b'<?xml version="1.0"?><rss xmlns:News="http://www.bing.com/schema/news"><channel><item>'
           b'<title>Ronaldo retires from Portugal</title>'
           b'<link>http://www.bing.com/news/apiclick.aspx?ref=FexRss&amp;url=https%3a%2f%2fapnews.com%2farticle%2fx&amp;c=1</link>'
           b'<description>The forward announced it on Wednesday</description>'
           b'<pubDate>Wed, 05 Nov 2026 10:00:00 GMT</pubDate><News:Source>AP News</News:Source></item></channel></rss>')
    items = list(sources._bing_items(rss))
    assert items[0]["url"] == "https://apnews.com/article/x" and items[0]["author"] == "AP News", items
    assert items[0]["meta"]["outlet_domain"] == "apnews.com"
    now = datetime.now(UTC)
    fresh = Ev(id="E1", source="news", url="u1", author="a", ts=now - _td(days=2), text="x")
    old = Ev(id="E2", source="news", url="u2", author="a", ts=now - _td(days=400), text="x")
    undated = Ev(id="E3", source="news", url="u3", author="a", ts=now, text="x", meta={"undated": True})
    assert local.recency_bonus(fresh) > local.recency_bonus(old) == local.recency_bonus(undated) == 0.0


if __name__ == "__main__":
    setup()
    for t in (test_mutation_labels, test_scrape_links, test_local_matching, test_json_extraction, test_pipeline,
              test_single_call_pipeline, test_langgraph_wiring, test_llm_routing, test_stance_and_matching_accuracy,
              test_verdict_checks, test_local_models,
              test_image_provenance, test_recent_sources):
        t()
        print("PASS", t.__name__)
