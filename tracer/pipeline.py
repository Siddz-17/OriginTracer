"""Pipeline wiring.

mode="single" (default, free-tier friendly): exactly ONE LLM call per run.
    local query expansion -> discovery (reputable outlets + fact-checkers first) -> local claim matching
    -> [retrieval round 2+: pseudo-relevance-feedback queries] -> origin tracing (Wayback) -> evidence
    collection (article bodies, citation chains, Wikipedia, user primaries) -> local evidence ranking
    -> ONE consolidated LLM analysis (fact-check + self-critique + verdict) -> code-computed confidence

mode="agents" (legacy): LangGraph if installed, otherwise an equivalent sequential runner.
expansion -> discovery -> matcher -> origin -> evidence -> factcheck -> devil --(holds | out of rounds)--> judge
                 ^                                                      |
                 +------------------- follow-up queries -----------------+
"""
from typing import Any, TypedDict

import httpx

from . import agents, analysis, config, local, sources
from .models import State


class Flow(TypedDict, total=False):
    st: State
    c: Any
    round: int
    max_rounds: int
    depth: int
    archive: bool


# ------------------------------------------------------------------ single-call mode
async def run_single(c, st, rounds=2, depth=3, archive=True):
    plan, st.paraphrases, st.orgs, st.base_query = local.expand_queries(st.claim)
    st.plan = {k: v for k, v in plan.items() if k in sources.SOURCES}
    if "reddit" in sources.SOURCES:
        st.plan["reddit"] = [st.base_query]
    st.emit("expansion", f"{sum(len(v) for v in st.plan.values())} keyword queries across {len(st.plan)} sources "
                         f"(no LLM); base query '{st.base_query}'")
    for r in range(max(1, rounds)):
        st.emit("pipeline", f"retrieval round {r + 1}/{max(1, rounds)}")
        await agents.source_discovery(c, st)
        n = local.match_items(st)
        st.emit("matcher", f"{n} new items match the claim locally ({len(st.assign)} total, "
                           f"{sum(1 for a in st.assign.values() if a.get('outlet_tier'))} from reputable outlets, "
                           f"{sum(1 for a in st.assign.values() if a.get('stance') == 'debunk')} debunk-stance)")
        if r + 1 >= rounds:
            break
        fu = local.followup_queries(st)
        if not fu:
            st.emit("expansion", "no new distinctive terms for another retrieval round; stopping early")
            break
        st.plan = {"outlets": fu[:1], "news": fu, "factcheck": fu[:1], "duckduckgo": fu[:1]}
        st.emit("expansion", f"follow-up queries from matched headlines: {fu}")
    local.build_variants(st)
    st.emit("matcher", f"{len(st.variants)} variants by stated numbers/dates")
    await agents.origin_tracer(c, st, archive=archive)
    await agents.evidence_collection(c, st, depth=depth)
    await agents.fact_checker_background(c, st)
    for k, r in enumerate(st.refs, 1):  # stable ids for the analysis prompt (P = primary/page, W = Wikipedia)
        r["label"] = f"{r['kind']}{sum(1 for x in st.refs[:k] if x['kind'] == r['kind'])}"
    await analysis.consolidated_analysis(c, st, config.ANALYSIS_MAX_ITEMS, config.ANALYSIS_MAX_CHARS)
    await agents.origin_tracer(c, st, archive=False, quiet=True)  # re-score origin with corrected stances
    agents.build_result(st)
    st.emit("judge", f"final verdict {st.result['verdict']} confidence {st.result['confidence']}")


# ------------------------------------------------------------------ legacy multi-agent mode
def route(f: Flow) -> str:
    st = f["st"]
    holds = bool(st.critiques and st.critiques[-1].get("verdict_holds"))
    return "judge" if holds or f["round"] >= f["max_rounds"] else "again"


async def n_expansion(f):
    f["st"].emit("pipeline", f"round {f.get('round', 0) + 1}/{f['max_rounds']}")
    await agents.query_expansion(f["c"], f["st"])
    return {"round": f.get("round", 0) + 1}


async def n_discovery(f):
    await agents.source_discovery(f["c"], f["st"])
    return {}


async def n_matcher(f):
    await agents.claim_matcher(f["c"], f["st"])
    return {}


async def n_origin(f):
    await agents.origin_tracer(f["c"], f["st"], archive=f.get("archive", True))
    return {}


async def n_evidence(f):
    await agents.evidence_collection(f["c"], f["st"], depth=f.get("depth", 3))
    return {}


async def n_factcheck(f):
    st = f["st"]
    await agents.fact_checker(f["c"], st, st.critiques[-1] if st.critiques else None)
    return {}


async def n_devil(f):
    await agents.devils_advocate(f["c"], f["st"])
    return {}


async def n_judge(f):
    await agents.judge(f["c"], f["st"])
    return {}


NODES = [("expansion", n_expansion), ("discovery", n_discovery), ("matcher", n_matcher), ("origin", n_origin),
         ("evidence", n_evidence), ("factcheck", n_factcheck), ("devil", n_devil)]


def build_langgraph():
    from langgraph.graph import END, StateGraph
    g = StateGraph(Flow)
    for name, fn in NODES + [("judge", n_judge)]:
        g.add_node(name, fn)
    g.set_entry_point("expansion")
    for (a, _), (b, _) in zip(NODES, NODES[1:]):
        g.add_edge(a, b)
    g.add_conditional_edges("devil", route, {"again": "expansion", "judge": "judge"})
    g.add_edge("judge", END)
    return g.compile()


async def run_sequential(flow: Flow):
    while True:
        for _, fn in NODES:
            flow.update(await fn(flow) or {})
        if route(flow) == "judge":
            break
    await n_judge(flow)


async def run_pipeline(claim, *, rounds=2, depth=3, primary=(), archive=True, on_event=None,
                       use_langgraph=True, mode=None) -> State:
    st = State(claim, primary, on_event)
    mode = (mode or config.MODE).lower()
    limits = httpx.Limits(max_keepalive_connections=20, max_connections=50)
    timeout = httpx.Timeout(connect=8.0, read=15.0, write=8.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=True,
                                 headers={"User-Agent": config.UA}) as c:
        if mode != "agents":
            await run_single(c, st, rounds=rounds, depth=depth, archive=archive)
            return st
        flow: Flow = {"st": st, "c": c, "round": 0, "max_rounds": rounds, "depth": depth, "archive": archive}
        app = None
        if use_langgraph:
            try:
                app = build_langgraph()
            except ImportError:
                st.emit("pipeline", "langgraph not installed; using the sequential runner")
        if app:
            await app.ainvoke(flow, config={"recursion_limit": 25 + 10 * rounds})
        else:
            await run_sequential(flow)
    return st
