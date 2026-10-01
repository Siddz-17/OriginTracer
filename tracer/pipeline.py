"""Pipeline wiring: LangGraph if installed, otherwise an equivalent sequential runner.

expansion -> discovery -> matcher -> origin -> evidence -> factcheck -> devil --(holds | out of rounds)--> judge
                 ^                                                      |
                 +------------------- follow-up queries -----------------+
"""
from typing import Any, TypedDict

import httpx

from . import agents, config
from .models import State


class Flow(TypedDict, total=False):
    st: State
    c: Any
    round: int
    max_rounds: int
    depth: int
    archive: bool


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
                       use_langgraph=True) -> State:
    st = State(claim, primary, on_event)
    limits = httpx.Limits(max_keepalive_connections=20, max_connections=50)
    timeout = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=True, headers={"User-Agent": config.UA}) as c:
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
