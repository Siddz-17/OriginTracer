"""Propagation graph (quotes / links / citations / mentions) and probabilistic origin scoring."""
import math
from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta

from .classify import UTC, norm_url

# ---------------------------------------------- propagation graph & origin scoring
ORIGIN_W = {"timestamp": 0.35, "citation": 0.30, "repost": 0.15, "direct": 0.20}


def build_graph(st):
    ev = st.by_id()
    M = {i: ev[i] for i in st.assign if i in ev}
    uidx = {}
    for i, e in M.items():
        uidx[norm_url(e.url)] = i
        if e.meta.get("resolved_url"):
            uidx[norm_url(e.meta["resolved_url"])] = i
    aidx = {e.meta["at_uri"]: i for i, e in M.items() if e.meta.get("at_uri")}
    edges, anomalies = {}, []

    def add(b, a, kind, w):
        if a == b:
            return
        if M[a].eff_ts > M[b].eff_ts:
            anomalies.append(f"{b} references {a} but is timestamped earlier; edge ignored")
            return
        if (b, a) not in edges or w > edges[(b, a)][1]:
            edges[(b, a)] = (kind, w)

    for b, e in M.items():
        for l in e.meta.get("links", []):
            a = uidx.get(norm_url(l))
            if a:
                add(b, a, "link", 1.0)
        for q in e.meta.get("quotes", []):
            a = aidx.get(q)
            if a:
                add(b, a, "quote", 1.0)
        low = e.text.lower()
        for a, x in M.items():
            au = x.author.lower()
            if a != b and len(au) > 2 and x.source in ("reddit", "bluesky") and (f"@{au}" in low or f"u/{au}" in low):
                add(b, a, "mention", 0.5)
    for u, l in st.cite_edges:
        b, a = uidx.get(norm_url(u)), uidx.get(norm_url(l))
        if a and b:
            add(b, a, "citation", 1.0)
    return M, edges, anomalies


def origin_analysis(st, now=None):
    """OriginScore = timestamp + citation + repost + direct-reference (originality); softmaxed to
    probabilities with explicit mass on 'origin not observed'. Heuristic, not calibrated."""
    M, edges, anomalies = build_graph(st)
    if not M:
        return None
    now = now or datetime.now(UTC)
    cited_by, cites = defaultdict(set), defaultdict(set)
    for (b, a) in edges:
        cited_by[a].add(b)
        cites[b].add(a)

    def descendants(a):
        seen, dq = set(), deque([a])
        while dq:
            for b in cited_by[dq.popleft()]:
                if b not in seen:
                    seen.add(b)
                    dq.append(b)
        return seen

    desc = {i: descendants(i) for i in M}
    maxd = max(len(d) for d in desc.values())
    maxe = max(e.engagement for e in M.values())
    t0, t1 = min(e.eff_ts for e in M.values()), max(e.eff_ts for e in M.values())
    span = (t1 - t0).total_seconds()
    vroot = {v["id"] for v in st.variants if not v.get("parent")}
    pool = {i: e for i, e in M.items() if st.assign[i].get("stance") != "debunk"} or M
    cand = []
    for i, e in pool.items():
        parts = {
            "timestamp": 1.0 if span == 0 else (t1 - e.eff_ts).total_seconds() / span,
            "citation": math.log1p(len(desc[i])) / math.log1p(maxd) if maxd else 0.0,
            "repost": math.log1p(e.engagement) / math.log1p(maxe) if maxe else 0.0,
            "direct": (0.0 if cites[i] else 1.0) * (1.0 if (not vroot or st.assign[i].get("variant") in vroot) else 0.6),
        }
        cand.append((sum(ORIGIN_W[k] * v for k, v in parts.items()), i, parts))
    cand.sort(reverse=True)
    T = 0.12
    z = [math.exp(s / T) for s, _, _ in cand]
    n_match = len(M)
    p_unobs = min(0.6, max(0.05, 0.6 / math.sqrt(n_match)))
    top = M[cand[0][1]]
    if top.source == "gdelt" and top.eff_ts < now - timedelta(days=83):  # near GDELT's 90-day horizon
        p_unobs += 0.2
    p_unobs = min(0.8, p_unobs)
    out = []
    for (s, i, parts), zi in list(zip(cand, z))[:5]:
        e = M[i]
        out.append({"id": i, "score": round(s, 3), "prob": round(zi / sum(z) * (1 - p_unobs), 3), "parts": {k: round(v, 2) for k, v in parts.items()},
                    "source": e.source, "author": e.author, "ts": e.eff_ts, "orig_ts": e.ts, "url": e.url,
                    "text": e.text[:200], "n_descendants": len(desc[i]),
                    "platforms": sorted({M[d].source for d in desc[i]})})
    # human-readable evidence for the top candidate
    t = out[0]
    ti = M[t["id"]]
    bullets = [f"First observed on {t['source']} at {t['ts']:%Y-%m-%d %H:%M UTC}"
               + (f" (Wayback capture pushed this back from {t['orig_ts']:%Y-%m-%d})" if t["ts"] < t["orig_ts"] else "")]
    if t["n_descendants"]:
        bullets.append(f"Referenced (quote/link/citation/mention) by {t['n_descendants']} later items across "
                       f"{len(t['platforms'])} platform(s): {', '.join(t['platforms'])}")
    else:
        bullets.append("No later matched item links to it directly (weakens the citation evidence)")
    news_ts = [e.eff_ts for e in M.values() if e.source in ("news", "gdelt")]
    if news_ts and ti.source not in ("news", "gdelt") and ti.eff_ts < min(news_ts):
        bullets.append(f"Appears before all {len(news_ts)} detected news articles")
    if not cites[t["id"]]:
        bullets.append("Cites no earlier matched item (nothing upstream found)")
    return {"candidates": out, "p_unobserved": round(p_unobs, 3), "bullets": bullets,
            "anomalies": anomalies[:10], "n_nodes": len(M), "n_edges": len(edges),
            "edge_kinds": dict(Counter(k for k, _ in edges.values()))}
