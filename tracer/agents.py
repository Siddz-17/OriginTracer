"""The eight agents. Each is `async def agent(c, st, ...)`: it reads/mutates the shared State.

1 query_expansion    2 source_discovery   3 claim_matcher     4 origin_tracer
5 evidence_collection 6 fact_checker      7 devils_advocate   8 judge
Modules are referenced as `llm.`, `sources.`, `scrape.` so tests can monkeypatch them.
"""
import asyncio
import json
import os
import re
from collections import Counter, defaultdict
from urllib.parse import urlparse

from . import config, llm, scrape, sources
from .classify import SOURCE_WEIGHTS, TERMINAL, classify_url
from .llm import NoJsonReturned
from .mutation import MUTATION_LABELS, refine_mutations
from .provenance import origin_analysis
from .scoring import compute_confidence, source_weight

DEBUNK_RE = re.compile(r"debunk|fact.?check|false|hoax|retract|correction|denied|misleading|no evidence", re.I)


def jsonable(x):
    return json.loads(json.dumps(x, default=lambda o: o.isoformat() if hasattr(o, "isoformat") else str(o)))


# =============================================================== 1. Query expansion
EXPAND_SYS = """You are the Query Expansion agent. Turn a viral claim into many search formulations to
maximise recall across Bluesky, GDELT and news RSS (do not search Reddit).
Return {"paraphrases":[up to 8 distinct rewordings incl. synonyms (e.g. shutting down / discontinuing /
ending / closing); if the claim contains numbers, names or dates add variants with those altered],
"keywords":{"gdelt":[up to 3 queries: 2-4 distinctive keywords each, or one quoted phrase],
"news":[up to 3 short keyword queries]},
"orgs":["organisations or people named in the claim"],
"official_domains":["domains those orgs use for official statements, e.g. apple.com (only if confident)"],
"sec_relevant":true|false  (true only if a publicly listed company's SEC filings could settle the claim)}"""


async def query_expansion(c, st):
    if st.follow_up:  # devil's advocate asked for more searching
        plan = {k: list(st.follow_up.get(k, []) or []) for k in sources.SOURCES}
        st.follow_up = None
        st.emit("expansion", "using the devil's-advocate follow-up queries")
    else:
        out = await llm.ask_json(EXPAND_SYS, f"Claim: {st.claim}", "fast")
        para = [p for p in out.get("paraphrases", []) if isinstance(p, str)][:8]
        if st.claim not in para:
            para.insert(0, st.claim)
        kw = out.get("keywords") or {}
        gd = (kw.get("gdelt") or [])[:3] or [" ".join(sorted(re.findall(r"\w{4,}", st.claim), key=len)[-4:])]
        st.base_query = gd[0]
        bsky_qs = (kw.get("news") or [])[:2] + gd[:2] or para[:2]
        plan = {"bluesky": bsky_qs[:3], "gdelt": gd,
                "news": (kw.get("news") or [])[:3] or [st.claim],
                "duckduckgo": (kw.get("news") or [])[:2] or [st.claim][:2]}
        if "reddit" in sources.SOURCES:
            plan["reddit"] = para[:3]
        for src, suffixes in sources.CONTRA.items():  # contradiction search
            if plan.get(src):
                plan[src] = plan[src][:3] + [f"{plan[src][0]} {s}" for s in suffixes]
        st.paraphrases, st.orgs = para, [o for o in out.get("orgs", []) if isinstance(o, str)][:5]
        st.official_domains = [d.lower().removeprefix("www.") for d in out.get("official_domains", [])
                               if isinstance(d, str) and re.fullmatch(r"[a-z0-9.-]+\.[a-z]{2,}", d.lower())][:4]
        st.sec_relevant = bool(out.get("sec_relevant"))
    st.plan = plan
    st.emit("expansion", f"{sum(len(v) for v in plan.values())} queries across {len(plan)} sources; "
                         f"{len(st.paraphrases)} paraphrases")


# ================================================================ 2. Source discovery
async def source_discovery(c, st):
    jobs = []
    for src, qs in st.plan.items():
        for q in (qs or [])[:6]:
            if src in sources.SOURCES and (src, q) not in st.queries_used:
                st.queries_used.add((src, q))
                jobs.append((src, q, sources.SOURCES[src](c, q)))
    results = await asyncio.gather(*(j[2] for j in jobs), return_exceptions=True)
    for (src, q, _), res in zip(jobs, results):
        if isinstance(res, Exception):
            st.emit("discovery", f"! {src} '{q[:40]}' failed: {type(res).__name__}: {str(res)[:120]}")
            continue
        for raw in res:
            st.add(raw)
    cnt = Counter(e.source for e in st.pool.values())
    st.discovery = {"reddit_posts": cnt["reddit"], "bluesky_posts": cnt["bluesky"],
                    "news_articles": cnt["news"] + cnt["gdelt"]}
    if cnt["reddit"] == 0 and cnt["bluesky"] == 0 and (cnt["news"] + cnt["gdelt"]) > 0:
        st.emit("discovery", "social sources unavailable/empty; prioritizing global news coverage as main priority")
    st.emit("discovery", json.dumps(st.discovery))


# =================================================================== 3. Claim matcher
MATCHER_SYS = """You are the Claim Matcher. You get a claim and a chronological list of posts/articles.
1. Decide which items genuinely assert, repeat, or report on the claim (not merely share keywords).
   A debunk/correction/fact-check item is related too - give it stance 'debunk'.
2. Group wordings of the same narrative into variants. Items telling the SAME story that differ only in a
   number, name or date are variants of the same root (set parent), never separate roots.
3. Extract WHO / WHAT / WHEN / WHERE / NUMBER per variant. V0 = closest reconstruction of the original form.
4. Label each variant's mutation relative to its parent, using exactly one of: __LABELS__.
Return {"variants":[{"id":"V0","text":"canonical wording keeping exact numbers/names/dates",
"entities":{"who":"","what":"","when":"","where":"","number":""},"mutation":"original","parent":null,"notes":""}],
"assignments":{"E1":{"variant":"V0","stance":"assert|report|debunk|joke"},"E2":null}}
Use null for unrelated items. Be skeptical; do not invent items."""
MATCHER_SYS = MATCHER_SYS.replace("__LABELS__", " | ".join(MUTATION_LABELS))


# Max new items the matcher will classify per round (keeps prompt size manageable).
_MATCHER_NEW_CAP = int(os.getenv("MATCHER_NEW_CAP", "40"))
_MATCHER_CHUNK = int(os.getenv("MATCHER_CHUNK", "25"))


def sample_for_matcher(st, already_assigned: set):
    """Return items not yet classified, newest-first, capped at _MATCHER_NEW_CAP."""
    items = st.sorted()
    new_items = [e for e in items if e.id not in already_assigned]
    # prioritise: news/gdelt first (most likely to contain the claim), then by engagement
    news = [e for e in new_items if e.source in ("news", "gdelt")]
    other = [e for e in new_items if e.source not in ("news", "gdelt")]
    debunks = [e for e in new_items if DEBUNK_RE.search(e.text)]
    seen, out = set(), []
    for e in news[:60] + other[:20] + debunks[:20]:
        if e.id not in seen:
            seen.add(e.id)
            out.append(e)
    return sorted(out, key=lambda e: e.eff_ts)[:_MATCHER_NEW_CAP]


async def _classify_chunk(chunk, existing_variants, claim):
    """Run one matcher chunk and return (variants_list, assignments_dict)."""
    context = f"Claim: {claim}\n\n"
    if existing_variants:
        context += f"Existing variants:\n{json.dumps(existing_variants)}\n\n"
    context += f"Items:\n{json.dumps(chunk)}"
    try:
        out = await llm.ask_json(MATCHER_SYS, context, "fast", 4000)
    except NoJsonReturned as e:
        return [], {}
    return out.get("variants", []), out.get("assignments") or {}


async def claim_matcher(c, st):
    already_assigned = set(st.assign.keys())  # IDs classified in prior rounds
    new_evs = sample_for_matcher(st, already_assigned)
    if not new_evs:
        st.emit("matcher", "no new items to classify this round")
        return
    items = [e.brief() for e in new_evs]
    st.emit("matcher", f"classifying {len(items)} new items (skipping {len(already_assigned)} already assigned)")
    ids = {e.id for e in st.pool.values()}
    # Build chunks and run all in parallel
    chunks = [items[i:i + _MATCHER_CHUNK] for i in range(0, max(len(items), 1), _MATCHER_CHUNK)]
    tasks = [_classify_chunk(chunk, list(st.variants), st.claim) for chunk in chunks if chunk]
    results = await asyncio.gather(*tasks)
    for new_variants, assignments in results:
        for v in new_variants:
            if isinstance(v, dict) and v.get("id"):
                if not any(x.get("id") == v["id"] for x in st.variants):
                    st.variants.append(v)
        for k, v in assignments.items():
            if v and k in ids and isinstance(v, dict):
                st.assign[k] = v
    refine_mutations(st.variants)
    st.emit("matcher", f"{len(st.variants)} variants, {len(st.assign)} matched items total")


# =================================================================== 4. Origin tracer
async def archive_pass(c, st, limit=25):
    ev = st.by_id()
    targets = sorted((ev[i] for i in st.assign if i in ev and ev[i].url not in st.archive_tried
                      and "news.google.com" not in ev[i].url), key=lambda e: e.ts)[:limit]
    sem = asyncio.Semaphore(5)

    async def one(e):
        async with sem:
            st.archive_tried.add(e.url)
            try:
                e.first_capture = await sources.wayback_first(c, e.url)
            except Exception:
                pass

    await asyncio.gather(*(one(e) for e in targets))
    moved = sum(1 for e in targets if e.first_capture and e.first_capture < e.ts)
    st.emit("origin", f"Wayback: checked {len(targets)} URLs, {moved} pushed earlier")


async def origin_tracer(c, st, archive=True):
    if archive:
        await archive_pass(c, st)
    st.origin = origin_analysis(st)
    ev = st.by_id()
    matched = sorted((ev[i] for i in st.assign if i in ev), key=lambda e: e.eff_ts)
    st.timeline = [{"id": e.id, "time": e.eff_ts.isoformat(), "source": e.source, "author": e.author,
                    "variant": st.assign[e.id].get("variant"), "stance": st.assign[e.id].get("stance"),
                    "text": e.text[:160], "url": e.url} for e in matched][:200]
    agg = defaultdict(lambda: [0, 0])
    for e in matched:
        agg[(e.source, e.author)][0] += 1
        agg[(e.source, e.author)][1] += e.engagement
    st.amplifiers = [{"source": s, "author": a, "items": n, "engagement": en} for (s, a), (n, en) in
                     sorted(agg.items(), key=lambda x: -(x[1][0] + x[1][1] / 50))[:10]]
    if st.origin:
        top = st.origin["candidates"][0]
        st.emit("origin", f"likely origin {top['source']}/{top['author']} p={top['prob']} "
                          f"(p_unobserved={st.origin['p_unobserved']})")
    else:
        st.emit("origin", "no matched items; no origin")


# ============================================================ 5. Evidence collection
def _official_type(st, url):
    typ = classify_url(url)
    if typ in TERMINAL:
        return typ
    host = urlparse(url).netloc.lower().removeprefix("www.")
    if any(host == d or host.endswith("." + d) for d in st.official_domains) and \
            any(o.lower().split()[0] in host for o in st.orgs if o):
        return "press_release"  # organisation's own domain, matched to a named org
    return typ


async def _official_hits(c, st):
    q0 = st.base_query
    doms = st.official_domains[:4]
    if not doms or not q0:
        return
    res = await asyncio.gather(*(sources.gdelt(c, f"domain:{d} {q0}") for d in doms), return_exceptions=True)
    urls = [raw["url"] for r in res if not isinstance(r, Exception) for raw in r[:2]]
    pages = await asyncio.gather(*(scrape.fetch_page(c, u) for u in urls), return_exceptions=True)
    for u, p in zip(urls, pages):
        if not isinstance(p, Exception):
            st.add_ref("P", u, p.title or u, p.text, _official_type(st, u), via=p.via,
                       note="found via GDELT domain filter; domain suggested by LLM")
    st.emit("evidence", f"official-domain pages: {sum(1 for p in pages if not isinstance(p, Exception))}")


async def _sec_search(c, st):
    """SEC EDGAR full-text search (no key; needs a contact in the User-Agent). Best-effort."""
    if not st.sec_relevant:
        return
    q = re.sub(r'[()"]|\bOR\b', " ", st.base_query or st.claim).strip()[:150]
    try:
        r = await c.get("https://efts.sec.gov/LATEST/search-index", params={"q": q, "forms": "8-K,10-K,10-Q,6-K"},
                        headers={"User-Agent": config.UA})
        hits = r.json().get("hits", {}).get("hits", [])[:3]
        for h in hits:
            s = h.get("_source", {})
            adsh, fname = s.get("adsh") or h["_id"].split(":")[0], h["_id"].split(":")[-1]
            cik = int((s.get("ciks") or ["0"])[0])
            u = f"https://www.sec.gov/Archives/edgar/data/{cik}/{adsh.replace('-', '')}/{fname}"
            p = await scrape.fetch_page(c, u)
            st.add_ref("P", u, f"SEC {s.get('form')} {', '.join(s.get('display_names', [])[:1])}", p.text,
                       "government", via=p.via, note="SEC EDGAR full-text hit")
        st.emit("evidence", f"SEC EDGAR hits: {len(hits)}")
    except Exception as ex:
        st.emit("evidence", f"! SEC search failed: {type(ex).__name__}")


async def _trace_citations(c, st, max_depth=3, budget=25):
    ev = st.by_id()
    matched = sorted((ev[i] for i in st.assign if i in ev), key=lambda e: e.eff_ts)
    seeds, n_news = [], 0
    news_candidates = [e for e in matched if e.source in ("news", "gdelt")][:16]

    async def _resolve_news(e):
        u = e.url
        if "news.google.com" in u:
            ru = e.meta.get("resolved_url") or await scrape.resolve_redirect(u, c)
            if ru:
                e.meta["resolved_url"] = ru
                return ru
            return None
        return u

    resolved_news = await asyncio.gather(*(_resolve_news(e) for e in news_candidates), return_exceptions=True)
    for ru in resolved_news:
        if ru and not isinstance(ru, Exception):
            seeds.append((ru, [ru]))
            n_news += 1

    for e in matched:
        for l in e.meta.get("links", []):
            if classify_url(l) != "social":
                seeds.append((l, [e.url, l]))
                st.cite_edges.append((e.url, l))
    level, seen = [], set()
    for u, p in seeds:
        if u not in st.fetched and u not in seen:
            seen.add(u)
            level.append((u, p))
    for depth in range(max_depth):
        if not level or budget <= 0:
            break
        batch, level = level[:budget], level[budget:]
        budget -= len(batch)
        pages = await asyncio.gather(*(scrape.fetch_page(c, u) for u, _ in batch), return_exceptions=True)
        nxt = []
        for (u, path), pg in zip(batch, pages):
            st.fetched.add(u)
            kind = _official_type(st, u)
            if isinstance(pg, Exception):
                st.chains.append({"path": path, "resolved": kind in TERMINAL, "end": kind, "note": "unreachable"})
                continue
            if kind in TERMINAL:
                st.add_ref("P", u, pg.title or u, pg.text, kind, via=pg.via)
                st.chains.append({"path": path, "resolved": True, "end": kind})
                continue
            host = urlparse(u).netloc
            cands = []
            for l in pg.links:
                k = classify_url(l)
                if k == "social" or (urlparse(l).netloc == host and k not in TERMINAL):
                    continue
                cands.append((SOURCE_WEIGHTS.get(k, 0.4), l))
            cands.sort(reverse=True)
            for _, l in cands[:6]:
                st.cite_edges.append((u, l))
                if l not in st.fetched and l not in seen:
                    seen.add(l)
                    nxt.append((l, path + [l]))
            if not cands:
                st.chains.append({"path": path, "resolved": False, "end": kind, "note": "no outbound citations"})
            elif depth == max_depth - 1:
                st.chains.append({"path": path, "resolved": False, "end": kind, "note": "depth limit"})
        level = nxt + level
    res = sum(1 for ch in st.chains if ch["resolved"])
    st.emit("evidence", f"{len(st.chains)} citation chains, {res} reached a primary-type source")


async def evidence_collection(c, st, depth=3):
    await _official_hits(c, st)
    await _sec_search(c, st)
    await _trace_citations(c, st, depth)
    todo = [u for u in st.user_primary if u not in st.fetched]
    pages = await asyncio.gather(*(scrape.fetch_page(c, u) for u in todo), return_exceptions=True)
    for u, p in zip(todo, pages):
        st.fetched.add(u)
        if isinstance(p, Exception):
            st.emit("evidence", f"! could not fetch user-supplied {u}")
        else:
            t = classify_url(u)
            st.add_ref("P", u, p.title or u, p.text, t if t in TERMINAL else "institutional", via=p.via,
                       note="user-supplied primary source")
    sem = asyncio.Semaphore(5)

    async def cap(r):  # archived-page history: when did the Wayback Machine first see this primary page?
        async with sem:
            try:
                ts = await sources.wayback_first(c, r["url"])
                r["first_capture"] = ts.isoformat() if ts else None
            except Exception:
                r["first_capture"] = None

    await asyncio.gather(*(cap(r) for r in st.refs if r["kind"] == "P" and "first_capture" not in r))
    st.refs.sort(key=lambda r: -r["weight"])
    prim = sum(1 for r in st.refs if r["weight"] >= 0.75)
    st.emit("evidence", f"{len(st.refs)} reference docs ranked ({prim} primary-type)")


# ================================================================== 6. Fact checker
FC_PLAN_SYS = """You are the Fact Checker. Break the claim into atomic, checkable sub-claims with
dependencies (e.g. S3 only matters if S1 is true). Plan Wikipedia lookups for background. If critiques from a
prior round are given, address them.
Return {"subclaims":[{"id":"S1","text":"...","depends_on":[]}],"wikipedia":["query"],
"primary_hint":"which primary source would settle this (court docket, agency statement, dataset...)"}
(max 5 subclaims, 3 wikipedia queries)."""

FC_VERDICT_SYS = """You are the Fact Checker. Judge each sub-claim ONLY from the provided sources.
Each reference doc has a type and weight (court 1.0, government/dataset 0.95, peer-reviewed 0.9,
press release 0.75, news 0.5, Wikipedia 0.4, social 0.2). A dissenting high-weight source outweighs many
low-weight ones. E* ids are news/social items: they show what was REPORTED, never proof of truth.
If nothing settles a sub-claim, verdict is 'unverifiable'. Cite ids.
For each sub-claim list "supporting" = ids backing YOUR verdict, "dissenting" = ids pointing the other way.
Also list context a reader would need that the claim omits.
Return {"subclaims":[{"id":"S1","verdict":"supported|refuted|misleading|unverifiable","reasoning":"...",
"supporting":["P1"],"dissenting":[]}],
"missing_context":["..."],
"overall":{"verdict":"true|mostly_true|misleading|mostly_false|false|unverifiable","summary":"..."}}"""


def evidence_cards(st, ids):
    ev, out = st.by_id(), []
    for i in dict.fromkeys(ids):
        r = next((r for r in st.refs if r.get("label") == i), None)
        if r:
            out.append({"id": i, "type": r["type"], "weight": r["weight"], "url": r["url"], "title": r["title"]})
        elif i in ev:
            e = ev[i]
            out.append({"id": i, "type": e.source, "weight": source_weight(st, i), "url": e.url, "title": e.text[:100]})
    return out


async def fact_checker(c, st, critique=None):
    ctx = f"Claim: {st.claim}"
    if critique:
        ctx += f"\n\nCritique from last round to address:\n{json.dumps(critique)[:3000]}"
    try:
        plan = await llm.ask_json(FC_PLAN_SYS, ctx, "smart")
    except NoJsonReturned:
        st.emit("factcheck", "! fact-checker plan returned no JSON; marking run degraded")
        st.fact = {"overall": {"verdict": "unverifiable", "summary": "Fact-checker plan failed (no JSON)"},
                   "subclaims": [], "missing_context": [], "scores": compute_confidence(st)}
        return
    qs = [q for q in plan.get("wikipedia", [])[:3] if ("wiki", q) not in st.queries_used]
    for q in qs:
        st.queries_used.add(("wiki", q))
    res = await asyncio.gather(*(sources.wikipedia(c, q) for q in qs), return_exceptions=True)
    for r in res:
        if isinstance(r, Exception):
            continue
        for p in r:
            st.add_ref("W", p["url"], p["title"], p["text"], "reference")
    cnt = Counter()
    for r in st.refs:
        cnt[r["kind"]] += 1
        r["label"] = f"{r['kind']}{cnt[r['kind']]}"
    docs = [{"id": r["label"], "type": r["type"], "weight": r["weight"], "url": r["url"], "title": r["title"],
             "text": r["text"][:2500]} for r in st.refs]
    ev = st.by_id()
    matched = [(ev[i], a) for i, a in st.assign.items() if i in ev]
    contra = [e.brief(240) for e, a in matched if a.get("stance") == "debunk"][:20]
    reported = [e.brief(240) for e, a in matched if a.get("stance") != "debunk" and e.source in ("news", "gdelt")][:30]
    st.emit("factcheck", f"judging with {len(docs)} reference docs, {len(contra)} debunk-stance items")
    try:
        st.fact = await llm.ask_json(FC_VERDICT_SYS, json.dumps({
            "claim": st.claim, "subclaims": plan.get("subclaims", []), "reference_docs": docs,
            "news_items_reporting_claim": reported, "debunk_or_correction_items": contra}), "smart", 5000)
    except NoJsonReturned:
        st.emit("factcheck", "! fact-checker verdict returned no JSON; marking run degraded")
        st.fact = {"overall": {"verdict": "unverifiable", "summary": "Fact-checker verdict failed (no JSON)"},
                   "subclaims": [], "missing_context": []}
    st.fact["plan"] = plan.get("subclaims", [])
    st.fact["primary_hint"] = plan.get("primary_hint", "")
    st.fact["missing_context"] = [m for m in st.fact.get("missing_context", []) if isinstance(m, str)]
    st.fact["scores"] = compute_confidence(st)
    subs = st.fact.get("subclaims", [])
    st.fact["supporting_evidence"] = evidence_cards(st, [i for s in subs for i in s.get("supporting", [])])
    st.fact["contradicting_evidence"] = evidence_cards(st, [i for s in subs for i in s.get("dissenting", [])])
    st.emit("factcheck", f"verdict {st.fact.get('overall', {}).get('verdict')} "
                         f"confidence {st.fact['scores']['confidence']}")


# =============================================================== 7. Devil's advocate
DA_SYS = """You are the Devil's Advocate. Break the Fact Checker's verdict and the origin analysis.
Attack: (a) weak/missing primary sources; (b) verdicts resting on Wikipedia/news alone; (c) 'earliest seen'
being an artifact of API coverage (GDELT ~3 months, ranking limits, deleted posts); (d) skipped or bundled
sub-claims; (e) could the source article be OUTDATED or superseded?; (f) could the statement be SATIRE or
parody?; (g) was CONTEXT REMOVED (quote cut, date dropped)?; (h) could ANOTHER INTERPRETATION fit the same
evidence?; (i) are propagation-graph edges real references or just co-occurrence?
Return {"verdict_holds":true|false,
"attacks":[{"target":"origin|variant|verdict|source","argument":"...","severity":"low|medium|high"}],
"alternative_explanations":["..."],"missing_evidence":["..."],
"follow_up":{"gdelt":[],"bluesky":[],"news":[]}}
Only set verdict_holds=true if no high-severity attack remains unanswered."""


async def devils_advocate(c, st):
    oa = origin_analysis(st)
    summary = {
        "claim": st.claim,
        "origin": ({k: v for k, v in oa.items() if k != "anomalies"} if oa else None),
        "variants": st.variants, "pool_size": len(st.pool), "matched": len(st.assign),
        "citation_chains": [{"path": ch["path"], "end": ch["end"], "resolved": ch["resolved"]} for ch in st.chains[:15]],
        "fact_check": {k: v for k, v in st.fact.items() if k not in ("plan", "supporting_evidence", "contradicting_evidence")},
        "reference_docs": [{"id": r["label"], "type": r["type"], "url": r["url"]} for r in st.refs],
    }
    try:
        out = await llm.ask_json(DA_SYS, json.dumps(summary, default=str), "adversary", 4000)
    except NoJsonReturned:
        st.emit("devil", "! devil's advocate returned no JSON; skipping this critique")
        return {"verdict_holds": True, "attacks": [], "alternative_explanations": [], "missing_evidence": []}
    st.critiques.append(out)
    atk = out.get("attacks", [])
    hi, med = sum(a.get("severity") == "high" for a in atk), sum(a.get("severity") == "medium" for a in atk)
    sc = st.fact.setdefault("scores", {})
    sc["after_critique"] = round(max(0.0, sc.get("confidence", 0) - 0.10 * hi - 0.04 * med), 2)
    st.follow_up = None if out.get("verdict_holds") else out.get("follow_up")
    st.emit("devil", f"holds={out.get('verdict_holds')} attacks={len(atk)} (high {hi}); "
                     f"confidence {sc.get('confidence')} -> {sc['after_critique']}")
    return out


# ======================================================================== 8. Judge
JUDGE_SYS = """You are the Judge. You receive the full case file: origin analysis, mutation graph, the Fact
Checker's sub-claim verdicts with evidence weights, and the Devil's Advocate's attacks.
Reconcile them into a final verdict. You may keep or change the Fact Checker's verdict label, but only on the
basis of the case file (no outside facts). If a high-severity attack is unresolved, prefer a more cautious
label ('misleading' or 'unverifiable') over a confident true/false. Be explicit about what is NOT known.
Return {"verdict":"true|mostly_true|misleading|mostly_false|false|unverifiable",
"rationale":"3-5 sentences","origin_statement":"one sentence on what is known/unknown about the origin",
"changed_from_factchecker":true|false,"caveats":["..."]}"""


def mutation_graph(st):
    ev, per = st.by_id(), defaultdict(list)
    for i, a in st.assign.items():
        if i in ev:
            per[a.get("variant")].append(ev[i])
    nodes, edges = [], []
    for v in st.variants:
        items = sorted(per.get(v["id"], []), key=lambda e: e.eff_ts)
        nodes.append({"id": v["id"], "label": v.get("mutation"), "text": v.get("text"), "notes": v.get("notes"),
                      "entities": v.get("entities"), "n_items": len(items),
                      "first_seen": items[0].eff_ts if items else None,
                      "first_platform": items[0].source if items else None})
        if v.get("parent"):
            edges.append({"from": v["parent"], "to": v["id"], "label": v.get("mutation"), "diff": v.get("diff", {})})
    return {"nodes": nodes, "edges": edges}


def build_result(st):
    oa = origin_analysis(st) or {}
    sc, J = st.fact.get("scores", {}), st.judge
    conf = sc.get("after_critique", sc.get("confidence", 0.0))
    if J.get("changed_from_factchecker"):
        conf = max(0.0, round(conf - 0.05, 2))
    if not sc.get("primary_sources"):
        conf = min(conf, 0.6)
    cands = oa.get("candidates", [])
    st.result = jsonable({
        "claim": st.claim,
        "verdict": J.get("verdict") or st.fact.get("overall", {}).get("verdict"),
        "confidence": round(conf, 2),
        "rationale": J.get("rationale"),
        "caveats": J.get("caveats", []),
        "origin": {"likely": cands[0] if cands else None, "probability": cands[0]["prob"] if cands else None,
                   "p_unobserved": oa.get("p_unobserved"), "evidence": oa.get("bullets", []),
                   "first_platform": cands[0]["source"] if cands else None, "candidates": cands,
                   "propagation": {k: oa.get(k) for k in ("n_nodes", "n_edges", "edge_kinds")},
                   "statement": J.get("origin_statement")},
        "timeline": st.timeline,
        "mutation_graph": mutation_graph(st),
        "amplifiers": st.amplifiers,
        "fact_check": {"subclaims": st.fact.get("subclaims", []),
                       "supporting_evidence": st.fact.get("supporting_evidence", []),
                       "contradicting_evidence": st.fact.get("contradicting_evidence", []),
                       "missing_context": st.fact.get("missing_context", [])},
        "scores": {k: v for k, v in sc.items() if k != "per_subclaim"} | {"per_subclaim": sc.get("per_subclaim", {})},
        "citation_chains": st.chains,
        "critique": st.critiques[-1] if st.critiques else None,
        "discovery": st.discovery,
    })
    return st.result


async def judge(c, st):
    oa = origin_analysis(st)
    case = {"claim": st.claim,
            "origin": ({k: v for k, v in oa.items() if k != "anomalies"} if oa else None),
            "mutation_graph": mutation_graph(st),
            "fact_check": {k: v for k, v in st.fact.items() if k not in ("plan", "supporting_evidence", "contradicting_evidence")},
            "critiques": st.critiques[-2:], "sources": [{"id": r["label"], "type": r["type"], "weight": r["weight"]} for r in st.refs]}
    try:
        st.judge = await llm.ask_json(JUDGE_SYS, json.dumps(case, default=str), "smart", 3000)
    except NoJsonReturned:
        st.emit("judge", "! judge returned no JSON; using fact-checker verdict directly")
        st.judge = {"verdict": st.fact.get("overall", {}).get("verdict", "unverifiable"),
                    "rationale": "Judge call failed; fact-checker verdict used as fallback.",
                    "origin_statement": "", "changed_from_factchecker": False, "caveats": ["Judge LLM call failed"]}
    build_result(st)
    st.emit("judge", f"final verdict {st.result['verdict']} confidence {st.result['confidence']}")
