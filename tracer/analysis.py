"""Single-call consolidated analysis.

Discovery, matching, origin scoring, citation tracing and evidence ranking all run locally first (no LLM).
This module then makes exactly ONE LLM request that does the work of the old fact-checker, devil's advocate
and judge agents together, over the top-ranked evidence only. Code (not the model) then computes confidence.
"""
import json
import os
from datetime import datetime

from . import llm
from .agents import evidence_cards
from .classify import UTC
from .llm import NoJsonReturned
from .local import rank_evidence
from .scoring import (compute_confidence, consensus_verdict, fact_check_consensus, reconcile_verdict,
                      verdict_side)

VERDICTS = ("true", "mostly_true", "misleading", "mostly_false", "false", "unverifiable")
SUB_VERDICTS = ("supported", "refuted", "misleading", "unverifiable")

ANALYSIS_SYS = """You are a meticulous fact-checker and provenance analyst. You get ONE viral claim plus an
evidence pack that was already collected and ranked from reputable news outlets, fact-checkers, official /
primary documents, Wikipedia and social posts. Do the whole analysis in one pass:

1. Split the claim into 1-5 atomic, checkable sub-claims (who/what/when/where/how many).
2. Judge each sub-claim from the evidence. Each item has a type and weight: court 1.0, government/dataset
   0.95, peer_reviewed 0.9, press_release 0.75, fact_check 0.7, news 0.5, reference(Wikipedia) 0.5,
   social 0.2. A credible fact-check or primary document outweighs many low-weight repeats. Headlines that
   merely REPEAT the claim show it was reported, not that it is true. Reputable outlets reporting that an
   event happened (several independent ones) IS evidence it happened. A fact-check rating counts as strong
   evidence for its verdict. Watch for: outdated stories, satire, quotes cut from context, a number / date /
   name changed from the original, and claims that bundle a true part with a false part. Check SCOPE: an item
   only refutes the claim if it addresses the same thing ("largest landowner" is not "largest farmland
   owner"; "visible from the Moon" is not "visible from space"; a fact-check of a different quote by the same
   person says nothing about this claim). Prefer the most recent reputable evidence when facts change over time.
   RECENCY: compare every item's date with "today". For current or developing events (resignations,
   retirements, deaths, results, launches, policy changes) the NEWEST reputable reports decide: an older report
   ("X says he is not planning to retire") does not refute a newer one ("X retires"); say which is latest.
   If the newest evidence is older than the claim's event could be, the claim may be too recent to verify.
3. Background knowledge: you may use well-established general facts for context, but the evidence pack wins
   over your memory (events after your training cutoff are real if reputable evidence reports them). Never
   invent sources; cite only ids from the pack. If nothing settles a sub-claim, it is 'unverifiable'.
4. Attack your own verdict like a devil's advocate (weak sources, coverage gaps, alternative readings),
   then give the final verdict, cautious if a high-severity attack is unresolved.
5. Review the item stances: list ids that actually DEBUNK/CORRECT the claim and ids that are NOT about this
   claim at all (keyword coincidences), so the origin analysis can be corrected.
6. Review origin_analysis.candidates (earliest items found that seem to state the claim). List in
   origin_rejected_ids every candidate that does NOT itself assert or report this claim: a preview of a
   future event, background on the same topic, a different event with the same names, or a debunk.
   Fact-checks reacting to a claim are never its origin. If an item's date is obviously before the event it
   describes could have happened, reject it too.
7. If image_provenance is present, the claim is about that image. An image whose earliest known appearance or
   EXIF capture date is BEFORE the event the claim describes is being reused out of context (say so, and
   treat the claim as misleading or false); 'variant' matches mean edited or cropped versions exist.

Return exactly this JSON shape:
{"subclaims":[{"id":"S1","text":"atomic sub-claim","verdict":"supported|refuted|misleading|unverifiable",
  "reasoning":"1-3 sentences citing ids","for_verdict":["ids whose content backs YOUR verdict on this
  sub-claim, e.g. the debunks when the verdict is refuted"],"against_verdict":["ids pointing the other way"]}],
 "verdict":"true|mostly_true|misleading|mostly_false|false|unverifiable",
 "rationale":"3-5 sentences: what the best evidence says and why",
 "missing_context":["facts a reader needs that the claim omits"],
 "origin_statement":"one sentence on where/when the claim seems to have started and what is unknown",
 "self_critique":{"attacks":[{"target":"origin|variant|verdict|source","argument":"...","severity":"low|medium|high"}],
   "alternative_explanations":["..."],"missing_evidence":["what source would settle it"],"verdict_holds":true},
 "debunk_ids":["E.."],"irrelevant_ids":["E.."],"origin_rejected_ids":["E.."],
 "caveats":["..."],"primary_hint":"which primary source would settle this"}"""


def _pack(st, items):
    oa = st.origin or {}
    top = (oa.get("candidates") or [None])[0]
    pack = {
        "claim": st.claim,
        "today": datetime.now(UTC).strftime("%Y-%m-%d"),
        "evidence": [{k: v for k, v in {
            "id": x["id"], "type": x["kind"], "weight": x["weight"], "outlet": x["outlet"],
            "reputable": x["tier"], "date": x["date"], "stance_guess": x.get("stance"),
            "fact_check_rating": x.get("rating"),
            "title": x["title"], "text": x["text"] if x["text"][:150] != x["title"][:150] else None,
            "url": x["url"]}.items() if v not in (None, "", [])} for x in items],
        "origin_analysis": None if not top else {
            "candidates": [{"id": c["id"], "source": c["source"], "author": c["author"],
                            "time": str(c["ts"])[:16], "prob": c["prob"], "text": c["text"][:200]}
                           for c in oa.get("candidates", [])[:5]],
            "p_origin_unobserved": oa.get("p_unobserved"), "notes": oa.get("bullets", [])[:4]},
        "variants": [{"id": v["id"], "text": v.get("text", "")[:200], "mutation": v.get("mutation"),
                      "parent": v.get("parent"), "diff": v.get("diff")} for v in st.variants[:6]],
        "image_provenance": None if not st.image or st.image.get("error") else {
            "earliest_appearance": st.image.get("earliest"), "exif_capture": st.image.get("exif_date"),
            "exif": st.image.get("exif"), "ocr_text": st.image.get("ocr_text"),
            "matches": [{k: m.get(k) for k in ("site", "first_seen", "relation", "title")}
                        for m in st.image.get("matches", [])[:8]],
            "notes": st.image.get("notes")},
        "fact_check_ratings": {d: v["rating"] for d, v in fact_check_consensus(st)["publishers"].items()},
        "coverage": {"items_found": len(st.pool), "items_matched": len(st.assign),
                     "citation_chains_to_primary": sum(1 for c in st.chains if c.get("resolved"))},
    }
    return json.dumps(pack, ensure_ascii=False, default=str)


def _clean(out, valid_ids):
    """Validate the model's JSON against the schema and the ids that really exist."""
    subs = []
    for i, s in enumerate(out.get("subclaims") or []):
        if not isinstance(s, dict):
            continue
        v = str(s.get("verdict", "unverifiable")).lower().strip()
        subs.append({"id": str(s.get("id") or f"S{i + 1}"), "text": str(s.get("text", ""))[:400],
                     "verdict": v if v in SUB_VERDICTS else "unverifiable",
                     "reasoning": str(s.get("reasoning", ""))[:1200],
                     "supporting": [x for x in s.get("for_verdict", s.get("supporting")) or [] if x in valid_ids],
                     "dissenting": [x for x in s.get("against_verdict", s.get("dissenting")) or []
                                    if x in valid_ids]})
    verdict = str(out.get("verdict", "")).lower().strip().replace(" ", "_")
    crit = out.get("self_critique") if isinstance(out.get("self_critique"), dict) else {}
    crit = {"verdict_holds": bool(crit.get("verdict_holds", True)),
            "attacks": [a for a in crit.get("attacks") or [] if isinstance(a, dict)],
            "alternative_explanations": [a for a in crit.get("alternative_explanations") or [] if isinstance(a, str)],
            "missing_evidence": [a for a in crit.get("missing_evidence") or [] if isinstance(a, str)]}
    strs = lambda k: [x for x in out.get(k) or [] if isinstance(x, str)]
    return {"subclaims": subs, "verdict": verdict if verdict in VERDICTS else "unverifiable",
            "rationale": str(out.get("rationale", "")), "missing_context": strs("missing_context"),
            "origin_statement": str(out.get("origin_statement", "")), "critique": crit,
            "debunk_ids": strs("debunk_ids"), "irrelevant_ids": strs("irrelevant_ids"),
            "origin_rejected_ids": strs("origin_rejected_ids"),
            "caveats": strs("caveats"), "primary_hint": str(out.get("primary_hint", ""))}


def _local_fallback(st, items, err):
    """No LLM available: fall back on the published ratings of independent fact-checkers when at least two
    agree; otherwise report the ranked evidence honestly instead of guessing a verdict."""
    fc = [x for x in items if x["kind"] == "fact_check" or x.get("stance") == "debunk"]
    cons = fact_check_consensus(st)
    verdict = consensus_verdict(cons) or "unverifiable"
    subs = []
    if verdict != "unverifiable":
        ids = [v["id"] for v in cons["publishers"].values()
               if {"false": "false", "satire": "false", "misleading": "mixed", "true": "true"}[v["rating"]]
               == cons["majority"]]
        subs = [{"id": "S1", "text": st.claim, "verdict": {"false": "refuted", "true": "supported"}.get(
                    cons["majority"], "misleading"),
                 "reasoning": "Rated by independent fact-checkers: " + ", ".join(
                     f"{d} ({v['rating']})" for d, v in cons["publishers"].items()),
                 "supporting": ids, "dissenting": []}]
    rationale = (f"The LLM analysis step failed ({err}). "
                 + (f"Verdict taken from the agreeing ratings of {len(subs[0]['supporting'])} independent "
                    f"fact-checkers. " if subs else "Verdict left as unverifiable. ")
                 + f"{len(items)} ranked evidence items were collected"
                 + (f", including {len(fc)} fact-check/debunk item(s): " + "; ".join(x['title'][:90] for x in fc[:3])
                    if fc else "") + ".")
    return {"subclaims": subs, "verdict": verdict, "rationale": rationale, "missing_context": [],
            "origin_statement": "", "critique": {"verdict_holds": False, "attacks": [{
                "target": "verdict", "argument": "No LLM analysis was performed.", "severity": "high"}],
                "alternative_explanations": [], "missing_evidence": []},
            "debunk_ids": [], "irrelevant_ids": [], "origin_rejected_ids": [], "caveats": ["LLM analysis unavailable: " + str(err)[:200]],
            "primary_hint": ""}


def _orient_evidence(st, subclaims):
    """Models often list the evidence that contradicts the CLAIM as "dissenting" even when their verdict is
    'refuted' (so it actually backs the verdict), which made confident debunks score 0 confidence. Use the item
    stances to put each id on the side it really argues: for a refuted / misleading sub-claim, debunk-stance
    items back the verdict and plain reports of the claim argue against it; for a supported one, the reverse.

    Only items that closely match THIS claim are moved: a fact-check debunking a different rumour about the same
    event ("fake video of the Paris Olympics") also has debunk stance, but says nothing against "the 2024
    Olympics were held in Paris"; for those the model's own placement is kept."""
    def side(i):
        a = st.assign.get(i) or {}
        sim = a.get("similarity")
        close = a.get("relevance", 0) >= 0.75 and (sim is None or sim >= 0.7)
        return a.get("stance") if close else None
    for s in subclaims:
        v = s["verdict"]
        if v == "unverifiable":
            continue
        backs_claim = v == "supported"
        sup, dis = [], []
        for i in s["supporting"] + s["dissenting"]:
            stc = side(i)
            if stc in ("debunk", "joke"):
                (dis if backs_claim else sup).append(i)
            elif stc == "support":
                (sup if backs_claim else dis).append(i)
            else:  # reports, primary documents, loosely related items: keep the model's placement
                (sup if i in s["supporting"] else dis).append(i)
        moved = (set(sup) - set(s["supporting"])) | (set(dis) - set(s["dissenting"]))
        if moved:
            st.emit("analysis", f"{s['id']}: re-oriented {sorted(moved)} by stance for the '{v}' verdict")
        s["supporting"], s["dissenting"] = list(dict.fromkeys(sup)), [i for i in dict.fromkeys(dis) if i not in sup]


async def consolidated_analysis(c, st, max_items=30, max_chars=28000):
    items = rank_evidence(st, max_items=max_items, max_chars=max_chars)
    by_kind = {}
    for x in items:
        by_kind[x["kind"]] = by_kind.get(x["kind"], 0) + 1
    st.emit("analysis", f"single LLM call over {len(items)} ranked evidence items {by_kind}")
    valid = {x["id"] for x in items}
    try:
        if os.getenv("TRACER_NO_LLM") == "1":  # evaluation of retrieval + fact-check consensus alone
            raise NoJsonReturned("LLM disabled (TRACER_NO_LLM=1)")
        raw = await llm.ask_json(ANALYSIS_SYS, _pack(st, items), "smart", 2000)
        res = _clean(raw, valid)
        if not res["subclaims"] and res["verdict"] == "unverifiable" and not res["rationale"]:
            raise NoJsonReturned("model returned JSON without an analysis")
    except NoJsonReturned as e:
        st.emit("analysis", f"! LLM analysis failed: {str(e)[:200]}")
        res = _local_fallback(st, items, e)

    # Apply the model's stance review to the matcher output (no extra call).
    for i in res["irrelevant_ids"]:
        if i in st.assign and i not in {s for x in res["subclaims"] for s in x["supporting"] + x["dissenting"]}:
            st.assign.pop(i, None)
    shown = {c["id"] for c in (st.origin or {}).get("candidates", [])[:5]}
    st.origin_rejected = [i for i in res["origin_rejected_ids"] if i in shown]
    if st.origin_rejected:
        st.emit("analysis", f"origin review rejected {st.origin_rejected} as not asserting the claim")
    for i in res["debunk_ids"]:
        if i in st.assign:
            st.assign[i]["stance"] = "debunk"
    st.evidence_pack = items
    _orient_evidence(st, res["subclaims"])
    # Deterministic cross-checks of the model's verdict (sub-claim consistency, fact-check consensus).
    consensus = fact_check_consensus(st)
    res["verdict"], checks = reconcile_verdict(res["verdict"], res["subclaims"], consensus)
    res["caveats"] = res["caveats"] + checks
    for n in checks:
        st.emit("analysis", "check: " + n)
    st.fact = {"subclaims": res["subclaims"], "missing_context": res["missing_context"],
               "overall": {"verdict": res["verdict"], "summary": res["rationale"]},
               "plan": [{"id": s["id"], "text": s["text"], "depends_on": []} for s in res["subclaims"]],
               "primary_hint": res["primary_hint"]}
    st.fact["scores"] = compute_confidence(st)
    st.fact["supporting_evidence"] = evidence_cards(st, [i for s in res["subclaims"] for i in s["supporting"]])
    st.fact["contradicting_evidence"] = evidence_cards(st, [i for s in res["subclaims"] for i in s["dissenting"]])
    st.critiques = [res["critique"]]
    atk = res["critique"]["attacks"]
    hi = sum(a.get("severity") == "high" for a in atk)
    med = sum(a.get("severity") == "medium" for a in atk)
    sc = st.fact["scores"]
    conflict = {verdict_side(res["verdict"]), consensus.get("majority")} == {"true", "false"}
    sc["fact_check_conflict"] = bool(conflict)
    sc["after_critique"] = round(max(0.0, sc["confidence"] - 0.10 * hi - 0.04 * med - (0.15 if conflict else 0)), 2)
    st.judge = {"verdict": res["verdict"], "rationale": res["rationale"], "origin_statement": res["origin_statement"],
                "changed_from_factchecker": False, "caveats": res["caveats"]}
    st.emit("analysis", f"verdict {res['verdict']} | {len(res['subclaims'])} sub-claims | "
                        f"confidence {sc['confidence']} -> {sc['after_critique']} after self-critique "
                        f"({hi} high / {med} medium attacks)")
    return res
