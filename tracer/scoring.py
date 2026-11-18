"""Evidence-weighted confidence and deterministic verdict checks. The LLM lists supporting / dissenting
sources; code does the maths, and code cross-checks the model's verdict against its own sub-claims and against
the ratings published by independent fact-checkers."""
from collections import Counter
from urllib.parse import urlparse

from .classify import PRIMARY_MIN_W, SOURCE_WEIGHTS
from .outlets import outlet_tier

TRUE_SIDE = {"true", "mostly_true"}
FALSE_SIDE = {"false", "mostly_false"}


def _item(st, sid):
    for r in st.refs:
        if r.get("label") == sid:
            return r
    return next((x for x in st.pool.values() if x.id == sid), None)


def source_weight(st, sid):
    x = _item(st, sid)
    if isinstance(x, dict):
        return x["weight"]
    if x is not None:
        if x.source in ("bluesky", "reddit"):
            return SOURCE_WEIGHTS["social"]
        tier = outlet_tier(x.meta.get("outlet_domain") or x.meta.get("resolved_url") or x.url)
        return SOURCE_WEIGHTS["fact_check"] if tier == "fact_check" else SOURCE_WEIGHTS["news"]
    return 0.3


def source_domain(st, sid):
    """Publisher domain of an evidence id: independent corroboration is counted per publisher, not per item."""
    x = _item(st, sid)
    if isinstance(x, dict):
        u = x["url"]
    elif x is not None:
        u = x.meta.get("outlet_domain") or x.meta.get("resolved_url") or x.url
        if x.source in ("bluesky", "reddit"):
            return f"{x.source}:{x.author}"
    else:
        return sid
    u = u if "://" in u else "https://" + u
    host = urlparse(u).netloc.lower().removeprefix("www.")
    return ".".join(host.split(".")[-3:]) if host.endswith((".co.uk", ".com.au", ".co.in")) else \
        ".".join(host.split(".")[-2:])


def subclaim_confidence(st, s):
    sup_ids, dis_ids = list(dict.fromkeys(s.get("supporting", []))), list(dict.fromkeys(s.get("dissenting", [])))
    sup = [source_weight(st, i) for i in sup_ids]
    dis = [source_weight(st, i) for i in dis_ids]
    tot = sum(sup) + sum(dis)
    quality = max(sup) if sup else 0.0
    agree = sum(sup) / tot if tot else 0.5
    primary = 1.0 if any(w >= 0.9 for w in sup) else 0.5 if any(w >= PRIMARY_MIN_W for w in sup) else 0.0
    # Several independent credible publishers (>= news weight) saying the same thing is real corroboration;
    # many copies from one outlet or from social accounts are not.
    indep = len({source_domain(st, i) for i, w in zip(sup_ids, sup) if w >= SOURCE_WEIGHTS["news"]})
    corroboration = min(1.0, max(0, indep - 1) / 3)
    contra = (max(dis) * sum(dis) / tot) if dis else 0.0
    missing = 1.0 if s.get("verdict") == "unverifiable" else 0.0
    conf = (0.15 + 0.30 * quality + 0.20 * agree + 0.25 * primary + 0.15 * corroboration
            - 0.30 * contra - 0.25 * missing)
    return max(0.0, min(1.0, conf)), dict(quality=quality, agree=round(agree, 2), primary=primary,
                                           independent_sources=indep, corroboration=round(corroboration, 2),
                                           contradiction=round(contra, 2), missing=missing)


def confidence_cap(primary_sources, fact_checkers_agreeing):
    """Without a primary-type source a verdict rests on reporting, so it is capped; agreement among several
    independent fact-checkers (who did the primary-source work themselves) lifts that cap part of the way."""
    if primary_sources:
        return 1.0
    return 0.8 if fact_checkers_agreeing >= 2 else 0.7 if fact_checkers_agreeing == 1 else 0.6


def compute_confidence(st):
    """evidence confidence = 0.15 + .30*quality + .20*agreement + .25*primary + .15*corroboration
    - .30*contradiction - .25*missing, averaged over sub-claims and capped by confidence_cap().
    'unverifiable' sub-claims score low by design: low confidence = insufficient evidence."""
    subs = st.fact.get("subclaims", [])
    per, cited, dissent = {}, set(), set()
    for s in subs:
        c, parts = subclaim_confidence(st, s)
        per[s.get("id")] = {"confidence": round(c, 2), **parts}
        cited |= set(s.get("supporting", [])) | set(s.get("dissenting", []))
        dissent |= set(s.get("dissenting", []))
    prim = {i for i in cited if source_weight(st, i) >= PRIMARY_MIN_W}
    overall = sum(v["confidence"] for v in per.values()) / len(per) if per else 0.0
    verdict = (st.fact.get("overall") or {}).get("verdict")
    fc = fact_check_consensus(st)
    agreeing = fc["agreeing"].get(verdict_side(verdict), 0) if verdict else 0
    cap = confidence_cap(len(prim), agreeing)
    overall = min(overall, cap)
    return {"confidence": round(overall, 2), "per_subclaim": per, "primary_sources": len(prim),
            "secondary_sources": len(cited - prim), "contradictions": len(dissent), "cap": cap,
            "fact_checkers_agreeing": agreeing}


# ------------------------------------------------------------------ deterministic verdict checks
def verdict_side(v):
    return "true" if v in TRUE_SIDE else "false" if v in FALSE_SIDE else "mixed" if v == "misleading" else None


def fact_check_consensus(st):
    """Ratings published by fact-checkers that matched this claim, one vote per publisher (its most relevant
    item). Returns counts per side and the majority, which is only trusted with >= 2 independent publishers."""
    ev = st.by_id()
    best = {}
    for i, a in st.assign.items():
        rating = a.get("rating")
        if not rating or i not in ev or rating == "unproven":
            continue
        side = {"false": "false", "misleading": "mixed", "satire": "false", "true": "true"}[rating]
        dom = source_domain(st, i)
        if dom not in best or a.get("relevance", 0) > best[dom][1]:
            best[dom] = (side, a.get("relevance", 0), i, rating)
    votes = Counter(v[0] for v in best.values())
    majority, n = (votes.most_common(1)[0] if votes else (None, 0))
    trusted = n >= 2 and n > sum(votes.values()) - n
    return {"votes": dict(votes), "agreeing": dict(votes), "majority": majority if trusted else None,
            "publishers": {d: {"rating": v[3], "id": v[2]} for d, v in best.items()}}


def implied_verdict(subclaims):
    """Overall verdict implied by the sub-claim verdicts alone."""
    vs = [s.get("verdict") for s in subclaims or [] if s.get("verdict") != "unverifiable"]
    if not vs:
        return "unverifiable"
    sup, ref, mis = vs.count("supported"), vs.count("refuted"), vs.count("misleading")
    if sup == len(vs):
        return "true"
    if ref == len(vs):
        return "false"
    if ref and not sup:
        return "mostly_false"
    if sup and not ref and not mis:
        return "true"
    if sup > ref + mis:
        return "mostly_true"
    if ref > sup + mis:
        return "mostly_false"
    return "misleading"


def reconcile_verdict(verdict, subclaims, consensus):
    """Cross-check the model's verdict. Returns (verdict, notes). Only clear contradictions are changed:
    a true-side verdict over sub-claims that are all refuted (or vice versa) takes the sub-claims' verdict.
    A conflict with the fact-check consensus is reported (and costs confidence) but not overridden, because
    fact-checks can cover an older or different version of the claim."""
    notes = []
    implied = implied_verdict(subclaims)
    side, iside = verdict_side(verdict), verdict_side(implied)
    if side and iside and {side, iside} == {"true", "false"}:
        notes.append(f"Overall verdict '{verdict}' contradicted the sub-claim verdicts; using '{implied}'.")
        verdict, side = implied, iside
    elif verdict == "unverifiable" and implied in TRUE_SIDE | FALSE_SIDE and \
            all(s.get("verdict") != "unverifiable" for s in subclaims):
        notes.append(f"Every sub-claim was settled; overall verdict set to '{implied}' instead of 'unverifiable'.")
        verdict, side = implied, iside
    maj = consensus.get("majority")
    if maj and side and maj != side and "mixed" not in (maj, side):
        notes.append(f"Fact-checkers ({', '.join(consensus['publishers'])}) mostly rated this claim "
                     f"'{maj}', which disagrees with the verdict '{verdict}'. Check whether they cover the same "
                     f"version of the claim.")
    return verdict, notes


def consensus_verdict(consensus):
    """Verdict to fall back on when the LLM is unavailable: the trusted fact-check majority, else None."""
    return {"false": "false", "true": "true", "mixed": "misleading"}.get(consensus.get("majority"))
