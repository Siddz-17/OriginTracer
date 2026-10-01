"""Evidence-weighted confidence. The LLM lists supporting/dissenting sources; code does the maths."""
from .classify import PRIMARY_MIN_W, SOURCE_WEIGHTS
from .outlets import outlet_tier


def source_weight(st, sid):
    for r in st.refs:
        if r.get("label") == sid:
            return r["weight"]
    e = next((x for x in st.pool.values() if x.id == sid), None)
    if e:
        if e.source in ("bluesky", "reddit"):
            return SOURCE_WEIGHTS["social"]
        tier = outlet_tier(e.meta.get("outlet_domain") or e.meta.get("resolved_url") or e.url)
        return SOURCE_WEIGHTS["fact_check"] if tier == "fact_check" else SOURCE_WEIGHTS["news"]
    return 0.3


def subclaim_confidence(st, s):
    sup = [source_weight(st, i) for i in s.get("supporting", [])]
    dis = [source_weight(st, i) for i in s.get("dissenting", [])]
    tot = sum(sup) + sum(dis)
    quality = max(sup) if sup else 0.0
    agree = sum(sup) / tot if tot else 0.5
    primary = 1.0 if any(w >= 0.9 for w in sup) else 0.5 if any(w >= PRIMARY_MIN_W for w in sup) else 0.0
    contra = (max(dis) * sum(dis) / tot) if dis else 0.0
    missing = 1.0 if s.get("verdict") == "unverifiable" else 0.0
    conf = 0.15 + 0.30 * quality + 0.20 * agree + 0.25 * primary - 0.30 * contra - 0.25 * missing
    return max(0.0, min(1.0, conf)), dict(quality=quality, agree=round(agree, 2), primary=primary,
                                           contradiction=round(contra, 2), missing=missing)


def compute_confidence(st):
    """evidence confidence = 0.15 + .30*quality + .20*agreement + .25*primary - .30*contradiction - .25*missing.
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
    if not prim:
        overall = min(overall, 0.6)  # no primary-type source cited -> capped
    return {"confidence": round(overall, 2), "per_subclaim": per, "primary_sources": len(prim),
            "secondary_sources": len(cited - prim), "contradictions": len(dissent)}
