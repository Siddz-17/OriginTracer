"""Entity extraction (numbers / years / names) and deterministic mutation labelling."""
import re

MUTATION_LABELS = ["original", "exaggeration", "missing_context", "misattribution", "fabrication",
                   "quote_distortion", "date_shift", "entity_swap", "partial_truth",
                   "number_shift", "satire_literal", "other"]


NUM_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(k|m|bn|b|thousand|million|billion|%|percent)?(?![\w])", re.I)
SCALE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9}
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b|\b[A-Z]{2,}\b")


def entities(text):
    nums = []
    for m in NUM_RE.finditer(text or ""):
        raw, unit = m.group(1), (m.group(2) or "").lower()
        if not unit and YEAR_RE.fullmatch(raw):
            continue
        v = float(raw.replace(",", ""))
        nums.append(v * SCALE.get(unit, 1) if unit not in ("%", "percent") else v)
    return {"numbers": sorted(set(nums)), "years": sorted(set(YEAR_RE.findall(text or ""))),
            "names": sorted(set(NAME_RE.findall(text or "")))}


def diff_entities(parent_text, child_text):
    A, B = entities(parent_text), entities(child_text)
    d = {}
    if A["numbers"] and B["numbers"] and A["numbers"] != B["numbers"]:
        pa, pb = max(A["numbers"]), max(B["numbers"])
        if pa and pa != pb:
            d["numbers"] = {"from": pa, "to": pb, "ratio": round(pb / pa, 2)}
    if A["years"] != B["years"]:
        d["years"] = {"from": A["years"], "to": B["years"]}
    added, removed = sorted(set(B["names"]) - set(A["names"])), sorted(set(A["names"]) - set(B["names"]))
    if added or removed:
        d["names"] = {"added": added, "removed": removed}
    return d



def refine_mutations(variants):
    """Normalise labels, drop dangling parents, then entity-diff each variant against its parent.
    A pure single-entity change (numbers / years / names) overrides the LLM label."""
    ids = {v["id"] for v in variants}
    for v in variants:
        if v.get("parent") not in ids or v.get("parent") == v["id"]:
            v["parent"] = None
        if v.get("mutation") not in MUTATION_LABELS:
            v["mutation"] = "other"
    byid = {v["id"]: v for v in variants}
    for v in variants:
        par = byid.get(v.get("parent"))
        if not par:
            continue
        d = diff_entities(par.get("text", ""), v.get("text", ""))
        v["diff"] = d
        only = set(d)
        if only == {"numbers"}:
            v["mutation"] = "exaggeration" if d["numbers"]["ratio"] >= 1.5 else "number_shift"
        elif only == {"years"}:
            v["mutation"] = "date_shift"
        elif only == {"names"}:
            v["mutation"] = "entity_swap"
