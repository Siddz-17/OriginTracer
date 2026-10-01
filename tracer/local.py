"""Zero-LLM retrieval and ranking: query expansion, claim matching, variants, evidence ranking.

Everything here is deterministic so the whole evidence-gathering phase spends no LLM quota; the single
consolidated analysis call (tracer.analysis) then sees only the best-ranked, de-duplicated evidence.
"""
import re
from collections import Counter
from urllib.parse import urlparse

from .classify import SOURCE_WEIGHTS
from .mutation import entities, refine_mutations
from .outlets import TIER_SCORE, outlet_tier

STOP = set("""a about above after again against all also am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further had has have having
he her here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not
now of off on once only or other our ours out over own same she should so some such than that the their theirs
them themselves then there these they this those through to too under until up very was we were what when where
which while who whom why will with would you your yours yourself says said claim claims reportedly report reports
viral post posts shared sharing new news latest breaking really actually just per via amid""".split())

DEBUNK_RE = re.compile(
    r"\b(debunk\w*|fact[- ]?check\w*|false(ly)?|hoax|fake|fabricated|doctored|misleading|misinformation|"
    r"disinformation|no evidence|not true|untrue|baseless|unfounded|retract\w*|correction|denie[sd]|deny|"
    r"satire|satirical|rumou?r|myth|scam|altered|out of context|did not|didn't|never said|isn't|is not|"
    r"won't|will not|nope|not predicting|no proof|conspiracy theor\w*|viral claim|claims? (?:is|are) wrong)\b"
    r"|^\W*(?:no|nope|false|fact check)\b[,.:!]", re.I)
GENERIC = set("""article articles story stories video photos update updates watch latest today week weeks
month months years world times daily press report reports total about could would should their there these
those which while again still since first after other people""".split())
MONTHS_DAYS = set("""january february march april may june july august september october november december
monday tuesday wednesday thursday friday saturday sunday""".split())
# Reporting / modal verbs that say little about *what* is claimed: low weight, last in search queries.
WEAK = set("""confirm confirmed confirms announce announced announces say tell told reveal revealed reveals
experience experiences will going plan plans planning make makes made get gets got set start started
starts people officially official according""".split())
SATIRE_RE = re.compile(r"\b(satire|satirical|parody|the onion|babylon bee)\b", re.I)

# Common verb/noun swaps so paraphrased headlines still match ("shutting down" vs "discontinue").
SYNONYMS = [("shut down", "shutting down", "discontinue", "discontinued", "end", "ending", "close", "closing", "kill"),
            ("ban", "banned", "bans", "prohibit", "outlaw"),
            ("die", "died", "dead", "death", "passed away", "killed"),
            ("arrest", "arrested", "detained", "custody"),
            ("resign", "resigned", "resignation", "quit", "steps down"),
            ("announce", "announced", "announces", "confirms", "confirmed"),
            ("increase", "raise", "rises", "hike", "surge"),
            ("decrease", "cut", "cuts", "drop", "falls", "slash")]


def tokens(text):
    return [t for t in re.findall(r"[a-z0-9][a-z0-9'.-]*[a-z0-9]|[a-z0-9]", (text or "").lower())
            if t not in STOP and len(t) > 1]


def _stem(t):
    t = t.removesuffix("'s")
    for suf in ("ing", "ed", "es", "s"):
        if len(t) > 4 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def claim_profile(claim):
    """Weighted key terms of the claim: names / numbers count more than common words."""
    ents = entities(claim)
    names = {w.lower() for n in ents["names"] for w in n.split()}
    # single proper nouns / brands too ("Apple", "iCloud", "NASA"), which the multi-word NAME_RE skips
    names |= {w.lower() for w in re.findall(r"\b[\w'-]+\b", claim)
              if re.search(r"[A-Z]", w) and w.lower() not in STOP and len(w) > 1}
    names -= MONTHS_DAYS
    toks = tokens(claim)
    weights = {}
    for t in toks:
        if t in names:
            w = 2.0
        elif re.fullmatch(r"\d[\d,.]*", t):
            w = 1.2
        elif t in WEAK:
            w = 0.5
        else:
            w = 1.0 if len(t) < 6 else 1.3 if len(t) < 9 else 1.5  # longer words tend to be more specific
        weights[_stem(t)] = max(weights.get(_stem(t), 0), w)
    syn = {}
    low = claim.lower()
    for group in SYNONYMS:
        if any(re.search(rf"\b{re.escape(g)}\b", low) for g in group):
            for g in group:
                syn[_stem(g.split()[0])] = group
    return {"weights": weights, "names": names, "entities": ents, "syn": syn, "tokens": toks}


def relevance(profile, text):
    """0..1 weighted coverage of the claim's key terms in `text` (synonyms count as hits)."""
    w = profile["weights"]
    if not w or not text:
        return 0.0
    have = {_stem(t) for t in tokens(text)}
    low = text.lower()
    hit, content_hit = 0.0, False
    for t, wt in w.items():
        got = 0.0
        if t in have:
            got = wt
        # whole words only: "end" must not match "end-to-end", nor "close" match "closest"
        elif t in profile["syn"] and any(re.search(rf"(?<![\w-]){re.escape(g)}(?![\w-])", low)
                                         for g in profile["syn"][t]):
            got = wt * 0.8
        hit += got
        if got and t not in profile["names"] and t not in WEAK and t not in MONTHS_DAYS and not t[0].isdigit():
            content_hit = True
    score = hit / sum(w.values())
    # Missing every name is a strong negative; names alone ("NASA ... Earth") without any of the claim's
    # content words is usually a different story that shares keywords.
    if profile["names"] and not any(n in low for n in profile["names"]):
        score *= 0.5
    if not content_hit and any(t not in profile["names"] and t not in WEAK for t in w):
        score *= 0.55
    return round(min(1.0, score), 3)


def stance(text):
    if SATIRE_RE.search(text or ""):
        return "joke"
    return "debunk" if DEBUNK_RE.search(text or "") else "report"


# ----------------------------------------------------------------- query expansion
def expand_queries(claim):
    """Keyword queries for each source (no LLM). Returns (plan, paraphrases, orgs, base_query)."""
    prof = claim_profile(claim)
    ents = prof["entities"]
    toks = list(dict.fromkeys(prof["tokens"]))
    # search order: names, then specific words, then numbers; reporting verbs last
    ranked = sorted(toks, key=lambda t: (-(prof["weights"].get(_stem(t), 1) + (0.5 if t in prof["names"] else 0)
                                          - (0.4 if re.fullmatch(r"\d[\d,.]*", t) else 0)), toks.index(t)))
    core = " ".join(ranked[:6]) or claim
    short = " ".join(ranked[:4]) or claim
    names = ents["names"][:3]
    queries = [core]
    if len(claim) <= 90:
        queries.insert(0, claim.strip().rstrip("?.!"))
    if names and short != " ".join(names):
        queries.append(short)
    paraphrases = [claim]
    multi = [n for n in names if " " in n]  # GDELT rejects quoted single words ("phrase is too short")
    gd = [short] + ([f'"{multi[0]}" {" ".join(t for t in ranked[:4] if t not in multi[0].lower())}'.strip()]
                    if multi else [])
    plan = {
        "outlets": [queries[0]] + ([core] if core != queries[0] else []),
        "factcheck": [short],
        "news": list(dict.fromkeys(queries))[:3] + [f"{short} fact check", f"{short} false"],
        "gdelt": [q for q in gd if len(q.split()) >= 2][:2] or [short],
        "duckduckgo": [queries[0], f"{short} fact check"],
        "bluesky": [short],
    }
    return plan, paraphrases[:6], names, short


def followup_queries(st, k=2):
    """Pseudo-relevance feedback: distinctive terms that recur in the best-matching reputable headlines."""
    prof = claim_profile(st.claim)
    top = sorted((e for e in st.pool.values() if e.id in st.assign),
                 key=lambda e: -(e.meta.get("relevance", 0) + 0.3 * (outlet_score(e) > 0)))[:15]
    cnt = Counter()
    for e in top:
        for t in set(tokens(e.text[:160])):
            if (_stem(t) not in prof["weights"] and len(t) > 4 and t.isalpha() and t not in WEAK
                    and t not in GENERIC and outlet_tier(t + ".com") is None):
                cnt[t] += 1
    extra = [t for t, n in cnt.most_common(6) if n >= 3]
    base = " ".join(sorted(prof["tokens"], key=lambda t: -prof["weights"].get(_stem(t), 1))[:3])
    out = []
    for i in range(0, min(len(extra), 2 * k), 2):
        out.append(f"{base} {' '.join(extra[i:i + 2])}")
    return out[:k]


# ----------------------------------------------------------------- matching
def outlet_score(e):
    tier = outlet_tier(e.meta.get("outlet_domain") or e.url) or outlet_tier(e.meta.get("resolved_url") or "")
    return TIER_SCORE.get(tier, 0.0)


def match_items(st, threshold=0.5):
    """Assign relevant pool items to the claim with a local stance; returns number of new matches."""
    prof = claim_profile(st.claim)
    new = 0
    for e in st.pool.values():
        if e.id in st.assign:
            continue
        r = relevance(prof, e.text)
        e.meta["relevance"] = r
        tier = outlet_tier(e.meta.get("outlet_domain") or e.url)
        # fact-check pages often phrase the claim differently ("No, X did not..."): lower bar for them
        bar = threshold - (0.15 if tier == "fact_check" else 0.0)
        if e.meta.get("engine") == "ddgs_text" and not tier:
            bar += 0.1  # generic web pages share keywords far more often than news headlines do
        if r >= bar:
            st.assign[e.id] = {"variant": None, "stance": stance(e.text), "relevance": r,
                               "outlet_tier": tier, "by": "local"}
            new += 1
    return new


def _signature(text):
    en = entities(text)
    return (tuple(en["numbers"][:2]), tuple(en["years"][:2]))


def build_variants(st, max_variants=6):
    """V0 = the claim as given. Items stating other numbers / years become child variants (mutation labels
    come from mutation.refine_mutations, which diffs each child against V0); everything else stays in V0."""
    ev = st.by_id()
    matched = sorted((ev[i] for i in st.assign if i in ev), key=lambda e: e.eff_ts)
    st.variants = [{"id": "V0", "text": st.claim, "mutation": "original", "parent": None,
                    "entities": entities(st.claim), "notes": "claim as submitted"}]
    if not matched:
        return
    root = _signature(st.claim)
    groups = {}
    for e in matched:
        sig = _signature(e.text[:200])  # title / lead only: descriptions add unrelated numbers
        if sig != root and sig != ((), ()) and st.assign[e.id]["stance"] != "debunk":
            groups.setdefault(sig, []).append(e)
    # a lone item from an unknown site is more often noise than a real mutation
    groups = {k: v for k, v in groups.items() if len(v) >= 2 or any(st.assign[x.id].get("outlet_tier") for x in v)
              or any(x.source in ("reddit", "bluesky") for x in v)}
    keep = sorted(groups, key=lambda k: (-len(groups[k]), groups[k][0].eff_ts))[: max_variants - 1]
    keep.sort(key=lambda k: groups[k][0].eff_ts)  # ids in order of first appearance
    vid = {}
    for i, k in enumerate(keep, 1):
        first = groups[k][0]
        vid[k] = f"V{i}"
        # Claim states no numbers/dates: the earliest numeric variant is the parent of later ones, so
        # "1 million" -> "5 million" is diffed (exaggeration) instead of each child being compared to V0.
        parent = "V1" if (i > 1 and root == ((), ())) else "V0"
        st.variants.append({"id": f"V{i}", "text": first.text[:240], "mutation": "other", "parent": parent,
                            "entities": entities(first.text[:240]),
                            "notes": f"{len(groups[k])} item(s); first {first.eff_ts:%Y-%m-%d} on {first.source} "
                                     f"({first.author})"})
    for e in matched:
        st.assign[e.id]["variant"] = vid.get(_signature(e.text[:200]), "V0")
    refine_mutations(st.variants)


# ----------------------------------------------------------------- evidence ranking
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“])")


def best_passage(profile, text, limit=700):
    """Highest-relevance window of consecutive sentences (keeps quotes, numbers and denials intact)."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    sents = _SENT.split(text)[:400]
    scores = [relevance(profile, s) + (0.25 if DEBUNK_RE.search(s) else 0) for s in sents]
    best, best_i = -1.0, 0
    for i in range(len(sents)):
        s = sum(scores[i:i + 3])
        if s > best:
            best, best_i = s, i
    out, j = "", max(0, best_i - 1)
    while j < len(sents) and len(out) + len(sents[j]) < limit:
        out += sents[j] + " "
        j += 1
    return (out.strip() or text[:limit])


def _domain(u):
    return urlparse(u or "").netloc.lower().removeprefix("www.")


def rank_evidence(st, max_items=30, max_chars=28000):
    """Score every candidate document locally and return the de-duplicated top set for the analysis prompt.

    score = 0.50*relevance + 0.25*source weight + 0.15*reputable-outlet tier + 0.10*(debunk / primary bonus)
    At most 2 items per domain; fact-check items and primary-type documents are always kept if relevant."""
    prof = claim_profile(st.claim)
    cands = []
    for r in st.refs:  # scraped primary pages (P*), Wikipedia (W*)
        rel = relevance(prof, r.get("title", "") + " " + r.get("text", "")[:4000])
        if rel < 0.25 and r["weight"] < 0.75:
            continue
        tier = outlet_tier(r["url"])
        bonus = 1.0 if r["weight"] >= 0.75 else 0.0
        cands.append({"id": r["label"], "kind": r["type"], "weight": r["weight"], "url": r["url"],
                      "outlet": _domain(r["url"]), "tier": tier, "date": r.get("first_capture"),
                      "title": (r.get("title") or "")[:160], "rel": rel,
                      "score": 0.5 * rel + 0.25 * r["weight"] + 0.15 * TIER_SCORE.get(tier, 0) + 0.1 * bonus,
                      "text": best_passage(prof, r.get("text", ""))})
    ev = st.by_id()
    for i, a in st.assign.items():
        e = ev.get(i)
        if not e:
            continue
        dom = e.meta.get("outlet_domain") or _domain(e.meta.get("resolved_url") or e.url)
        tier = a.get("outlet_tier") or outlet_tier(dom) or outlet_tier(e.meta.get("resolved_url") or "")
        kind = "fact_check" if tier == "fact_check" else ("social" if e.source in ("bluesky", "reddit") else "news")
        w = SOURCE_WEIGHTS.get(kind, 0.5)
        full = e.meta.get("fulltext") or ""
        rel = a.get("relevance", e.meta.get("relevance", 0.0))
        bonus = 1.0 if a.get("stance") == "debunk" else 0.0
        text = e.text if not full else (e.text[:200] + " … " + best_passage(prof, full, 900))
        cands.append({"id": i, "kind": kind, "weight": w, "url": e.meta.get("resolved_url") or e.url,
                      "outlet": dom or e.author, "tier": tier,
                      "date": None if e.meta.get("undated") else e.eff_ts.strftime("%Y-%m-%d"),
                      "title": e.text[:160], "rel": rel, "stance": a.get("stance"),
                      "score": 0.5 * rel + 0.25 * w + 0.15 * TIER_SCORE.get(tier, 0) + 0.1 * bonus
                               + (0.05 if full else 0.0),
                      "text": text})
    cands.sort(key=lambda x: -x["score"])
    must = [x for x in cands if (x["tier"] == "fact_check" or x["weight"] >= 0.75) and x["rel"] >= 0.3][:8]
    out, per_dom, seen_titles, used = [], Counter(), set(), 0
    for x in must + cands:
        key = re.sub(r"\W+", " ", x["title"].lower())[:70]
        if x["id"] in {o["id"] for o in out} or key in seen_titles or per_dom[x["outlet"]] >= 2:
            continue
        size = len(x["text"]) + 200
        if len(out) >= max_items or used + size > max_chars:
            break
        out.append(x)
        seen_titles.add(key)
        per_dom[x["outlet"]] += 1
        used += size
    return out


def earliest_reputable(st):
    """Earliest matched item from a wire/major outlet or fact-checker (a sanity anchor for the origin)."""
    ev = st.by_id()
    rep = [ev[i] for i, a in st.assign.items() if i in ev and a.get("outlet_tier")]
    return min(rep, key=lambda e: e.eff_ts) if rep else None

