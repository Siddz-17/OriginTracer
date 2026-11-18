"""Zero-LLM retrieval and ranking: query expansion, claim matching, variants, evidence ranking.

Everything here is deterministic so the whole evidence-gathering phase spends no LLM quota; the single
consolidated analysis call (tracer.analysis) then sees only the best-ranked, de-duplicated evidence.
"""
import re
from collections import Counter
from urllib.parse import urlparse

from . import nlp
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

# Broad cue list, kept for passage selection (which sentence of an article talks about the claim's truth).
DEBUNK_RE = re.compile(
    r"\b(debunk\w*|fact[- ]?check\w*|false(ly)?|hoax|fake|fabricated|doctored|misleading|misinformation|"
    r"disinformation|no evidence|not true|untrue|baseless|unfounded|retract\w*|correction|denie[sd]|deny|"
    r"satire|satirical|rumou?r|myth|scam|altered|out of context|did not|didn't|never said|isn't|is not|"
    r"won't|will not|nope|not predicting|no proof|conspiracy theor\w*|viral claim|claims? (?:is|are) wrong)\b"
    r"|^\W*(?:no|nope|false|fact check)\b[,.:!]", re.I)

# Stance cues, applied to the headline / lead only. STRONG cues say the claim is false on their own; NEGATION
# cues ("did not", "denies") only count when they sit next to one of the claim's own key terms, so "lander did
# not wake up" is not read as a debunk of "Chandrayaan-3 landed near the south pole".
STRONG_DEBUNK_RE = re.compile(
    r"\b(debunk\w*|hoax(es)?|fake news|is fake|are fake|fabricated|doctored|misinformation|disinformation|"
    r"no evidence|not true|untrue|baseless|unfounded|false(ly)? (?:claim|claims|claimed|report|reports|"
    r"rumou?r|post|posts|story|information)|(?:is|are|was|were) false|myth(s)?\b(?! and facts)|"
    r"conspiracy theor\w*|no proof|scam|digitally altered|out of context|misattributed|"
    r"claims? (?:is|are) wrong|pants on fire)\b"
    r"|^\W*(?:no|nope|false|fake|wrong|misleading)\b[,.:!—–-]"
    r"|\?\s*(?:no|nope|false|not quite|not really)\b", re.I)
NEGATION_RE = re.compile(r"\b(not|n't|never|no longer|denie[sd]|deny|denying|dismiss(es|ed)?|refute[sd]?|"
                         r"rejects?|rejected|reject(ing)?|rubbish(es|ed)?|slams?|clarif(y|ies|ied))\b", re.I)
RATING_RE = re.compile(
    r"(?:\brating|\bverdict|\bclaim review|\brated|\bwe rate[sd]? (?:this|it|the claim)(?: as)?)\s*[:\-]?\s*"
    r"([a-z][a-z /-]{1,30})|[-–—|(]\s*(true|false|mostly true|mostly false|half true|misleading|"
    r"pants on fire|mixture|unproven|fake|satire)\s*\)?\s*$", re.I)
_RATING_MAP = [  # first match wins: "mostly false" / "half true" must be tested before "false" / "true"
    (r"mostly false|half[- ]true|partly false|partially false|partly true|mixture|mixed|misleading|"
     r"missing context|needs context|lacks context|out of context|exaggerat\w*|distorted|outdated|cherry",
     "misleading"),
    (r"satire|satirical|parody|joke", "satire"),
    (r"unproven|unverified|unsubstantiated|research in progress|no conclusion|insufficient", "unproven"),
    (r"pants on fire|pinocchio|fake|fabricated|hoax|scam|false|incorrect|wrong|untrue|baseless|"
     r"no evidence|unsupported|debunked|not true|bogus|misattributed|altered|\bno\b", "false"),
    (r"mostly true|mostly accurate|\btrue\b|correct|accurate|confirmed|legit|\breal\b", "true"),
]
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

# Common verb/noun swaps so paraphrased headlines still match ("shutting down" vs "discontinues").
# Single words are matched by stem, so listing one form is enough ("discontinue" covers "discontinues").
SYNONYMS = [("shut down", "discontinue", "end", "close", "kill", "scrap", "axe", "retire", "sunset", "pull the plug"),
            ("ban", "prohibit", "outlaw", "block", "forbid"),
            ("die", "dead", "death", "passed away", "killed", "obituary"),
            ("arrest", "detain", "custody", "jail", "charged"),
            ("resign", "resignation", "quit", "step down", "steps down", "stepping down"),
            ("announce", "confirm", "unveil", "reveal", "declare"),
            ("increase", "raise", "rise", "hike", "surge", "jump", "soar"),
            ("decrease", "cut", "drop", "fall", "slash", "plunge", "reduce"),
            ("cure", "treat", "heal", "remedy"),
            ("land", "touchdown", "touch down", "soft-land"),
            ("own", "owner", "ownership", "possess"),
            ("buy", "bought", "purchase", "acquire", "acquisition"),
            ("largest", "biggest"),
            ("visible", "seen", "see", "spot", "observe"),
            ("darkness", "dark", "blackout", "black out"),
            ("vaccine", "vaccination", "jab")]
NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}
NUMBER_WORDS.update({"thirteen": "13", "fifteen": "15", "twenty": "20", "thirty": "30", "fifty": "50",
                     "hundred": "100", "dozen": "12"})


def tokens(text):
    out = []
    for t in re.findall(r"[a-z0-9][a-z0-9'.-]*[a-z0-9]|[a-z0-9]", (text or "").lower()):
        t = NUMBER_WORDS.get(t, t)  # "six days" and "6 days" are the same claim
        if t not in STOP and (len(t) > 1 or t.isdigit()):
            out.append(t)
    return out


def _stem(t):
    """Light suffix stripper: shutting/shuts/shut -> shut, cures/cured/cure -> cur, landed/landing -> land."""
    t = t.removesuffix("'s").removesuffix("’s")
    if len(t) > 4 and not t[0].isdigit():
        if t.endswith("ies"):
            t = t[:-3] + "y"
        elif t.endswith(("sses", "shes", "ches", "xes", "zes")):
            t = t[:-2]
        elif t.endswith("ing") and len(t) > 5:
            t = t[:-3]
        elif t.endswith("ed") and len(t) > 4:
            t = t[:-2]
        elif t.endswith("s") and not t.endswith(("ss", "us", "is")):
            t = t[:-1]
    elif len(t) == 4 and t.endswith("s") and not t.endswith(("ss", "us", "is")) and not t[0].isdigit():
        t = t[:-1]  # days -> day, jabs -> jab
    if len(t) > 4 and not t[0].isdigit():
        if len(t) > 3 and t[-1] == t[-2] and t[-1] not in "lsz":
            t = t[:-1]  # shutt -> shut, stopp -> stop
    if len(t) > 3 and t.endswith("e") and not t[0].isdigit():
        t = t[:-1]
    return t


def _syn_matchers(group):
    singles = {_stem(g) for g in group if " " not in g}
    phrases = [re.compile(r"(?<![\w-])" + r"\w*\s+".join(map(re.escape, g.split())) + r"\w*(?![\w-])")
               for g in group if " " in g]
    return singles, phrases


def _strip_poss(w):
    return w.removesuffix("'s").removesuffix("’s")


def claim_profile(claim):
    """Weighted key terms of the claim: names / numbers count more than common words."""
    ents = entities(claim)
    names = {w.lower() for n in ents["names"] for w in n.split()}
    # single proper nouns / brands too ("Apple", "iCloud", "NASA"), which the multi-word NAME_RE skips
    names |= {w.lower() for w in re.findall(r"\b[\w'-]+\b", claim)
              if re.search(r"[A-Z]", w) and w.lower() not in STOP and len(w) > 1}
    names = {_strip_poss(n) for n in names} - MONTHS_DAYS - WEAK
    toks = tokens(claim)
    weights = {}
    for t in toks:
        if _strip_poss(t) in names:
            w = 2.0
        elif re.fullmatch(r"\d[\d,.]*", t):
            w = 1.2
        elif t in WEAK:
            w = 0.5
        else:
            w = 1.0 if len(t) < 6 else 1.3 if len(t) < 9 else 1.5  # longer words tend to be more specific
        weights[_stem(t)] = max(weights.get(_stem(t), 0), w)
    stems = {_stem(t) for t in toks}
    low = claim.lower()
    syn = {}
    for group in SYNONYMS:
        singles, phrases = _syn_matchers(group)
        for g in group:
            parts = g.split()
            found = (re.search(r"(?<![\w-])" + r"\w*\s+".join(map(re.escape, parts)) + r"\w*", low) if len(parts) > 1
                     else _stem(g) in stems)
            if not found:
                continue
            key = next((_stem(p) for p in parts if _stem(p) in weights), None)
            if key:
                syn[key] = (singles, phrases)
            break
    return {"weights": weights, "names": names, "entities": ents, "syn": syn, "tokens": toks,
            "negated": bool(NEGATION_RE.search(claim))}


def _has_word(low, w):
    return re.search(rf"(?<![\w-]){re.escape(w)}(?![\w-])", low) is not None


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
        elif t in profile["syn"]:
            singles, phrases = profile["syn"][t]
            if singles & have or any(p.search(low) for p in phrases):
                got = wt * 0.8
        hit += got
        if got and t not in profile["names"] and t not in WEAK and t not in MONTHS_DAYS and not t[0].isdigit():
            content_hit = True
    score = hit / sum(w.values())
    # Missing every name is a strong negative; names alone ("NASA ... Earth") without any of the claim's
    # content words is usually a different story that shares keywords. Names match whole words only
    # ("apple" must not match "pineapple").
    if profile["names"] and not any(_has_word(low, n) or _stem(n) in have for n in profile["names"]):
        score *= 0.5
    if not content_hit and any(t not in profile["names"] and t not in WEAK for t in w):
        score *= 0.55
    return round(min(1.0, score), 3)


PAST_RE = re.compile(r"\b(\w{3,}ed|has|have|had|was|were|did|died|won|lost|bought|sold|said|made|found|"
                     r"became|went|took|gave|left|shut|hit|set)\b", re.I)
PREVIEW_RE = re.compile(r"\b(will|would|plans?|planning|planned|set to|aims?|aiming|to attempt|attempts? to|"
                        r"preparing|prepares?|could|may|might|hopes?|scheduled|upcoming|expected to|"
                        r"ahead of|countdown|to launch|gears? up|eyes|bid to|mission to)\b", re.I)


def predicate_coverage(profile, text):
    """Share (by weight) of the claim's content words (not names, numbers or reporting verbs) found in `text`.
    An origin candidate must state what the claim says happened, not merely mention the same names."""
    w = {t: wt for t, wt in profile["weights"].items()
         if t not in profile["names"] and t not in WEAK and t not in MONTHS_DAYS and not t[0].isdigit()}
    if not w:
        return 1.0
    have = {_stem(t) for t in tokens(text)}
    low = (text or "").lower()
    got = 0.0
    for t, wt in w.items():
        if t in have:
            got += wt
        elif t in profile["syn"]:
            singles, phrases = profile["syn"][t]
            if singles & have or any(p.search(low) for p in phrases):
                got += wt
    return round(got / sum(w.values()), 3)


def is_preview(claim, text):
    """A forward-looking item ("India plans to land on the Moon") cannot be the origin of a claim that says the
    event already happened ("India landed on the Moon")."""
    lead = (text or "")[:200]
    plain = re.sub(r"\b[A-Z][\w'-]*", " ", claim)  # "United States" / "Ted" are names, not past tense
    return bool(PAST_RE.search(plain) and not PREVIEW_RE.search(claim) and PREVIEW_RE.search(lead))


def fact_check_rating(text, meta=None):
    """Normalised fact-check rating ('false' | 'misleading' | 'true' | 'satire' | 'unproven') or None."""
    raw = (meta or {}).get("rating") or ""
    if not raw:
        m = RATING_RE.search((text or "")[:400])
        raw = (m.group(1) or m.group(2)) if m else ""
    raw = raw.strip().lower()
    if not raw:
        return None
    for pat, lab in _RATING_MAP:
        if re.search(pat, raw):
            return lab
    return None


def _negated_near_claim(lead, profile, window=5):
    """True when a negation / denial word sits within `window` tokens of one of the claim's predicate terms
    (content words / their synonyms, not names: "Chandrayaan-3 lander did not wake up" is not a denial)."""
    if not profile or profile.get("negated"):
        return False  # the claim itself is negative; a negative headline then agrees with it
    words = re.findall(r"[\w'’-]+", lead.lower())
    key = {t for t in profile["weights"] if t not in WEAK and t not in profile["names"] and not t[0].isdigit()}
    syn = profile.get("syn", {})
    pos = [i for i, wd in enumerate(words)
           if (s := _stem(NUMBER_WORDS.get(wd, wd))) in key or any(s in singles for singles, _ in syn.values())]
    if not pos:
        return False
    for i, wd in enumerate(words):
        if NEGATION_RE.fullmatch(wd) or wd.endswith(("n't", "n’t")):
            if any(abs(i - p) <= window for p in pos):
                return True
    return False


def stance(text, profile=None, meta=None, tier=None):
    """Local stance of one item toward the claim, from its headline / lead only:
    'joke' (satire), 'debunk' (says the claim is false or misleading), 'support' (a fact-check rating the
    claim true) or 'report' (repeats / reports it). With a claim profile, negations count only next to the
    claim's own key terms; without one, any negation cue in the lead counts (legacy behaviour)."""
    lead = (text or "")[:260]
    rating = fact_check_rating(text, meta)
    if rating in ("false", "misleading"):
        return "debunk"
    if rating == "satire":
        return "joke"
    if rating == "true":
        return "support"
    if SATIRE_RE.search(lead):
        return "joke"
    if STRONG_DEBUNK_RE.search(lead):
        return "debunk"
    if profile is None:
        return "debunk" if DEBUNK_RE.search(lead) else "report"
    if _negated_near_claim(lead, profile):
        return "debunk"
    if (tier == "fact_check" or re.search(r"\bfact[- ]?check", lead, re.I)) and "?" in lead[:160]:
        return "debunk"  # "Fact check: Is X true?" pages overwhelmingly answer no; a rating overrides this
    return "report"


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
        "claimreview": list(dict.fromkeys([claim.strip()[:200], short])),  # ClaimReview matches whole claims
        "websearch": [queries[0], f"{short} fact check"],
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
    """Assign relevant pool items to the claim with a local stance; returns number of new matches.

    Keyword relevance decides first; when the optional local models are available (tracer.nlp), embedding
    similarity vetoes keyword coincidences and rescues paraphrases, and NLI adds high-confidence debunks."""
    prof = claim_profile(st.claim)
    cands = []
    for e in st.pool.values():
        if e.id in st.assign:
            continue
        r = relevance(prof, e.text)
        e.meta["relevance"] = r
        tier = outlet_tier(e.meta.get("outlet_domain") or e.url)
        # fact-check pages often phrase the claim differently ("No, X did not..."): lower bar for them
        # ...but only when they share the claim's predicate: fact-checks of OTHER claims about the same person
        # ("Bill Gates did not say ...") match on the name alone and must not count as evidence here
        bar = threshold - (0.15 if tier == "fact_check" and predicate_coverage(prof, e.text) >= 0.34 else 0.0)
        if e.meta.get("engine") in ("ddgs_text", "brave", "tavily", "serper_search") and not tier:
            bar += 0.1  # generic web pages share keywords far more often than news headlines do
        if r >= 0.2:  # cheap keyword pre-filter: only these are worth embedding
            cands.append((e, r, bar, tier))
    sims = nlp.similarities(st.claim, [e.text for e, *_ in cands])
    keep = []
    for k, (e, r, bar, tier) in enumerate(cands):
        sim = sims[k] if sims else None
        if sim is not None:
            e.meta["similarity"] = round(sim, 3)
            ok = (r >= bar and sim >= nlp.MATCH_MIN_SIM) or (r >= 0.25 and sim >= nlp.RESCUE_SIM)
            rel = round(0.5 * r + 0.5 * max(0.0, min(1.0, (sim - 0.3) / 0.5)), 3)
        else:
            ok, rel = r >= bar, r
        if ok:
            keep.append((e, rel, tier, sim))
    # NLI only for matched items that are very close in meaning: it ADDS debunks the cue rules missed
    close = [(e, sim) for e, _, _, sim in keep if sim is not None and sim >= nlp.NLI_MIN_SIM]
    contra = dict(zip((e.id for e, _ in close), nlp.contradiction(st.claim, [e.text for e, _ in close]) or []))
    for e, rel, tier, sim in keep:
        stc = stance(e.text, prof, e.meta, tier)
        if stc == "report" and contra.get(e.id, 0) >= nlp.NLI_CONTRA:
            stc = "debunk"
            e.meta["stance_by"] = "nli"
        st.assign[e.id] = {"variant": None, "stance": stc, "relevance": rel, "keyword_relevance": e.meta["relevance"],
                           "similarity": sim, "outlet_tier": tier, "by": "local",
                           "rating": fact_check_rating(e.text, e.meta) if tier == "fact_check"
                           or e.meta.get("rating") else None}
    return len(keep)


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
        if sig != root and sig != ((), ()) and st.assign[e.id]["stance"] not in ("debunk", "support", "joke"):
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
        # primary-type pages must still be about the claim: a high source weight does not make an unrelated
        # government page evidence
        if rel < (0.35 if r["weight"] >= 0.75 else 0.25):
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
        rating = a.get("rating")
        bonus = 1.0 if (a.get("stance") in ("debunk", "support") or rating) else 0.0
        text = e.text if not full else (e.text[:200] + " … " + best_passage(prof, full, 900))
        cands.append({"id": i, "kind": kind, "weight": w, "url": e.meta.get("resolved_url") or e.url,
                      "outlet": dom or e.author, "tier": tier,
                      "date": None if e.meta.get("undated") else e.eff_ts.strftime("%Y-%m-%d"),
                      "title": e.text[:160], "rel": rel, "stance": a.get("stance"), "rating": rating,
                      "score": 0.5 * rel + 0.25 * w + 0.15 * TIER_SCORE.get(tier, 0) + 0.1 * bonus
                               + (0.05 if full else 0.0),
                      "text": text})
    cands.sort(key=lambda x: -x["score"])
    # Fact-checks about OTHER claims on the same person are common ("Bill Gates did not say ..."), so only
    # fact-checks that match this claim well are guaranteed a slot.
    must = [x for x in cands if (x["tier"] == "fact_check" and x["rel"] >= 0.5)
            or (x["weight"] >= 0.75 and x["rel"] >= 0.4)][:8]
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

