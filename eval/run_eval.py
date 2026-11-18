"""Accuracy evaluation on labelled claims.

    python eval/run_eval.py --n 20                     # 20 AVeriTeC dev claims, stratified by label, + curated set
    python eval/run_eval.py --n 40 --no-llm            # retrieval / fact-check-consensus only (no LLM quota)
    python eval/run_eval.py --curated-only
    python eval/run_eval.py --report eval/results/run.jsonl   # re-score an earlier run

Data: AVeriTeC dev split (CC BY-NC 4.0, https://fever.ai/dataset/averitec.html); fetch it with
`python eval/download_data.py`. Results are appended per claim to a JSONL file, so an interrupted run (or an
exhausted free LLM quota) resumes where it stopped.

Metrics
  side accuracy      verdict mapped to true / false / mixed / unverifiable vs the gold label
  decisive accuracy  the same, only over gold true / false claims (what most users care about)
  calibration        mean confidence of correct vs wrong verdicts (correct should be clearly higher)
  origin date        |likely-origin date - date the claim was made|; within 14 days counts as a hit
Caveat: AVeriTeC claims were fact-checked, so retrieval can find the very article that produced the label.
That is how the tool is meant to work, but it makes these numbers optimistic for brand-new claims.
"""
import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

GOLD_SIDE = {"Supported": "true", "Refuted": "false", "Conflicting Evidence/Cherrypicking": "mixed",
             "Not Enough Evidence": "unverifiable", "true": "true", "false": "false", "mixed": "mixed",
             "unverifiable": "unverifiable"}
VERDICT_SIDE = {"true": "true", "mostly_true": "true", "false": "false", "mostly_false": "false",
                "misleading": "mixed", "unverifiable": "unverifiable"}


def load_cases(n, seed, curated_only):
    cases = []
    cur = os.path.join(ROOT, "eval", "claims_curated.jsonl")
    for line in open(cur, encoding="utf-8"):
        if line.strip():
            c = json.loads(line)
            cases.append({"id": "cur-" + c["id"], "claim": c["claim"], "gold": c["label"],
                          "claim_date": c.get("claim_date"), "source": "curated"})
    if curated_only:
        return cases
    path = os.path.join(ROOT, "eval", "data", "averitec_dev.json")
    if not os.path.exists(path):
        print("AVeriTeC data missing: run `python eval/download_data.py` (using the curated set only)")
        return cases
    data = json.load(open(path, encoding="utf-8"))
    by = defaultdict(list)
    for i, x in enumerate(data):
        by[x["label"]].append((i, x))
    rng = random.Random(seed)
    # stratified: half Refuted / quarter Supported / rest split, roughly the real-world mix but not 61% Refuted
    share = {"Refuted": 0.45, "Supported": 0.3, "Conflicting Evidence/Cherrypicking": 0.125,
             "Not Enough Evidence": 0.125}
    for lab, frac in share.items():
        pick = rng.sample(by[lab], min(len(by[lab]), max(1, round(n * frac))))
        for i, x in pick:
            d = None
            try:
                d = datetime.strptime(x["claim_date"], "%d-%m-%Y").strftime("%Y-%m-%d")
            except (TypeError, ValueError):
                pass
            cases.append({"id": f"avt-{i}", "claim": x["claim"], "gold": lab, "claim_date": d,
                          "source": "averitec"})
    return cases


async def run_case(case, no_llm, rounds, depth):
    from tracer.pipeline import run_pipeline
    if no_llm:
        os.environ["TRACER_NO_LLM"] = "1"
    t = time.time()
    st = await run_pipeline(case["claim"], rounds=rounds, depth=depth, archive=True)
    r = st.result
    o = (r.get("origin") or {}).get("likely") or {}
    return {**case, "verdict": r.get("verdict"), "confidence": r.get("confidence"),
            "origin_ts": str(o.get("ts"))[:10] if o.get("ts") else None, "origin_source": o.get("source"),
            "origin_author": o.get("author"), "p_unobserved": (r.get("origin") or {}).get("p_unobserved"),
            "matched": len(st.assign), "pool": len(st.pool), "secs": round(time.time() - t),
            "fact_check_ratings": sorted({a["rating"] for a in st.assign.values() if a.get("rating")}),
            "rationale": (r.get("rationale") or "")[:600], "no_llm": no_llm,
            "llm_ok": not (r.get("rationale") or "").startswith("The LLM analysis step failed")}


def score(rows):
    rows = [r for r in rows if r.get("verdict")]
    if not rows:
        return "no results"
    out = []
    failed = [r for r in rows if not r.get("no_llm") and r.get("llm_ok") is False]
    if failed:
        out.append(f"WARNING: the LLM failed on {len(failed)} claim(s); they used the no-LLM fallback")
    ok = lambda r: VERDICT_SIDE.get(r["verdict"]) == GOLD_SIDE[r["gold"]]
    acc = sum(map(ok, rows)) / len(rows)
    dec = [r for r in rows if GOLD_SIDE[r["gold"]] in ("true", "false")]
    dacc = sum(map(ok, dec)) / len(dec) if dec else float("nan")
    out.append(f"claims scored        {len(rows)}")
    out.append(f"side accuracy        {acc:.0%}")
    out.append(f"decisive accuracy    {dacc:.0%}  (gold true/false only, n={len(dec)})")
    conf_ok = [r["confidence"] for r in rows if ok(r) and r.get("confidence") is not None]
    conf_bad = [r["confidence"] for r in rows if not ok(r) and r.get("confidence") is not None]
    avg = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    out.append(f"confidence correct   {avg(conf_ok):.2f}   wrong {avg(conf_bad):.2f}")
    cm = Counter((GOLD_SIDE[r["gold"]], VERDICT_SIDE.get(r["verdict"], "?")) for r in rows)
    sides = ["true", "false", "mixed", "unverifiable"]
    out.append("confusion (rows gold, cols predicted)")
    out.append("              " + "".join(f"{s[:6]:>8}" for s in sides))
    for g in sides:
        out.append(f"  {g:12}" + "".join(f"{cm[(g, p)]:>8}" for p in sides))
    errs = []
    for r in rows:
        if r.get("origin_ts") and r.get("claim_date"):
            a = datetime.fromisoformat(r["origin_ts"]).replace(tzinfo=timezone.utc)
            b = datetime.fromisoformat(r["claim_date"]).replace(tzinfo=timezone.utc)
            errs.append(abs((a - b).days))
    if errs:
        errs.sort()
        out.append(f"origin date error    median {errs[len(errs) // 2]} days, within 14 days "
                   f"{sum(e <= 14 for e in errs) / len(errs):.0%} (n={len(errs)})")
    out.append(f"mean runtime         {avg([r['secs'] for r in rows]):.0f}s per claim")
    by_src = Counter(r["source"] for r in rows)
    for s, n in by_src.items():
        sub = [r for r in rows if r["source"] == s]
        out.append(f"  {s:10} side accuracy {sum(map(ok, sub)) / n:.0%} (n={n})")
    return "\n".join(out)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="AVeriTeC claims to sample (plus the curated set)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--curated-only", action="store_true")
    ap.add_argument("--no-llm", action="store_true", help="skip the LLM: verdict from fact-check consensus only")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--out", default=None, help="JSONL results file (resumes if it exists)")
    ap.add_argument("--report", default=None, help="only score an existing results file")
    a = ap.parse_args()
    if a.report:
        print(score([json.loads(x) for x in open(a.report, encoding="utf-8") if x.strip()]))
        return
    logging.basicConfig(level=logging.ERROR)
    os.makedirs(os.path.join(ROOT, "eval", "results"), exist_ok=True)
    out = a.out or os.path.join(ROOT, "eval", "results",
                                f"run-{'nollm-' if a.no_llm else ''}n{a.n}-s{a.seed}.jsonl")
    done = {}
    if os.path.exists(out):
        done = {json.loads(x)["id"]: json.loads(x) for x in open(out, encoding="utf-8") if x.strip()}
        # claims where the LLM failed (quota, outage) used the fallback verdict: run them again
        done = {k: v for k, v in done.items() if v.get("no_llm") or v.get("llm_ok", True)}
    cases = load_cases(a.n, a.seed, a.curated_only)
    print(f"{len(cases)} claims ({len(done)} already done) -> {out}", flush=True)
    for k, c in enumerate(cases, 1):
        if c["id"] in done:
            continue
        try:
            row = await run_case(c, a.no_llm, a.rounds, a.depth)
        except Exception as e:  # one broken claim must not end a long evaluation
            print(f"[{k}/{len(cases)}] ERROR {type(e).__name__}: {e}", flush=True)
            continue
        done[c["id"]] = row
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        mark = "ok " if VERDICT_SIDE.get(row["verdict"]) == GOLD_SIDE[row["gold"]] else "BAD"
        print(f"[{k}/{len(cases)}] {mark} gold={row['gold'][:12]:12} got={row['verdict']:13} "
              f"conf={row['confidence']} origin={row['origin_ts']} | {c['claim'][:70]}", flush=True)
    print()
    print(score(list(done.values())))


if __name__ == "__main__":
    asyncio.run(main())
