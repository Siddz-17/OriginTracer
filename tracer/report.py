"""Markdown report renderer."""
import json
from collections import defaultdict
from datetime import datetime

from .classify import UTC, classify_url
from .provenance import ORIGIN_W, origin_analysis

# ------------------------------------------------------------------------ report
EMOJI = {"supported": "✅", "refuted": "❌", "misleading": "⚠️", "unverifiable": "❔"}


def render_report(st):
    oa = origin_analysis(st)
    sc = st.fact.get("scores", {})
    J = st.judge or {}
    ov = {"verdict": J.get("verdict") or st.fact.get("overall", {}).get("verdict", "n/a"),
          "summary": J.get("rationale") or st.fact.get("overall", {}).get("summary", "")}
    final_conf = (st.result or {}).get("confidence", sc.get("after_critique", sc.get("confidence")))
    last = st.critiques[-1] if st.critiques else {}
    ev = st.by_id()
    L = ["# Narrative Origin Report\n", f"**Claim:** {st.claim}\n",
         f"**Generated:** {datetime.now(UTC):%Y-%m-%d %H:%M UTC} | **Evidence pool:** {len(st.pool)} | "
         f"**Matched:** {len(st.assign)}\n",
         "## Verdict\n",
         f"**{ov.get('verdict', 'n/a').upper()}** — final confidence **{final_conf}** "
         f"(fact-checker {sc.get('confidence', '?')} → after devil's advocate {sc.get('after_critique', '?')}). Survived stress-test: **{last.get('verdict_holds')}**.\n",
         f"{ov.get('summary', '')}\n",
         "```json", json.dumps({"verdict": ov.get("verdict"), "confidence": final_conf,
                                "primary_sources": sc.get("primary_sources"),
                                "secondary_sources": sc.get("secondary_sources"),
                                "contradictions": sc.get("contradictions")}, indent=2), "```\n"]
    if not sc.get("primary_sources"):
        L.append(f"> ⚠ No primary-type source was cited, so confidence is capped at 0.6. What would settle it: "
                 f"{st.fact.get('primary_hint', 'n/a')}. Re-run with `--primary URL`.\n")
    L.append("### Sub-claims\n")
    plan = st.fact.get("plan", [])
    by = {s.get("id"): s for s in st.fact.get("subclaims", [])}
    L += ["```mermaid", "graph LR"]
    for s in plan:
        v = by.get(s["id"], {}).get("verdict", "?")
        L.append(f'  {s["id"]}["{s["id"]} {EMOJI.get(v, "")} {v}<br/>{s["text"][:50].replace(chr(34), chr(39))}"]')
        for d in s.get("depends_on", []) or []:
            L.append(f"  {d} --> {s['id']}")
    L += ["```\n", "| Sub-claim | Verdict | Conf. | Supporting | Dissenting | Reasoning |", "|---|---|---|---|---|---|"]
    for s in st.fact.get("subclaims", []):
        pc = sc.get("per_subclaim", {}).get(s.get("id"), {})
        L.append(f"| {s.get('id')} | {s.get('verdict')} | {pc.get('confidence')} | {', '.join(s.get('supporting', []))} | "
                 f"{', '.join(s.get('dissenting', []))} | {s.get('reasoning', '')} |")

    if st.fact.get("missing_context"):
        L += ["\n**Missing context:**"] + [f"- {m}" for m in st.fact["missing_context"]]
    if J.get("caveats"):
        L += ["\n**Judge caveats:**"] + [f"- {m}" for m in J["caveats"]]
    if J.get("origin_statement"):
        L.append(f"\n_Origin (judge):_ {J['origin_statement']}")
    L += ["\n## Likely origin\n"]
    if oa:
        c0 = oa["candidates"][0]
        L += [f"**Likely origin: {c0['prob'] * 100:.0f}% confidence** — {c0['source']} `{c0['author']}` at "
              f"{c0['ts']:%Y-%m-%d %H:%M UTC} — {c0['url']}\n", f"> {c0['text']}\n", "Evidence:"]
        L += [f"- {b}" for b in oa["bullets"]]
        L += [f"\n**Probability the true origin was never observed: {oa['p_unobserved'] * 100:.0f}%** "
              f"(heuristic: shrinks with more matched items, grows when the best candidate sits at GDELT's 90-day edge).\n",
              "| Candidate | P(origin) | Score | time | citation | repost | direct | Platform | Account |", "|---|---|---|---|---|---|---|---|---|"]
        for c in oa["candidates"]:
            p = c["parts"]
            L.append(f"| {c['id']} | {c['prob'] * 100:.0f}% | {c['score']} | {p['timestamp']} | {p['citation']} | "
                     f"{p['repost']} | {p['direct']} | {c['source']} | {c['author']} |")
        L.append(f"\nPropagation graph: {oa['n_nodes']} nodes, {oa['n_edges']} edges {oa['edge_kinds']}. "
                 f"Weights {ORIGIN_W}. Heuristic scores, not calibrated probabilities.")
        for a in oa["anomalies"]:
            L.append(f"- _anomaly:_ {a}")
    else:
        L.append("No matched items found.")

    L += ["\n## Mutation lineage\n", "```mermaid", "graph TD"]
    per_v = defaultdict(list)
    for i, a in st.assign.items():
        if i in ev:
            per_v[a.get("variant")].append(ev[i])
    for v in st.variants:
        label = (v.get("text", "")[:60] + "…").replace('"', "'")
        L.append(f'  {v["id"]}["{v["id"]} ({v.get("mutation")}, {len(per_v.get(v["id"], []))} items)<br/>{label}"]')
        if v.get("parent"):
            L.append(f'  {v["parent"]} --> {v["id"]}')
    L.append("```\n")
    for v in st.variants:
        items = sorted(per_v.get(v["id"], []), key=lambda e: e.eff_ts)
        first = f"first seen {items[0].eff_ts:%Y-%m-%d} on {items[0].source}" if items else "no items"
        d = v.get("diff") or {}
        dtxt = ""
        if "numbers" in d:
            dtxt += f" | numbers {d['numbers']['from']:g} → {d['numbers']['to']:g} (×{d['numbers']['ratio']})"
        if "years" in d:
            dtxt += f" | years {d['years']['from']} → {d['years']['to']}"
        if "names" in d:
            dtxt += f" | names +{d['names']['added']} −{d['names']['removed']}"
        L.append(f"- **{v['id']}** [{v.get('mutation')}] {v.get('text')} — _{v.get('notes', '')}_ ({first}){dtxt}")

    L += ["\n## Citation chains\n"]
    if not st.chains:
        L.append("None traced (no fetchable articles or links).")
    for ch in st.chains[:20]:
        hops = " → ".join(f"{classify_url(u)}" for u in ch["path"])
        mark = "✅ reached" if ch["resolved"] else f"✖ stopped ({ch.get('note', ch['end'])})"
        L.append(f"- {mark}: {hops}\n  - " + "\n  - ".join(ch["path"]))
    L += ["\n## Source ranking\n", "| Ref | Type | Weight | Link |", "|---|---|---|---|"]
    for r in sorted(st.refs, key=lambda r: -r["weight"]):
        L.append(f"| {r.get('label')} | {r['type']} | {r['weight']} | {r['url']} |")

    deb = [(ev[i], a) for i, a in st.assign.items() if i in ev and a.get("stance") == "debunk"]
    L += [f"\n## Contradicting / debunk items ({len(deb)})\n"]
    for e, a in sorted(deb, key=lambda x: x[0].eff_ts)[:15]:
        L.append(f"- {e.eff_ts:%Y-%m-%d} {e.source} `{e.author}`: {e.text[:140]} — {e.url}")

    L += ["\n## Who spread it\n", "| Platform | Account | Items | Engagement |", "|---|---|---|---|"]
    agg = defaultdict(lambda: [0, 0])
    for i in st.assign:
        if i in ev:
            agg[(ev[i].source, ev[i].author)][0] += 1
            agg[(ev[i].source, ev[i].author)][1] += ev[i].engagement
    for (s, a), (n, en) in sorted(agg.items(), key=lambda x: -(x[1][0] + x[1][1] / 50))[:10]:
        L.append(f"| {s} | {a} | {n} | {en} |")

    L += ["\n## Devil's advocate\n"]
    for i, cr in enumerate(st.critiques, 1):
        L.append(f"**Round {i}** — holds: {cr.get('verdict_holds')}")
        for a in cr.get("attacks", []):
            L.append(f"- [{a.get('severity')}] ({a.get('target')}) {a.get('argument')}")
        L += [f"- _alt:_ {a}" for a in cr.get("alternative_explanations", [])]
        L += [f"- _missing:_ {a}" for a in cr.get("missing_evidence", [])]
        L.append("")

    L += ["## Matched evidence (chronological)\n", "| ID | Time | Archive-adjusted | Source | Author | Variant | Stance | Link |",
          "|---|---|---|---|---|---|---|---|"]
    for e in sorted((ev[i] for i in st.assign if i in ev), key=lambda e: e.eff_ts)[:80]:
        a = st.assign[e.id]
        adj = "yes" if e.first_capture and e.first_capture < e.ts else ""
        L.append(f"| {e.id} | {e.ts:%Y-%m-%d %H:%M} | {adj} | {e.source} | {e.author} | {a.get('variant')} | {a.get('stance')} | {e.url} |")
    return "\n".join(L)
