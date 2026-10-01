# Narrative Origin Tracer

Give it a viral claim; it finds where the claim first appeared, how it mutated, who amplified it, and
fact-checks it against primary sources, then tries to knock its own verdict down before reporting.

```
claim -> 1 Query Expansion -> 2 Source Discovery -> 3 Claim Matcher -> 4 Origin Tracer
              ^                                                            |
              |                                                            v
   follow-up queries <- 7 Devil's Advocate <- 6 Fact Checker <- 5 Evidence Collection
                              | (verdict holds or rounds exhausted)
                              v
                          8 Judge -> final report (JSON + Markdown + Postgres rows)
```

| Agent | Tier | What it does |
|---|---|---|
| 1 Query Expansion | fast | paraphrases + keyword queries + contradiction queries ("fact check", "debunked"...), orgs, official domains |
| 2 Source Discovery | - | Reddit, Bluesky, GDELT, RSS/Google News in parallel |
| 3 Claim Matcher | fast | clusters items into variants; labels mutations; code entity-diffs numbers/years/names |
| 4 Origin Tracer | - | Wayback first-capture dates, propagation graph, probabilistic origin, timeline, amplifiers |
| 5 Evidence Collection | - | citation chains to court/gov/dataset/paper/press release, official-domain pages, SEC EDGAR, user `--primary` URLs, ranked by source weight |
| 6 Fact Checker | smart | sub-claim graph; verdicts from retrieved sources only; supporting / contradicting / missing context |
| 7 Devil's Advocate | adversary | outdated? satire? context removed? alternative reading? is the origin just API coverage? |
| 8 Judge | smart | reconciles 6 and 7; code assembles final JSON |

**Confidence is computed, not self-reported:** `0.15 + .30*evidence_quality + .20*agreement + .25*primary_source
- .30*contradiction - .25*missing`, capped at 0.6 with no primary-type source, minus 0.10 per unanswered
high-severity attack. **Origin probabilities** are a softmax over `0.35*time + 0.30*citation + 0.15*repost +
0.20*originality`, with explicit probability mass on "origin never observed". Both are heuristics, not calibrated.

Mutation labels: exaggeration, missing_context, misattribution, fabrication, quote_distortion, date_shift,
entity_swap, partial_truth (+ number_shift, satire_literal, original, other).

## LLMs: Gemini and Groq only
Tiers map to (provider, model): `fast` (agents 1, 3) and `smart` (6, 8) default to Gemini 2.5 Flash; `adversary` (7) defaults to Groq `openai/gpt-oss-120b`, so the challenger is a different model family from the fact checker. Override any tier with `<TIER>_PROVIDER` / `<TIER>_MODEL` (see `.env.example`). Groq's free tier is only about 6k tokens/min, so prompts over `GROQ_MAX_INPUT_CHARS` are sent to Gemini instead, and with both keys set a rate-limited or failing call falls back to the other provider. Groq model IDs change often: list yours with `curl -H "Authorization: Bearer $GROQ_API_KEY" https://api.groq.com/openai/v1/models` and override if a default is missing. Needs `GEMINI_API_KEY` and/or `GROQ_API_KEY`.

## Run

```bash
pip install -r requirements.txt
cp .env.example .env && set -a && . ./.env && set +a
python -m tracer.cli "Apple is shutting down iCloud" --rounds 2 --out report      # no DB needed
docker compose up -d db && uvicorn tracer.api:app --reload                         # API + Postgres
curl -X POST localhost:8000/runs -H 'content-type: application/json' -d '{"claim":"Apple is shutting down iCloud"}'
curl localhost:8000/runs/<id>/events ; curl localhost:8000/runs/<id>/report
python tests/test_smoke.py                                                         # offline end-to-end test
```
Postgres holds everything relationally (`runs`, `events`, `evidence`, `variants`, `refs`, final result as JSONB).
No graph DB: variants carry `parent`, evidence carries `variant`; add Neo4j only if graph queries get hairy.

## Status / known limits
* Tested offline only (fake LLM + fake HTTP): the pipeline, entity diffing, origin scoring, confidence maths,
  link extraction, report. **Not yet run against live APIs**, nor against real LangGraph / FastAPI / SQLAlchemy / Gemini
  / Playwright / Trafilatura (not installable where this was built). Expect small integration fixes on first run.
* Reddit search is often 403 without OAuth; Bluesky search usually needs an app password; GDELT DOC covers ~3 months.
* `official_domains` are LLM-suggested and only used as GDELT filters; a page counts as a press release only if the
  domain also matches a named organisation. The SEC EDGAR full-text endpoint parsing is best-effort.
* Not built: image provenance (pHash / reverse search / EXIF / OCR), source-reliability priors, Next.js frontend.

## First-time setup (SQLite, free-tier keys)
1. `python -m venv .venv && source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
2. `pip install -r requirements.txt`
3. `cp .env.example .env`, open `.env`, paste `GEMINI_API_KEY`, `GROQ_API_KEY`, `BSKY_HANDLE`, `BSKY_APP_PASSWORD`. Leave `DATABASE_URL` commented out or unset: it defaults to SQLite (`tracer.db`). Never commit `.env`.
4. `python check_setup.py` (tests each key and lists your Groq models)
5. `python -m tracer.cli "Apple is shutting down iCloud" --rounds 1 --out report` (use `--rounds 1` on free tiers)
6. Optional API: `uvicorn tracer.api:app --reload`, then open http://localhost:8000/docs
