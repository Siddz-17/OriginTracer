# Narrative Origin Tracer

Give it a viral claim; it finds where the claim first appeared, how it mutated, who amplified it, and
fact-checks it against the best evidence it can find, then attacks its own verdict before reporting.

## Single-call consolidated analysis (default)
Free OpenRouter models are rate limited (~20 requests/min, 50/day without credits), so the default
`TRACER_MODE=single` spends **exactly one LLM call per run**. Everything before it is deterministic:

```
claim -> local query expansion (keywords, names, synonyms, "fact check"/"false" variants)
      -> discovery, reputable outlets first:
           fact-checkers (Snopes, PolitiFact, FactCheck.org, AFP, Reuters/AP fact check, Full Fact, Alt News...)
           wire services + major newsrooms (Reuters, AP, BBC, NYT, Guardian, WaPo, NPR, Al Jazeera, The Hindu...)
           via site-restricted Google News, plus Google News, GDELT, DuckDuckGo, Bluesky, optional Google Fact Check API
      -> local claim matching (weighted term coverage + synonyms) and stance (debunk / report / satire)
      -> retrieval round 2: pseudo-relevance-feedback queries from the matched headlines
      -> origin tracing (Wayback first captures, propagation graph) and variants (numbers/dates diffed)
      -> evidence: full article text from the top outlets, citation chains to primary sources, Wikipedia
      -> local ranking: 0.5 relevance + 0.25 source weight + 0.15 outlet tier + 0.1 debunk/primary bonus,
         max 2 items per domain, best passage per document, fits ANALYSIS_MAX_CHARS
      -> ONE LLM call: sub-claims + verdicts with cited ids, self-critique, final verdict, stance corrections
      -> code computes confidence; origin is re-scored with the corrected stances
```
If the model invents an evidence id it is dropped; if the LLM is unavailable the run still finishes with the
ranked evidence and an explicit `unverifiable` verdict. `TRACER_MODE=agents` (or `--mode agents`) keeps the
original 8-agent loop below.

| Agent (legacy `agents` mode) | Tier | What it does |
|---|---|---|
| 1 Query Expansion | fast | paraphrases + keyword queries + contradiction queries, orgs, official domains |
| 2 Source Discovery | - | outlets, fact-checkers, Google News, GDELT, DuckDuckGo, Bluesky in parallel |
| 3 Claim Matcher | fast | clusters items into variants; labels mutations; code entity-diffs numbers/years/names |
| 4 Origin Tracer | - | Wayback first-capture dates, propagation graph, probabilistic origin, timeline, amplifiers |
| 5 Evidence Collection | - | citation chains to court/gov/dataset/paper/press release, official-domain pages, SEC EDGAR, `--primary` URLs |
| 6 Fact Checker | smart | sub-claim graph; verdicts from retrieved sources only |
| 7 Devil's Advocate | adversary | outdated? satire? context removed? alternative reading? origin just API coverage? |
| 8 Judge | smart | reconciles 6 and 7; code assembles final JSON |

**Confidence is computed, not self-reported:** `0.15 + .30*evidence_quality + .20*agreement + .25*primary_source
- .30*contradiction - .25*missing`, capped at 0.6 with no primary-type source, minus 0.10 / 0.04 per high / medium
self-critique attack. **Origin probabilities** are a softmax over `0.35*time + 0.30*citation + 0.15*repost +
0.20*originality`, with explicit probability mass on "origin never observed" (at least 50% when only debunks were
found). Both are heuristics, not calibrated.

## LLMs: OpenRouter free models (Groq / Gemini optional fallbacks)
All tiers default to OpenRouter. A call walks `OPENROUTER_MODELS` in order: a retired model (404), a model that
rejects the request (400), an empty / non-JSON reply, a timeout or a per-model 429 moves straight on to the next
free model; an exhausted daily quota or a bad key stops OpenRouter at once. `response_format` is only sent to models
that advertise it, reasoning is excluded from the reply, and requests are spaced `OPENROUTER_MIN_INTERVAL` (3.2 s)
apart. `python check_setup.py` prints the live free-model list and flags configured models that disappeared.

## Run

```bash
pip install -r requirements.txt
cp .env.example .env && set -a && . ./.env && set +a
python -m tracer.cli "Apple is shutting down iCloud" --out report                 # no DB needed, 1 LLM call
docker compose up -d db && uvicorn tracer.api:app --reload                         # API + Postgres
curl -X POST localhost:8000/runs -H 'content-type: application/json' -d '{"claim":"Apple is shutting down iCloud"}'
curl localhost:8000/runs/<id>/events ; curl localhost:8000/runs/<id>/report
python tests/test_smoke.py                                                         # offline end-to-end test
```
Postgres holds everything relationally (`runs`, `events`, `evidence`, `variants`, `refs`, final result as JSONB).
No graph DB: variants carry `parent`, evidence carries `variant`; add Neo4j only if graph queries get hairy.

## Status / known limits
* Offline tests cover both modes (fake LLM + fake HTTP). Live retrieval (Google News outlet/fact-check search,
  GDELT, DuckDuckGo, Wayback, Wikipedia, article scraping) has been run end to end; the OpenRouter call itself needs
  your key (`python check_setup.py`).
* Reuters and AP block scrapers (401/403); their headlines and RSS snippets are still used, and the Wayback Machine
  is tried for the body. GDELT allows one request per 5 s, so its queries are serialised.
* Reddit search is often 403 without OAuth; Bluesky search usually needs an app password; GDELT DOC covers ~3 months.
* `official_domains` are LLM-suggested and only used as GDELT filters; a page counts as a press release only if the
  domain also matches a named organisation. The SEC EDGAR full-text endpoint parsing is best-effort.
* Not built: image provenance (pHash / reverse search / EXIF / OCR), source-reliability priors, Next.js frontend.

## First-time setup (SQLite, free-tier keys)
1. `python -m venv .venv && source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
2. `pip install -r requirements.txt`
3. `cp .env.example .env`, open `.env`, paste `OPENROUTER_API_KEY` (optionally `GROQ_API_KEY`, `BSKY_HANDLE`, `BSKY_APP_PASSWORD`). Leave `DATABASE_URL` commented out or unset: it defaults to SQLite (`tracer.db`). Never commit `.env`.
4. `python check_setup.py` (one test call; lists the live free OpenRouter models)
5. `python -m tracer.cli "Apple is shutting down iCloud" --out report` (single mode: one LLM call regardless of `--rounds`)
6. Optional API: `uvicorn tracer.api:app --reload`, then open http://localhost:8000/docs
