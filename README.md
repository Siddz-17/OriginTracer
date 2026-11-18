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

**Confidence is computed, not self-reported:** per sub-claim `0.15 + .30*evidence_quality + .20*agreement
+ .25*primary_source + .15*corroboration - .30*contradiction - .25*missing`, where corroboration counts independent
publishers (not repeated items). It is capped at 0.6 with no primary-type source, 0.7 with one agreeing fact-checker
and 0.8 with two or more, then lowered by 0.10 / 0.04 per high / medium self-critique attack and by 0.15 when
the verdict disagrees with the fact-checkers' consensus. **Origin probabilities** are a softmax over `0.35*time +
0.30*citation + 0.15*repost + 0.20*originality`, with explicit probability mass on "origin never observed" (at
least 50% when only debunks were found). Both are heuristics, not calibrated.

### Accuracy safeguards
* **Claim-aware stance.** Stance is read from the headline / lead only. Strong cues ("hoax", "no evidence",
  "…? No") mark a debunk; negations ("did not", "denies") count only next to the claim's own predicate words,
  so "Chandrayaan-3 lander did not wake up" is not a debunk of "Chandrayaan-3 landed".
* **Fact-check ratings.** Ratings (ClaimReview `textualRating`, "Rating: Mostly True", "- False", "Pants on
  Fire") are normalised to false / misleading / true / satire / unproven. A fact-check rating a claim *true* gets
  stance `support`, not `debunk`. Ratings are passed to the LLM and voted one per publisher.
* **Matching.** Stemmed synonyms ("discontinues" matches "shutting down"), number words ("six" = "6"),
  whole-word name matching ("apple" never matches "pineapple").
* **Verdict checks.** Evidence the model filed on the wrong side (debunks listed as "dissenting" from a
  'refuted' verdict) is re-oriented by stance. A true-side verdict over all-refuted sub-claims (or the reverse)
  takes the sub-claims' verdict; disagreement with >= 2 agreeing fact-checkers is flagged as a caveat. If the LLM
  is down and >= 2 independent fact-checkers agree, their rating becomes the verdict instead of `unverifiable`.
* **Origin.** Candidates must state the claim's predicate (not just its names), may not be a preview of a
  past-tense event ("India plans to land…"), and are never fact-checks. The single LLM call reviews the top 5
  candidates and rejects ones that do not assert the claim; origin is then re-scored. Wayback captures more
  than `WAYBACK_MAX_LEAD_DAYS` (45) before an article's date are ignored as URL reuse.

## Optional data sources (free keys, switched on by `.env`)
| Key | Adds |
|---|---|
| `GOOGLE_FACTCHECK_API_KEY` | ClaimReview ratings from fact-checkers worldwide, plus the date the claim was made and its claimant (shown as origin evidence; an earlier claim date raises "origin unobserved") |
| `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` | Reddit search over OAuth (anonymous search is usually 403) |
| `BSKY_HANDLE` / `BSKY_APP_PASSWORD` | Bluesky search (the public endpoint is usually auth-gated) |
| `BRAVE_API_KEY`, `TAVILY_API_KEY` or `SERPER_API_KEY` | Dated web search results, more reliable than scraped DuckDuckGo |
| `TINEYE_API_KEY` or `SERPAPI_API_KEY` | Reverse image search for `--image` |

First-seen dates now come from the Wayback Machine **and** archive.today (earliest wins; no key needed).

## Local models (optional, CPU, no key)
`pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements-models.txt`
turns on, automatically (`TRACER_LOCAL_MODELS=auto`):
* **Embedding similarity** (`all-MiniLM-L6-v2`) in claim matching: keyword matches below 0.45 similarity are
  dropped ("Bill Gates did not promote water cryptocoin" shares a name, not the claim: 0.44), and paraphrases with
  few shared words but similarity >= 0.65 are added. Matched relevance blends keywords and similarity.
* **NLI** (`nli-deberta-v3-xsmall`) for stance, used only to ADD debunks: contradiction >= 0.8 on items with
  similarity >= 0.7. Probing showed the small NLI model also labels unrelated text "contradiction" (0.99 for
  "Pineapple farms shutting down" vs "Apple is shutting down iCloud"), so it is never trusted on its own and never
  overrides fact-check ratings or the cue rules.
Without the package everything falls back to the keyword heuristics. First use downloads ~170 MB of models.

## Image provenance
`python -m tracer.cli "This photo shows the 2026 floods in X" --image https://example.com/photo.jpg`
(API: `"image_url"`). Reports SHA-256 + perceptual hash, EXIF (capture date, camera, editing software, GPS),
OCR text (if `pytesseract` + tesseract are installed), and with a reverse-search key every page showing the image,
dated by TinEye's crawl date or the page's first archive capture, and labelled same / variant (cropped or edited) /
different by pHash distance. Without a key it returns ready-made Google Lens / TinEye / Yandex / Bing links. The
analysis is told that an image which existed before the event it supposedly shows is being used out of context.

## Evaluation
```bash
python eval/download_data.py                 # AVeriTeC dev split (500 real claims, CC BY-NC 4.0; not committed)
python eval/run_eval.py --n 20               # 20 stratified AVeriTeC claims + 14 curated claims, resumable
python eval/run_eval.py --n 60 --no-llm      # retrieval + fact-check consensus only (spends no LLM quota)
python eval/run_eval.py --report eval/results/run-n20-s7.jsonl
```
Reports side accuracy (true / false / mixed / unverifiable), decisive accuracy (gold true/false only), a confusion
matrix, mean confidence of right vs wrong verdicts, and origin-date error against the date the claim was made.
AVeriTeC claims were fact-checked, so retrieval can find the very article behind the label: the numbers are
optimistic for brand-new claims.

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
python tests/test_smoke.py                                                         # offline tests (12)
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
* Not built: source-reliability priors, video provenance (keyframes), Next.js frontend. Image reverse search
  needs a public image URL; local files get fingerprint / EXIF / OCR and manual search links only.

## First-time setup (SQLite, free-tier keys)
1. `python -m venv .venv && source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
2. `pip install -r requirements.txt`
3. `cp .env.example .env`, open `.env`, paste `OPENROUTER_API_KEY` (optionally `GROQ_API_KEY`, `BSKY_HANDLE`, `BSKY_APP_PASSWORD`). Leave `DATABASE_URL` commented out or unset: it defaults to SQLite (`tracer.db`). Never commit `.env`.
4. `python check_setup.py` (one test call; lists the live free OpenRouter models)
5. `python -m tracer.cli "Apple is shutting down iCloud" --out report` (single mode: one LLM call regardless of `--rounds`)
6. Optional API: `uvicorn tracer.api:app --reload`, then open http://localhost:8000/docs
