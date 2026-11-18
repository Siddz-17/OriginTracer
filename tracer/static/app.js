/* OriginTracer UI. Vanilla JS, no dependencies.
   API: POST /runs, GET /runs, GET /runs/{id}, GET /runs/{id}/events?after=N, GET /runs/{id}/report, GET /health */
"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const safeUrl = (u) => (/^https?:\/\//i.test(u || "") ? u : null);
const link = (u, text) => (safeUrl(u) ? `<a href="${esc(u)}" target="_blank" rel="noopener noreferrer">${esc(text)}</a>` : esc(text));
const host = (u) => { try { return new URL(u).hostname.replace(/^www\./, ""); } catch { return ""; } };
const fmtDate = (d) => { if (!d) return "–"; const t = new Date(d); return isNaN(t) ? String(d).slice(0, 10) : t.toISOString().slice(0, 10); };
const pct = (x) => `${Math.round((x || 0) * 100)}%`;
const clip = (s, n = 140) => { s = String(s || ""); if (s.length <= n) return s; const c = s.slice(0, n); return c.slice(0, Math.max(c.lastIndexOf(" "), n - 20)).replace(/[\s.,;:–-]+$/, "") + "…"; };

const VERDICT = {
  true: ["True", "true"], mostly_true: ["Mostly true", "true"], misleading: ["Misleading", "mixed"],
  mostly_false: ["Mostly false", "false"], false: ["False", "false"], unverifiable: ["Unverifiable", "unv"],
};
const SUB = { supported: ["Supported", "true"], refuted: ["Refuted", "false"], misleading: ["Misleading", "mixed"], unverifiable: ["Unverifiable", "unv"] };
const STANCE = { debunk: "Debunks", support: "Supports", report: "Reports", joke: "Satire", assert: "Reports" };
const TIER = { fact_check: "Fact-checker", wire: "Wire", major: "Major outlet" };
const KIND = { government: "Government", court: "Court", dataset: "Dataset", peer_reviewed: "Study", press_release: "Press release",
  reference: "Wikipedia", institutional: "Institution", preprint: "Preprint", social: "Social", fact_check: "Fact-check", news: "News", web: "Web" };
// pipeline agents -> progress steps
const STEP_OF = { pipeline: "search", expansion: "search", discovery: "search", matcher: "match", origin: "origin",
  evidence: "evidence", image: "search", analysis: "analysis", factcheck: "analysis", devil: "analysis", judge: "analysis" };
const STEPS = ["search", "match", "origin", "evidence", "analysis"];

const state = { runId: null, timer: null, clock: null, started: 0, lastEvent: 0, result: null, filter: "all" };

/* ---------------------------------------------------------------- server status */
async function checkHealth() {
  const el = $("#health");
  try {
    const h = await (await fetch("/health")).json();
    const hasKey = h.llm_key_configured;
    if (hasKey === false) {
      el.className = "status-dot err";
      el.textContent = "No AI key on server";
      el.title = "Set GROQ_API_KEY in the server environment. Evidence is still collected, but no verdict is made.";
    } else {
      el.className = "status-dot ok";
      el.textContent = h.llm_provider ? `Ready · ${h.llm_provider}` : "Ready";
      el.title = "Server is up";
    }
  } catch {
    el.className = "status-dot err"; el.textContent = "Server unreachable";
  }
}

/* ---------------------------------------------------------------- starting / polling a run */
async function startRun(claim) {
  clearRun();
  const body = { claim, rounds: +$("#rounds").value, depth: +$("#depth").value, archive: true };
  const img = $("#image-url").value.trim();
  if (img) body.image_url = img;
  setBusy(true);
  try {
    const res = await fetch("/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!res.ok) {
      let msg = `HTTP ${res.status}`;
      try { const j = await res.json(); msg = Array.isArray(j.detail) ? j.detail.map((d) => d.msg).join("; ") : (j.detail || msg); } catch {}
      throw new Error(msg);
    }
    const { id } = await res.json();
    history.pushState({}, "", `?run=${encodeURIComponent(id)}`);
    follow(id, claim);
  } catch (e) {
    setBusy(false);
    showError("Could not start the run", e.message);
  }
}

function follow(id, claim) {
  state.runId = id; state.lastEvent = 0; state.started = Date.now();
  $("#progress").classList.remove("hidden");
  $("#progress-title").textContent = "Tracing…";
  $("#progress-claim").textContent = claim ? `“${claim}”` : "";
  $("#log").textContent = "";
  setStep(null);
  setBusy(true);
  clearInterval(state.clock);
  state.clock = setInterval(() => {
    const s = Math.floor((Date.now() - state.started) / 1000);
    $("#elapsed").textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  }, 1000);
  poll();
}

async function poll() {
  clearTimeout(state.timer);
  const id = state.runId;
  try {
    const evs = await (await fetch(`/runs/${id}/events?after=${state.lastEvent}`)).json();
    for (const e of evs) {
      state.lastEvent = Math.max(state.lastEvent, e.id);
      logLine(e.agent, e.message);
      if (STEP_OF[e.agent]) setStep(STEP_OF[e.agent]);
      const m = /"news_articles": (\d+)/.exec(e.message);
      if (m) $("#progress-count").textContent = `${m[1]} articles found so far`;
    }
    const run = await (await fetch(`/runs/${id}`)).json();
    if (id !== state.runId) return;
    if (run.status === "done") return finish(run);
    if (run.status === "failed") {
      stopClock(); setBusy(false);
      $("#progress-title").textContent = "Run failed";
      return showError("The run failed", run.error || "Unknown error. See the activity log.");
    }
  } catch (e) {
    logLine("ui", `connection problem: ${e.message}; retrying`);
  }
  state.timer = setTimeout(poll, 2500);
}

function finish(run) {
  stopClock(); setBusy(false);
  setStep("done");
  $("#progress").classList.add("hidden");
  let result = run.result;
  if (typeof result === "string") { try { result = JSON.parse(result); } catch { result = {}; } }
  render(result || {}, run.id);
  $("#report").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function loadRun(id) {
  clearRun();
  try {
    const res = await fetch(`/runs/${encodeURIComponent(id)}`);
    if (!res.ok) throw new Error(res.status === 404 ? "This run no longer exists (the server's database may have been reset)." : `HTTP ${res.status}`);
    const run = await res.json();
    $("#claim").value = run.claim || "";
    if (run.status === "done") finish(run);
    else if (run.status === "failed") showError("The run failed", run.error || "Unknown error");
    else follow(run.id, run.claim);
  } catch (e) {
    showError("Could not load the run", e.message);
  }
}

function clearRun() {
  clearTimeout(state.timer); stopClock();
  state.runId = null; state.result = null;
  $("#error-box").innerHTML = "";
  $("#report").classList.add("hidden");
  $("#progress").classList.add("hidden");
}
function stopClock() { clearInterval(state.clock); }
function setBusy(b) { $("#trace-btn").disabled = b; $("#trace-btn").textContent = b ? "Tracing…" : "Trace claim"; }
function setStep(step) {
  const idx = step === "done" ? STEPS.length : STEPS.indexOf(step);
  $$("#steps .step").forEach((el, i) => {
    el.classList.toggle("done", idx > i);
    el.classList.toggle("active", idx === i);
  });
}
function logLine(agent, msg) {
  const box = $("#log");
  const d = document.createElement("div");
  d.innerHTML = `<span class="ag">${esc(agent)}</span> ${esc(msg)}`;
  box.appendChild(d);
  box.scrollTop = box.scrollHeight;
}
function showError(title, detail) {
  $("#error-box").innerHTML = `<div class="alert err" role="alert"><strong>${esc(title)}</strong>${esc(detail)}</div>`;
}

/* ---------------------------------------------------------------- report */
function render(r, runId) {
  state.result = r; state.runId = runId;
  $("#report").classList.remove("hidden");
  renderVerdict(r);
  renderEvidence();
  renderOrigin(r);
  renderSubclaims(r);
  renderImage(r.image);
  $("#raw").textContent = JSON.stringify(r, null, 2);
  const md = $("#dl-md");
  if (runId) { md.href = `/runs/${encodeURIComponent(runId)}/report`; md.classList.remove("hidden"); } else md.classList.add("hidden");
  document.title = `${(VERDICT[r.verdict] || VERDICT.unverifiable)[0]}: ${r.claim || "claim"} · OriginTracer`;
}

function renderVerdict(r) {
  const [label, cls] = VERDICT[r.verdict] || VERDICT.unverifiable;
  $("#v-claim").textContent = r.claim || "";
  const badge = $("#v-badge");
  badge.className = `badge ${cls}`; badge.textContent = label;
  const rationale = r.rationale || "";
  const llmFailed = /^The LLM analysis step failed/.test(rationale);
  $("#v-rationale").textContent = llmFailed ? "" : rationale;

  const notes = [];
  if (llmFailed) {
    const reason = (/failed \((.*?)\)\. /.exec(rationale) || [])[1] || "unknown error";
    let hint = "The evidence below was still collected.";
    if (/API key|not set|No LLM/i.test(reason)) hint = "The server has no AI key: set GROQ_API_KEY in its environment and redeploy.";
    else if (/413|too large|TPM/i.test(reason)) hint = "The request was too large for the AI provider's limit. Try again; it retries with less evidence.";
    else if (/429|rate|quota/i.test(reason)) hint = "The AI provider's free quota is used up for now. Try again later.";
    notes.push(["warn", `<strong>AI analysis did not run.</strong> ${esc(hint)} <span class="faint">(${esc(reason.slice(0, 160))})</span>`]);
  }
  for (const c of r.caveats || []) {
    if (/^LLM analysis unavailable/.test(c)) continue;
    notes.push([/disagree|contradict/i.test(c) ? "warn" : "info", esc(c)]);
  }
  const mc = (r.fact_check || {}).missing_context || [];
  if (mc.length) notes.push(["info", `<strong>Missing context:</strong> ${mc.map(esc).join(" · ")}`]);
  $("#v-notes").innerHTML = notes.map(([k, h]) => `<li class="${k}">${h}</li>`).join("");

  const conf = r.confidence ?? 0;
  const sc = r.scores || {};
  $("#c-num").innerHTML = `${pct(conf)} <small>${conf >= 0.65 ? "high" : conf >= 0.35 ? "moderate" : "low"}</small>`;
  $("#c-bar").style.width = pct(conf);
  $("#c-meter").setAttribute("aria-valuenow", Math.round(conf * 100));
  const why = [];
  if (sc.primary_sources) why.push(`${sc.primary_sources} primary source(s) cited (official, court, data or study)`);
  else why.push(`No primary source cited, so confidence is capped at ${pct(sc.cap ?? 0.6)}`);
  if (sc.fact_checkers_agreeing) why.push(`${sc.fact_checkers_agreeing} independent fact-checker(s) agree`);
  if (sc.secondary_sources) why.push(`${sc.secondary_sources} news / secondary source(s) cited`);
  if (sc.contradictions) why.push(`${sc.contradictions} source(s) point the other way`);
  if (sc.fact_check_conflict) why.push("Fact-checkers disagree with this verdict (−15%)");
  if (sc.after_critique != null && sc.confidence != null && sc.after_critique < sc.confidence)
    why.push(`Self-critique found weaknesses (−${pct(sc.confidence - sc.after_critique)})`);
  $("#c-why").innerHTML = why.map((w) => `<li>${esc(w)}</li>`).join("");
}

function renderEvidence() {
  const r = state.result || {};
  const all = r.evidence_ranked || [];
  const f = state.filter;
  const rows = all.filter((x) => f === "all" || (f === "fact_check" ? x.tier === "fact_check" || x.kind === "fact_check" : x.stance === f));
  $("#ev-count").textContent = `${all.length} sources`;
  $("#ev-rows").innerHTML = rows.map((x) => {
    const tier = TIER[x.tier] || (x.kind && x.kind !== "news" ? KIND[x.kind] || x.kind : "");
    const st = x.stance ? `<span class="stance ${esc(x.stance)}">${esc(STANCE[x.stance] || x.stance)}</span>` : `<span class="faint">–</span>`;
    return `<tr><td class="date">${esc(fmtDate(x.date))}</td>
      <td class="outlet">${esc(x.outlet || host(x.url))}${tier ? `<span class="tier">${esc(tier)}</span>` : ""}</td>
      <td>${link(x.url, clip(x.title || x.url))}</td><td>${st}</td></tr>`;
  }).join("") || `<tr><td colspan="4" class="muted">No sources match this filter.</td></tr>`;
}

function renderOrigin(r) {
  const o = r.origin || {};
  const c = o.likely;
  let h = "";
  if (c) {
    h += `<div class="lab muted">Earliest item found that states the claim</div>
      <div class="when">${esc(fmtDate(c.ts))}</div>
      <div class="who">${esc(c.author || "")} · ${esc(c.source || "")}</div>
      <blockquote>${link(c.url, c.text || c.url)}</blockquote>
      <dl class="kv"><dt>How sure</dt><dd>${originStrength(o)}</dd>
      <dt>Real origin not found</dt><dd>${pct(o.p_unobserved)} chance</dd></dl>`;
  } else {
    h += `<p class="muted">No item clearly stating the claim was found, so no origin could be estimated.</p>`;
  }
  if (o.statement) h += `<p style="margin:10px 0 0;font-size:14px">${esc(o.statement)}</p>`;
  if ((o.evidence || []).length) h += `<ul class="bullets">${o.evidence.map((b) => `<li>${esc(b)}</li>`).join("")}</ul>`;
  $("#or-main").innerHTML = h;
  $("#chart").innerHTML = timelineChart(r.timeline || [], c && c.ts);

  const amp = (r.amplifiers || []).slice(0, 6);
  $("#amp").innerHTML = amp.map((a) => `<li><span class="grow">${esc(a.author || "unknown")} <span class="faint">· ${esc(a.source)}</span></span>
    <span class="muted">${a.items} item${a.items === 1 ? "" : "s"}${a.engagement ? ` · ${a.engagement} eng.` : ""}</span></li>`).join("")
    || `<li class="muted">None found</li>`;
  const nodes = ((r.mutation_graph || {}).nodes || []).filter((n) => n.n_items || n.id === "V0");
  $("#variants").innerHTML = nodes.map((n) => `<li><span class="pill ${n.label === "original" ? "unv" : "mixed"}">${esc(n.label === "other" || !n.label ? "variant" : n.label.replace("_", " "))}</span>
    <span class="grow" title="${esc(n.text)}">${esc((n.text || "").slice(0, 110))}</span><span class="muted">${n.n_items || 0}</span></li>`).join("")
    || `<li class="muted">Only one version found</li>`;

  const d = r.discovery || {};
  const labels = { news_articles: "News articles", reputable_outlet_items: "From reputable outlets", recent_news: "Last 14 days",
    web_results: "Web results", bluesky_posts: "Bluesky posts", reddit_posts: "Reddit posts", mastodon_posts: "Mastodon posts",
    hackernews_posts: "Hacker News" };
  const stats = Object.entries(labels).filter(([k]) => d[k]).map(([k, l]) => `<div class="stat"><b>${d[k]}</b><span>${l}</span></div>`);
  stats.push(`<div class="stat"><b>${(r.timeline || []).length}</b><span>Matched the claim</span></div>`);
  $("#coverage").innerHTML = stats.join("");
}

function originStrength(o) {
  // candidate probabilities are split across near-identical early reports, so a low number is normal
  const cands = o.candidates || [];
  const t = o.likely && o.likely.ts ? String(o.likely.ts).slice(0, 10) : null;
  const sameDay = t ? cands.filter((c) => String(c.ts).slice(0, 10) === t).length : 0;
  const p = o.probability || 0;
  const level = (o.p_unobserved || 0) >= 0.5 ? "Weak" : p >= 0.4 ? "Strong" : p >= 0.15 ? "Moderate" : "Weak";
  const why = sameDay > 1 ? `${sameDay} similar reports appeared the same day; this is the earliest of them`
    : (o.p_unobserved || 0) >= 0.5 ? "the claim probably circulated before anything found here" : `${pct(p)} of the origin score`;
  return `<strong>${level}</strong> <span class="faint">· ${esc(why)}</span>`;
}

function timelineChart(items, originTs) {
  let pts = items.map((x) => ({ t: new Date(x.time), s: x.stance || "report" })).filter((x) => !isNaN(x.t));
  if (pts.length < 2) return `<p class="muted">Not enough dated items for a timeline.</p>`;
  pts.sort((a, b) => a.t - b.t);
  // zoom to where the coverage is: a few stray old articles would otherwise squash the story into one bar
  let hidden = 0;
  const ot0 = originTs ? new Date(originTs) : null;
  // window starts two weeks before the estimated origin (the story's start), else at the 10th percentile
  const q = ot0 && !isNaN(ot0) ? new Date(ot0 - 14 * 864e5) : pts[Math.floor(pts.length * 0.1)].t;
  if (pts.length >= 20 && (q - pts[0].t) / 864e5 > 30 && pts.filter((p) => p.t >= q).length >= 2) {
    const keep = pts.filter((p) => p.t >= q);
    hidden = pts.length - keep.length;
    pts = keep;
  }
  const t0 = pts[0].t, t1 = pts[pts.length - 1].t;
  const spanDays = (t1 - t0) / 864e5;
  const unit = spanDays <= 45 ? 1 : spanDays <= 400 ? 7 : spanDays <= 3000 ? 30.4 : 365;
  const unitName = { 1: "day", 7: "week", 30.4: "month", 365: "year" }[unit];
  const nb = Math.max(1, Math.min(60, Math.ceil(spanDays / unit) + 1));
  const bins = Array.from({ length: nb }, () => ({ report: 0, debunk: 0, support: 0, joke: 0 }));
  for (const p of pts) {
    const i = Math.min(nb - 1, Math.floor((p.t - t0) / 864e5 / unit));
    const k = p.s === "assert" ? "report" : (bins[i][p.s] !== undefined ? p.s : "report");
    bins[i][k]++;
  }
  const max = Math.max(...bins.map((b) => b.report + b.debunk + b.support + b.joke));
  const W = 520, H = 170, pad = 22, bw = (W - pad * 2) / nb;
  const col = { report: "var(--report)", debunk: "var(--debunk)", support: "var(--support)", joke: "url(#ot-hatch)" };
  let bars = "";
  bins.forEach((b, i) => {
    let y = H - pad;
    for (const k of ["report", "support", "debunk", "joke"]) {
      if (!b[k]) continue;
      const h = (b[k] / max) * (H - pad * 2);
      y -= h;
      bars += `<rect x="${(pad + i * bw + 1).toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(1, bw - 2).toFixed(1)}" height="${h.toFixed(1)}" fill="${col[k]}" stroke="var(--ink)" stroke-width="1.5"><title>${b[k]} ${STANCE[k] || k}</title></rect>`;
    }
  });
  let marker = "";
  const ot = originTs ? new Date(originTs) : null;
  if (ot && !isNaN(ot) && ot >= t0 && ot <= t1) {
    const x = pad + ((ot - t0) / 864e5 / unit) * bw + bw / 2;
    marker = `<line x1="${x.toFixed(1)}" x2="${x.toFixed(1)}" y1="${pad - 6}" y2="${H - pad}" stroke="var(--ink)" stroke-width="2" stroke-dasharray="5 4"/>
      <text x="${x > W - 70 ? (x - 4).toFixed(1) : (x + 4).toFixed(1)}" y="${pad}" text-anchor="${x > W - 70 ? "end" : "start"}">origin</text>`;
  }
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Matched items per ${unitName}, ${pts.length} in total">
    <defs><pattern id="ot-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
      <rect width="6" height="6" fill="var(--paper)"/><line x1="0" y1="0" x2="0" y2="6" stroke="var(--ink)" stroke-width="2"/></pattern></defs>
    <line x1="${pad}" x2="${W - pad}" y1="${H - pad}" y2="${H - pad}" stroke="var(--ink)" stroke-width="2"/>${bars}${marker}
    <text x="${pad}" y="${H - 6}">${fmtDate(t0)}</text><text x="${W - pad}" y="${H - 6}" text-anchor="end">${fmtDate(t1)}</text>
    <text x="${pad}" y="12">up to ${max} items per ${unitName}</text></svg>`
    + (hidden ? `<p class="faint" style="font-size:12.5px;margin:4px 0 0">${hidden} older item(s) outside this window not shown.</p>` : "");
}

function renderSubclaims(r) {
  const fc = r.fact_check || {};
  const byId = {};
  for (const x of r.evidence_ranked || []) byId[x.id] = x;
  for (const x of [...(fc.supporting_evidence || []), ...(fc.contradicting_evidence || [])]) byId[x.id] = byId[x.id] || x;
  const cite = (id, cls) => {
    const x = byId[id] || {};
    const label = x.outlet || host(x.url) || id;
    const title = x.title ? `${label}: ${x.title}` : label;
    return safeUrl(x.url) ? `<a class="cite ${cls}" href="${esc(x.url)}" target="_blank" rel="noopener noreferrer" title="${esc(title)}">${esc(label)}</a>`
      : `<span class="cite ${cls}" title="${esc(title)}">${esc(label)}</span>`;
  };
  const subs = fc.subclaims || [];
  $("#sc-list").innerHTML = subs.map((s) => {
    const [lab, cls] = SUB[s.verdict] || SUB.unverifiable;
    const cites = [...(s.supporting || []).map((i) => cite(i, "for")), ...(s.dissenting || []).map((i) => cite(i, "against"))].join("");
    return `<div class="subclaim"><div class="subclaim-head"><span class="pill ${cls}">${lab}</span><span class="subclaim-text">${esc(s.text)}</span></div>
      <p>${esc(s.reasoning)}</p>${cites ? `<div class="cites">${cites}</div>` : ""}</div>`;
  }).join("") || `<p class="muted">No sub-claims (the AI analysis did not run).</p>`;
  $("#missing").innerHTML = r.primary_hint ? `<p class="muted" style="font-size:14px"><strong>What would settle it:</strong> ${esc(r.primary_hint)}</p>` : "";
}

function renderImage(img) {
  const show = !!img;
  $("#s-image").classList.toggle("hidden", !show);
  $("#tab-image").classList.toggle("hidden", !show);
  if (!show) return;
  if (img.error) { $("#im-body").innerHTML = `<div class="alert err">${esc(img.error)}</div>`; return; }
  const ex = img.exif || {};
  const kv = [["Size", img.width ? `${img.width}×${img.height} ${img.format || ""}` : "–"], ["Perceptual hash", img.phash],
    ["Captured (EXIF)", ex.DateTimeOriginal], ["Camera", [ex.Make, ex.Model].filter(Boolean).join(" ")],
    ["Software", ex.Software], ["Reverse search", img.reverse_engine || "not configured"]].filter(([, v]) => v);
  const ms = (img.matches || []).slice(0, 10);
  const links = Object.entries(img.manual_search || {}).filter(([, v]) => safeUrl(v)).map(([k, v]) => link(v, k.replace("_", " "))).join(" · ");
  $("#im-body").innerHTML = `<dl class="kv">${kv.map(([k, v]) => `<dt>${esc(k)}</dt><dd class="${k === "Perceptual hash" ? "mono" : ""}">${esc(v)}</dd>`).join("")}</dl>
    ${(img.notes || []).length ? `<ul class="bullets">${img.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>` : ""}
    ${img.ocr_text ? `<p style="font-size:14px"><strong>Text in image:</strong> “${esc(img.ocr_text.slice(0, 300))}”</p>` : ""}
    ${ms.length ? `<div class="table-wrap" style="margin-top:10px"><table><thead><tr><th>First seen</th><th>Site</th><th>Match</th><th>Page</th></tr></thead><tbody>
      ${ms.map((m) => `<tr><td class="date">${esc(fmtDate(m.first_seen))}</td><td>${esc(m.site)}</td><td>${esc(m.relation || "?")}</td><td>${link(m.url, m.title || m.url)}</td></tr>`).join("")}
      </tbody></table></div>` : ""}
    ${links ? `<p style="font-size:14px">Search manually: ${links}</p>` : ""}`;
}

/* ---------------------------------------------------------------- history */
async function toggleHistory(force) {
  const box = $("#history");
  const open = force ?? box.classList.contains("hidden");
  box.classList.toggle("hidden", !open);
  $("#history-btn").setAttribute("aria-expanded", String(open));
  if (!open) return;
  const ul = $("#history-list");
  ul.innerHTML = `<li class="muted">Loading…</li>`;
  try {
    const runs = await (await fetch("/runs")).json();
    ul.innerHTML = runs.map((r) => {
      const [lab, cls] = VERDICT[r.verdict] || (r.status === "done" ? VERDICT.unverifiable : [r.status, "unv"]);
      return `<li><button data-id="${esc(r.id)}"><span class="pill ${cls}">${esc(lab)}</span><span class="grow">${esc(r.claim)}</span>
        <span class="muted">${esc(fmtDate(r.created_at))}</span></button></li>`;
    }).join("") || `<li class="muted">No runs yet.</li>`;
  } catch (e) {
    ul.innerHTML = `<li class="muted">Could not load runs: ${esc(e.message)}</li>`;
  }
}

/* ---------------------------------------------------------------- wiring */
function init() {
  checkHealth();
  $("#trace-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const claim = $("#claim").value.trim();
    if (claim.length >= 3) startRun(claim);
  });
  $("#examples").addEventListener("click", (e) => {
    if (e.target.matches("button.chip")) { $("#claim").value = e.target.textContent; $("#claim").focus(); }
  });
  $("#ev-filter").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-f]"); if (!b) return;
    state.filter = b.dataset.f;
    $$("#ev-filter .chip").forEach((c) => c.classList.toggle("active", c === b));
    renderEvidence();
  });
  $("#history-btn").addEventListener("click", () => toggleHistory());
  $("#history-close").addEventListener("click", () => toggleHistory(false));
  $("#history-list").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-id]"); if (!b) return;
    toggleHistory(false);
    history.pushState({}, "", `?run=${encodeURIComponent(b.dataset.id)}`);
    loadRun(b.dataset.id);
  });
  $("#copy-link").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(location.href); $("#copy-link").textContent = "Link copied"; }
    catch { $("#copy-link").textContent = "Copy failed"; }
    setTimeout(() => ($("#copy-link").textContent = "Copy link to this report"), 1800);
  });
  $("#dl-json").addEventListener("click", () => {
    const blob = new Blob([JSON.stringify(state.result || {}, null, 2)], { type: "application/json" });
    const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: `origintracer-${state.runId || "report"}.json` });
    a.click(); URL.revokeObjectURL(a.href);
  });
  // highlight the section in view
  const tabs = $$("#tabs a");
  $("#tabs").addEventListener("click", (e) => {  // explicit scroll: re-clicking the current hash must still work
    const a = e.target.closest("a[href^='#']"); if (!a) return;
    const target = document.getElementById(a.getAttribute("href").slice(1)); if (!target) return;
    e.preventDefault();
    const top = target.getBoundingClientRect().top + window.scrollY - 110;
    window.scrollTo({ top, behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
    tabs.forEach((t) => t.classList.toggle("on", t === a));
  });
  let ticking = false;
  window.addEventListener("scroll", () => {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(() => {
      ticking = false;
      const secs = $$("#report > section").filter((s) => !s.classList.contains("hidden"));
      let cur = secs[0];
      for (const s of secs) if (s.getBoundingClientRect().top <= 150) cur = s;
      if (window.innerHeight + window.scrollY >= document.body.scrollHeight - 4) cur = secs[secs.length - 1];
      tabs.forEach((a) => a.classList.toggle("on", cur && a.getAttribute("href") === `#${cur.id}`));
    });
  }, { passive: true });
  window.addEventListener("popstate", route);
  route();
}

function route() {
  const id = new URLSearchParams(location.search).get("run");
  if (id) loadRun(id); else clearRun();
}

document.addEventListener("DOMContentLoaded", init);
window.OriginTracerUI = { render }; // for previews / debugging
