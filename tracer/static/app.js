// NOTHING OS (NT-01) - DASHBOARD JAVASCRIPT CONTROLLER

// Audio Synth for tactile clicks
class NothingSound {
  constructor() {
    this.enabled = true;
    this.ctx = null;
  }
  init() {
    if (!this.ctx) {
      try {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        this.ctx = new AudioCtx();
      } catch (e) {
        this.enabled = false;
      }
    }
  }
  click(freq = 600, duration = 0.02) {
    if (!this.enabled) return;
    this.init();
    if (!this.ctx) return;
    try {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = "sine";
      osc.frequency.setValueAtTime(freq, this.ctx.currentTime);
      gain.gain.setValueAtTime(0.08, this.ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + duration);
      osc.connect(gain);
      gain.connect(this.ctx.destination);
      osc.start();
      osc.stop(this.ctx.currentTime + duration);
    } catch (e) {}
  }
}

const sound = new NothingSound();

// Glyph Matrix Controller
class GlyphMatrix {
  constructor() {
    this.elements = [
      document.getElementById("glyph-cam-1"),
      document.getElementById("glyph-cam-2"),
      document.getElementById("glyph-diag"),
      document.getElementById("glyph-arc-left"),
      document.getElementById("glyph-arc-right"),
      document.getElementById("glyph-line"),
      document.getElementById("glyph-dot")
    ].filter(Boolean);
    this.red = document.getElementById("glyph-red");
    this.mode = document.getElementById("glyph-mode");
    this.animInterval = null;
  }

  setAll(state) {
    this.elements.forEach(el => {
      if (state) el.classList.add("lit");
      else el.classList.remove("lit");
    });
  }

  startSequence() {
    this.stop();
    if (this.mode) this.mode.textContent = "ACTIVE";
    if (this.red) this.red.classList.add("active");
    let step = 0;
    this.animInterval = setInterval(() => {
      this.elements.forEach((el, idx) => {
        if (idx === step % this.elements.length || idx === (step + 3) % this.elements.length) {
          el.classList.add("lit");
        } else {
          el.classList.remove("lit");
        }
      });
      step++;
    }, 180);
  }

  pulse() {
    this.setAll(true);
    sound.click(900, 0.04);
    setTimeout(() => this.setAll(false), 200);
  }

  stop() {
    clearInterval(this.animInterval);
    this.setAll(false);
    if (this.mode) this.mode.textContent = "IDLE";
    if (this.red) this.red.classList.remove("active");
  }
}

const glyph = new GlyphMatrix();

// State & UI References
const state = {
  activeRunId: null,
  pollTimer: null,
  rounds: 1,
  depth: 3,
  playwright: true,
  currentReportData: null,
  evidenceFilter: "all"
};

// DOM Elements
const dom = {
  claimInput: document.getElementById("claim-input"),
  clearBtn: document.getElementById("clear-input-btn"),
  startBtn: document.getElementById("start-trace-btn"),
  roundsOpts: document.querySelectorAll("#rounds-selector .pill-opt"),
  depthOpts: document.querySelectorAll("#depth-selector .pill-opt"),
  playwrightToggle: document.getElementById("playwright-toggle"),
  samplePills: document.querySelectorAll(".sample-pill"),
  globalLed: document.getElementById("global-led"),
  globalStatus: document.getElementById("global-status-text"),
  currentStage: document.getElementById("current-stage"),
  currentRound: document.getElementById("current-round"),
  clock: document.getElementById("clock-display"),
  loadSampleBtn: document.getElementById("load-sample-btn"),
  historySelect: document.getElementById("history-runs-select"),
  soundBtn: document.getElementById("sound-btn"),
  soundIcon: document.getElementById("sound-icon"),
  // Verdict
  verdictText: document.getElementById("verdict-text"),
  verdictRationale: document.getElementById("verdict-rationale"),
  verdictTag: document.getElementById("verdict-type-tag"),
  confVal: document.getElementById("confidence-val"),
  confCircle: document.getElementById("confidence-circle"),
  statPrimary: document.getElementById("stat-primary"),
  statSecondary: document.getElementById("stat-secondary"),
  statContra: document.getElementById("stat-contradictions"),
  // Subclaims
  subclaimsContainer: document.getElementById("subclaims-container"),
  subclaimsCount: document.getElementById("subclaims-count"),
  // Sources Metrics (Count Only)
  heroSourcesCount: document.getElementById("hero-sources-count"),
  sourcesTotalTag: document.getElementById("sources-total-tag"),
  sourcesRatioText: document.getElementById("sources-ratio-text"),
  barSlicePrimary: document.getElementById("bar-slice-primary"),
  barSliceSecondary: document.getElementById("bar-slice-secondary"),
  barSliceContra: document.getElementById("bar-slice-contra"),
  sstatTotal: document.getElementById("sstat-total"),
  sstatPrimary: document.getElementById("sstat-primary"),
  sstatSecondary: document.getElementById("sstat-secondary"),
  sstatContra: document.getElementById("sstat-contra"),
  // Raw
  rawDisplay: document.getElementById("raw-display"),
  tabMd: document.getElementById("tab-md-btn"),
  tabJson: document.getElementById("tab-json-btn"),
  copyBtn: document.getElementById("copy-report-btn"),
  backendPing: document.getElementById("backend-ping")
};

// Digital Clock
function updateClock() {
  const now = new Date();
  const timeStr = now.toTimeString().split(" ")[0] + " UTC";
  if (dom.clock) dom.clock.textContent = timeStr;
}
setInterval(updateClock, 1000);
updateClock();

// Sound Toggle
if (dom.soundBtn) {
  dom.soundBtn.addEventListener("click", () => {
    sound.enabled = !sound.enabled;
    dom.soundIcon.textContent = sound.enabled ? "AUDIO [ON]" : "AUDIO [MUTED]";
    sound.click(500, 0.03);
  });
}

// Input Helpers
if (dom.claimInput) {
  dom.claimInput.addEventListener("input", (e) => {
    dom.clearBtn.style.display = e.target.value.length ? "block" : "none";
  });
}
if (dom.clearBtn) {
  dom.clearBtn.addEventListener("click", () => {
    dom.claimInput.value = "";
    dom.clearBtn.style.display = "none";
    dom.claimInput.focus();
    sound.click(400);
  });
}

// Quick Sample Pills
dom.samplePills.forEach(pill => {
  pill.addEventListener("click", () => {
    dom.claimInput.value = pill.dataset.query;
    dom.clearBtn.style.display = "block";
    sound.click(700);
  });
});

// Segmented Pills Selection (Rounds & Depth)
dom.roundsOpts.forEach(opt => {
  opt.addEventListener("click", () => {
    dom.roundsOpts.forEach(o => o.classList.remove("active"));
    opt.classList.add("active");
    state.rounds = parseInt(opt.dataset.val, 10);
    sound.click(650);
  });
});

dom.depthOpts.forEach(opt => {
  opt.addEventListener("click", () => {
    dom.depthOpts.forEach(o => o.classList.remove("active"));
    opt.classList.add("active");
    state.depth = parseInt(opt.dataset.val, 10);
    sound.click(650);
  });
});

// Playwright Toggle
if (dom.playwrightToggle) {
  dom.playwrightToggle.addEventListener("click", () => {
    state.playwright = !state.playwright;
    dom.playwrightToggle.classList.toggle("active", state.playwright);
    const txt = dom.playwrightToggle.querySelector(".toggle-text");
    if (txt) txt.textContent = state.playwright ? "ENABLED" : "DISABLED";
    sound.click(state.playwright ? 800 : 400);
  });
}

// Pipeline Stage Updater
function appendLog(agent, msg, type = "normal") {
  if (dom.currentStage) dom.currentStage.textContent = agent.toUpperCase();
}

function clearLog() {
  if (dom.currentStage) dom.currentStage.textContent = "STANDBY";
}

// Raw Report Tab Switcher
let activeRawTab = "md";
if (dom.tabMd) {
  dom.tabMd.addEventListener("click", () => {
    activeRawTab = "md";
    dom.tabMd.classList.add("active");
    dom.tabJson.classList.remove("active");
    renderRawCode();
    sound.click(600);
  });
}
if (dom.tabJson) {
  dom.tabJson.addEventListener("click", () => {
    activeRawTab = "json";
    dom.tabJson.classList.add("active");
    dom.tabMd.classList.remove("active");
    renderRawCode();
    sound.click(600);
  });
}

function renderRawCode() {
  if (!state.currentReportData) return;
  if (activeRawTab === "md") {
    dom.rawDisplay.textContent = state.currentReportData.markdown || "# No Markdown Available";
  } else {
    dom.rawDisplay.textContent = JSON.stringify(state.currentReportData.result || {}, null, 2);
  }
}

if (dom.copyBtn) {
  dom.copyBtn.addEventListener("click", () => {
    const text = dom.rawDisplay.textContent;
    navigator.clipboard.writeText(text).then(() => {
      dom.copyBtn.textContent = "COPIED!";
      sound.click(900);
      setTimeout(() => dom.copyBtn.textContent = "COPY TO CLIPBOARD", 1800);
    });
  });
}



// Render Report Results to UI
function renderReportData(data) {
  state.currentReportData = data;
  const res = data.result || {};

  // 1. Verdict & Rationale
  const verdict = (res.verdict || "UNVERIFIABLE").toUpperCase();
  dom.verdictText.textContent = verdict;
  dom.verdictText.className = "verdict-title ndot " + verdict.toLowerCase();
  dom.verdictTag.textContent = `STATUS: ${verdict}`;
  dom.verdictRationale.textContent = res.rationale || "Analysis complete.";

  // 2. Confidence Dial
  const confRaw = (res.confidence !== undefined && res.confidence !== null)
    ? res.confidence
    : ((res.scores && res.scores.confidence) || 0);
  const conf = Math.round(confRaw * 100);
  dom.confVal.textContent = conf + "%";
  const circleOffset = 264 - (264 * Math.min(100, Math.max(0, conf))) / 100;
  dom.confCircle.style.strokeDashoffset = circleOffset;

  // 3. Stats (check root or res.scores)
  const scores = res.scores || {};
  const pool = res.timeline || res.evidence_pool || res.matched_items || [];
  const pCount = res.primary_sources ?? scores.primary_sources ?? 0;
  let sCount = res.secondary_sources ?? scores.secondary_sources ?? 0;
  if (sCount === 0 && pool.length > pCount) {
    sCount = pool.length - pCount;
  }
  dom.statPrimary.textContent = pCount;
  dom.statSecondary.textContent = sCount;
  dom.statContra.textContent = res.contradictions ?? scores.contradictions ?? 0;

  // Sub-claims Cards (supports res.subclaims or res.fact_check.subclaims)
  const subclaims = res.subclaims || (res.fact_check && res.fact_check.subclaims) || [];
  renderSubclaims(subclaims);

  // 6. Sources Metrics (Number of sources only, no origins/domains displayed)
  renderSourcesMetrics();

  // 7. Raw Code
  renderRawCode();
}

function renderSubclaims(subs) {
  dom.subclaimsContainer.innerHTML = "";
  dom.subclaimsCount.textContent = `${subs.length} DECOMPOSED CLAIMS`;

  if (!subs.length) {
    dom.subclaimsContainer.innerHTML = `<div class="empty-state">No sub-claim components found in report.</div>`;
    return;
  }

  subs.forEach((s, idx) => {
    const card = document.createElement("div");
    card.className = "subclaim-card";
    const status = (s.verdict || "unverifiable").toLowerCase();
    const supCount = (s.supporting || []).length;
    const disCount = (s.dissenting || []).length;

    card.innerHTML = `
      <div class="subclaim-top">
        <span class="subclaim-id">S${idx + 1} // ${escapeHtml(s.id || 'SUB')}</span>
        <span class="subclaim-status-pill ${status}">${escapeHtml(s.verdict || 'unverifiable')}</span>
      </div>
      <div class="subclaim-text">${escapeHtml(s.text || s.claim || '')}</div>
      <div class="subclaim-reasoning">${escapeHtml(s.reasoning || '')}</div>
      <div class="subclaim-evidence-row">
        ${supCount ? `<span class="ev-tag">SUPPORTING SOURCES: ${supCount}</span>` : ''}
        ${disCount ? `<span class="ev-tag text-red">DISSENTING SOURCES: ${disCount}</span>` : ''}
      </div>
    `;
    dom.subclaimsContainer.appendChild(card);
  });
}

function renderSourcesMetrics() {
  if (!state.currentReportData) return;
  const res = state.currentReportData.result || {};
  const scores = res.scores || {};
  const pool = res.timeline || res.evidence_pool || res.matched_items || [];

  const primaryCount = res.primary_sources ?? scores.primary_sources ?? 0;
  let secondaryCount = (res.secondary_sources || scores.secondary_sources) ? (res.secondary_sources || scores.secondary_sources) : (pool.length > primaryCount ? pool.length - primaryCount : 0);
  const contraCount = res.contradictions ?? scores.contradictions ?? 0;
  
  let totalCount = pool.length;
  if (totalCount < primaryCount + secondaryCount) {
    totalCount = primaryCount + secondaryCount;
  }
  if (totalCount === 0 && (primaryCount > 0 || secondaryCount > 0)) {
    totalCount = primaryCount + secondaryCount;
  }

  if (dom.heroSourcesCount) dom.heroSourcesCount.textContent = totalCount;
  if (dom.sourcesTotalTag) dom.sourcesTotalTag.textContent = `${totalCount} TOTAL SOURCES`;
  if (dom.sstatTotal) dom.sstatTotal.textContent = totalCount;
  if (dom.sstatPrimary) dom.sstatPrimary.textContent = primaryCount;
  if (dom.sstatSecondary) dom.sstatSecondary.textContent = secondaryCount;
  if (dom.sstatContra) dom.sstatContra.textContent = contraCount;
  if (dom.sourcesRatioText) dom.sourcesRatioText.textContent = `${primaryCount} PRIMARY · ${secondaryCount} SECONDARY`;

  const denom = Math.max(1, totalCount + contraCount);
  const pPct = Math.round((primaryCount / denom) * 100);
  const sPct = Math.round((secondaryCount / denom) * 100);
  const cPct = Math.round((contraCount / denom) * 100);

  if (dom.barSlicePrimary) dom.barSlicePrimary.style.width = pPct + "%";
  if (dom.barSliceSecondary) dom.barSliceSecondary.style.width = sPct + "%";
  if (dom.barSliceContra) dom.barSliceContra.style.width = cPct + "%";
}

// API Calling & Polling Logic
async function triggerTrace() {
  const claim = (dom.claimInput.value || "").trim();
  if (claim.length < 3) {
    alert("Please enter a valid claim or narrative to trace (at least 3 characters).");
    dom.claimInput.focus();
    return;
  }

  // Set running state
  dom.startBtn.classList.add("loading");
  dom.startBtn.disabled = true;
  dom.globalLed.className = "live-led busy";
  dom.globalStatus.textContent = "TRACING // LIVE";
  if (dom.currentRound) dom.currentRound.textContent = `1 / ${state.rounds}`;
  
  clearLog();
  glyph.startSequence();
  sound.click(800, 0.05);

  appendLog("SYSTEM", `Initializing pipeline for query: "${claim}" (Rounds: ${state.rounds}, Depth: ${state.depth})`);

  try {
    const res = await fetch("/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        claim: claim,
        rounds: state.rounds,
        depth: state.depth,
        archive: true
      })
    });

    if (!res.ok) {
      let detailMsg = `HTTP ${res.status}`;
      try {
        const errJson = await res.json();
        if (errJson.detail) {
          detailMsg = Array.isArray(errJson.detail)
            ? errJson.detail.map(d => `${d.loc ? d.loc.slice(-1)[0] : ''}: ${d.msg}`).join(', ')
            : String(errJson.detail);
        }
      } catch (_) {}
      throw new Error(detailMsg);
    }

    const json = await res.json();
    state.activeRunId = json.id;
    appendLog("SYSTEM", `Run registered with ID: ${json.id}. Streaming live agent events...`);

    pollRunEvents(json.id);
  } catch (err) {
    appendLog("ERROR", `Run failed to initialize: ${err.message}. Loading local cache instead...`, "error");
    glyph.stop();
    dom.startBtn.classList.remove("loading");
    dom.startBtn.disabled = false;
    dom.globalLed.className = "live-led";
    dom.globalStatus.textContent = "STANDBY // CACHED";
    loadLocalReport();
  }
}

async function pollRunEvents(runId) {
  let lastEventId = 0;
  let checks = 0;

  state.pollTimer = setInterval(async () => {
    checks++;
    try {
      // 1. Fetch live events
      const evRes = await fetch(`/runs/${runId}/events?after=${lastEventId}`);
      if (evRes.ok) {
        const events = await evRes.json();
        events.forEach(e => {
          lastEventId = Math.max(lastEventId, e.id);
          appendLog(e.agent, e.message);
          glyph.pulse();
        });
      }

      // 2. Check run completion
      const runRes = await fetch(`/runs/${runId}`);
      if (runRes.ok) {
        const runData = await runRes.json();
        if (runData.status === "done") {
          clearInterval(state.pollTimer);
          appendLog("SYSTEM", "Pipeline finished successfully! Loading report...", "success");
          glyph.stop();
          dom.startBtn.classList.remove("loading");
          dom.startBtn.disabled = false;
          dom.globalLed.className = "live-led active";
          dom.globalStatus.textContent = "COMPLETED // REPORT READY";

          // Fetch full report markdown
          const repRes = await fetch(`/runs/${runId}/report`);
          const md = repRes.ok ? await repRes.text() : "";
          renderReportData({ result: runData.result, markdown: md });
          fetchRunsHistory();
        } else if (runData.status === "failed") {
          clearInterval(state.pollTimer);
          appendLog("ERROR", `Run failed: ${runData.error}`, "error");
          glyph.stop();
          dom.startBtn.classList.remove("loading");
          dom.startBtn.disabled = false;
          dom.globalLed.className = "live-led";
          dom.globalStatus.textContent = "FAILED";
          fetchRunsHistory();
        }
      }
    } catch (e) {
      if (checks > 60) {
        clearInterval(state.pollTimer);
        glyph.stop();
      }
    }
  }, 1200);
}

// Fetch and populate previous runs from database
async function fetchRunsHistory() {
  if (!dom.historySelect) return;
  try {
    const res = await fetch("/runs");
    if (res.ok) {
      const runs = await res.json();
      if (runs && runs.length > 0) {
        dom.historySelect.style.display = "inline-block";
        dom.historySelect.innerHTML = `<option value="">HISTORY (${runs.length})</option>`;
        runs.forEach(r => {
          const opt = document.createElement("option");
          opt.value = r.id;
          const statusIcon = r.status === "done" ? "✓" : r.status === "failed" ? "✗" : "⏳";
          const label = `${statusIcon} ${(r.claim || 'Run').slice(0, 24)}...`;
          opt.textContent = label;
          dom.historySelect.appendChild(opt);
        });
      }
    }
  } catch (e) {}
}

async function loadRunById(runId) {
  if (!runId) return;
  try {
    appendLog("SYSTEM", `Loading run ${runId.slice(0, 8)}...`);
    const runRes = await fetch(`/runs/${runId}`);
    if (!runRes.ok) throw new Error(`HTTP ${runRes.status}`);
    const runData = await runRes.json();
    
    const repRes = await fetch(`/runs/${runId}/report`);
    const md = repRes.ok ? await repRes.text() : "";
    
    renderReportData({ result: runData.result, markdown: md });
    if (dom.claimInput && runData.claim) {
      dom.claimInput.value = runData.claim;
      dom.clearBtn.style.display = "block";
    }
    appendLog("SYSTEM", `Run loaded: "${(runData.claim || '').slice(0, 50)}..."`, "success");
    sound.click(750);
  } catch (err) {
    appendLog("ERROR", `Failed to load run ${runId}: ${err.message}`, "error");
  }
}

// Load cached local report from server
async function loadLocalReport() {
  try {
    const res = await fetch("/api/local-report");
    if (res.ok) {
      const data = await res.json();
      if (data.result) {
        appendLog("CACHE", "Loaded existing verification report from local disk.", "success");
        renderReportData(data);
        dom.globalLed.className = "live-led active";
        dom.globalStatus.textContent = "STANDBY // CACHE LOADED";
        if (dom.claimInput && data.result.claim) {
          dom.claimInput.value = data.result.claim;
          dom.clearBtn.style.display = "block";
        }
        return;
      }
    }
  } catch (e) {}

  // Fallback demo data if no report yet
  appendLog("SYSTEM", "Awaiting initial claim trace to generate live data.");
}

// Initial Backend Health Ping
async function checkBackend() {
  try {
    const res = await fetch("/health");
    if (res.ok) {
      const h = await res.json();
      if (dom.backendPing) {
        dom.backendPing.textContent = `BACKEND: ONLINE (GROQ + GEMINI)`;
        dom.backendPing.style.color = "#00ff66";
      }
      loadLocalReport();
      fetchRunsHistory();
      return;
    }
  } catch (e) {}

  if (dom.backendPing) {
    dom.backendPing.textContent = `BACKEND: OFFLINE (START UVICORN)`;
    dom.backendPing.style.color = "var(--red-accent)";
  }
  loadLocalReport();
}

// HTML Escape helper
function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// Event Listeners
if (dom.startBtn) {
  dom.startBtn.addEventListener("click", triggerTrace);
}
if (dom.claimInput) {
  dom.claimInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") triggerTrace();
  });
}
if (dom.loadSampleBtn) {
  dom.loadSampleBtn.addEventListener("click", () => {
    loadLocalReport();
    sound.click(750);
  });
}
if (dom.historySelect) {
  dom.historySelect.addEventListener("change", (e) => {
    if (e.target.value) {
      loadRunById(e.target.value);
    }
  });
}

// Start
checkBackend();
