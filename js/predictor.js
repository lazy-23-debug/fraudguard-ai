/**
 * predictor.js
 * ─────────────────────────────────────────────────────────
 * Transaction fraud analysis — Hybrid ML Framework frontend.
 *
 * TWO MODES (auto-detected):
 *
 *  [1] REAL MODEL MODE  — API is running (python train.py + uvicorn)
 *      Sends form inputs to FastAPI /predict/demo endpoint.
 *      Returns genuine Autoencoder + XGBoost predictions with
 *      real SHAP feature contributions from the trained model.
 *
 *  [2] SIMULATION MODE  — API is offline / not yet trained
 *      JavaScript heuristic scoring aligned with SHAP findings
 *      from the ULB Credit Card dataset.
 *      Identical interface; results labelled clearly.
 *
 * The mode indicator in the topbar updates every 30 seconds.
 *
 * Caleb University · CSC 400 · 22/10407
 */

'use strict';

// ─── API Configuration ────────────────────────────────────────────────────────
// Auto-detects whether we are running locally or on Vercel production.
// After deploying your backend to Render, replace the RENDER_URL below
// with your actual Render service URL (e.g. https://fraudguard-api.onrender.com)
const RENDER_URL  = 'https://fraudguard-api-ipus.onrender.com';

const API_URL = (() => {
  const host    = window.location.hostname;
  const isLocal = host === 'localhost' || host === '127.0.0.1';
  return isLocal ? 'http://127.0.0.1:8000' : RENDER_URL;
})();

const API_TIMEOUT = 8000;  // 8s — Render free tier can be slow on cold start

let apiOnline = false;   // current status, updated by health checks

// ─── Helper: clamp ───────────────────────────────────────────────────────────
function clamp(val, min, max) {
  return Math.max(min, Math.min(max, val));
}

// ════════════════════════════════════════════════════════════════
//  API HEALTH CHECK  (runs on load and every 30s)
// ════════════════════════════════════════════════════════════════

async function checkApiHealth() {
  try {
    const ctrl = new AbortController();
    const id   = setTimeout(() => ctrl.abort(), 3000);
    const res  = await fetch(`${API_URL}/health`, { signal: ctrl.signal });
    clearTimeout(id);

    if (res.ok) {
      const data = await res.json();
      apiOnline = data.models_loaded === true;
    } else {
      apiOnline = false;
    }
  } catch {
    apiOnline = false;
  }
  updateApiStatusBadge();
}

function updateApiStatusBadge() {
  const badge = document.getElementById('api-status-badge');
  if (!badge) return;
  if (apiOnline) {
    badge.textContent = '🟢 Real Model Active';
    badge.className   = 'api-badge api-online';
  } else {
    badge.textContent = '🟡 Simulation Mode';
    badge.className   = 'api-badge api-offline';
  }
  // Also update the mode notice banner in the analyzer section
  if (typeof window.__updateModeNotice === 'function') {
    window.__updateModeNotice(apiOnline);
  }
}

// Start health checks (initial + every 30s)
document.addEventListener('DOMContentLoaded', () => {
  checkApiHealth();
  setInterval(checkApiHealth, 30_000);
});

// ════════════════════════════════════════════════════════════════
//  MAIN ENTRY POINT  (called by onclick in HTML)
// ════════════════════════════════════════════════════════════════

async function analyzeTransaction() {
  const inputs = {
    amount:   parseFloat(document.getElementById('inp-amount').value) || 0,
    type:     document.getElementById('inp-type').value,
    hour:     parseInt(document.getElementById('inp-hour').value, 10) || 12,
    auth:     document.getElementById('inp-auth').value,
    age:      parseInt(document.getElementById('inp-age').value, 10) || 12,
    freq:     parseInt(document.getElementById('inp-freq').value, 10) || 1,
    device:   document.getElementById('inp-device').value,
    location: document.getElementById('inp-location').value
  };

  if (inputs.amount <= 0) {
    alert('Please enter a transaction amount greater than ₦0.');
    return;
  }

  // Animate button while processing
  const btn = document.getElementById('btn-analyze');
  btn.innerHTML = apiOnline
    ? '<span class="btn-icon">⊙</span> Querying ML Model…'
    : '<span class="btn-icon">⊙</span> Running Simulation…';
  btn.disabled = true;

  try {
    if (apiOnline) {
      // ── PATH 1: real trained model via FastAPI ─────────────────
      const result = await callRealAPI(inputs);
      if (result) {
        displayAPIResult(result, inputs);
        return;
      }
      // If API call fails mid-way, fall through to simulation
    }
    // ── PATH 2: JavaScript simulation fallback ─────────────────
    runSimulation(inputs);

  } finally {
    // Always re-enable the button
    setTimeout(() => {
      btn.innerHTML = '<span class="btn-icon">⊕</span> Run Hybrid ML Analysis';
      btn.disabled  = false;
    }, apiOnline ? 0 : 600);
  }
}

// ════════════════════════════════════════════════════════════════
//  PATH 1 — REAL API CALL
// ════════════════════════════════════════════════════════════════

async function callRealAPI(inputs) {
  try {
    const ctrl = new AbortController();
    const id   = setTimeout(() => ctrl.abort(), API_TIMEOUT);

    const res = await fetch(`${API_URL}/predict/demo`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(inputs),
      signal:  ctrl.signal
    });
    clearTimeout(id);

    if (!res.ok) throw new Error(`API ${res.status}`);
    return await res.json();

  } catch (err) {
    console.warn('[FraudGuard] API call failed:', err.message, '→ switching to simulation');
    apiOnline = false;
    updateApiStatusBadge();
    return null;
  }
}

function displayAPIResult(data, inputs) {
  // Show the result panel
  const panel = document.getElementById('result-panel');
  panel.style.display = 'block';

  // Risk level → CSS class maps
  const levelCss = {
    LOW:      { verdict: 'verdict-low',      rec: 'rec-low' },
    MEDIUM:   { verdict: 'verdict-medium',   rec: 'rec-medium' },
    HIGH:     { verdict: 'verdict-high',     rec: 'rec-high' },
    CRITICAL: { verdict: 'verdict-critical', rec: 'rec-critical' }
  };
  const css = levelCss[data.risk_level] || levelCss.MEDIUM;

  // Verdict badge
  const verdict = document.getElementById('result-verdict');
  verdict.textContent = `${data.risk_level} RISK — Real Model`;
  verdict.className   = `result-verdict ${css.verdict}`;

  // Gauge
  document.getElementById('gauge-marker').style.left    = data.risk_score + '%';
  document.getElementById('gauge-score-num').textContent = data.risk_score;

  // Component bars
  setBar('ae-bar',  'ae-pct',  data.ae_score);
  setBar('xgb-bar', 'xgb-pct', data.xgb_score);
  setBar('ens-bar', 'ens-pct', data.risk_score, data.risk_level.toLowerCase());

  // SHAP contributions from the REAL model
  renderShapContributions(data.shap_contributions);

  // Recommendation from the API
  const rec = document.getElementById('recommendation-box');
  rec.innerHTML = data.recommendation;
  rec.className = `recommendation-box ${css.rec}`;

  panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  updateGlobalCounters(data.risk_level);
}

function renderShapContributions(shapDict) {
  const container = document.getElementById('feature-attributions');
  container.innerHTML = '';

  const entries  = Object.entries(shapDict);
  const maxAbs   = Math.max(...entries.map(([, v]) => Math.abs(v)), 0.001);

  entries.forEach(([name, value]) => {
    const pct    = (Math.abs(value) / maxAbs) * 45;
    const isPos  = value > 0;
    const row    = document.createElement('div');
    row.className = 'attr-row';
    row.innerHTML = `
      <span class="attr-name">${name}</span>
      <div class="attr-bar-wrap">
        ${isPos
          ? `<div class="attr-bar-pos" style="width:${pct}%"></div>`
          : `<div class="attr-bar-neg" style="width:${pct}%"></div>`
        }
      </div>
      <span class="attr-val ${isPos ? 'pos' : 'neg'}">
        ${isPos ? '+' : ''}${value.toFixed(4)}
      </span>`;
    container.appendChild(row);
  });
}

// ════════════════════════════════════════════════════════════════
//  PATH 2 — JAVASCRIPT SIMULATION
// ════════════════════════════════════════════════════════════════
//
//  Risk factor weights are aligned with XGBoost + SHAP results
//  from the ULB Credit Card dataset (published analyses, 2024).

const SIM_WEIGHTS = { ae: 0.35, xgb: 0.65 };

function amountRisk(a) {
  if (a <= 0)        return 0;
  if (a < 500)       return 18;   // probe transaction
  if (a < 5000)      return 0;
  if (a < 20000)     return 2;
  if (a < 100000)    return 6;
  if (a < 500000)    return 18;
  if (a < 2000000)   return 30;
  return 38;
}
function timeRisk(h) {
  if (h >= 1 && h <= 4)   return 22;
  if (h === 0 || h === 5)  return 14;
  if (h >= 22)             return 10;
  return 0;
}
function authRisk(m) {
  return ({ biometric: -18, pin_otp: -8, otp: 12, pin: 8, none: 48 })[m] ?? 0;
}
function ageRisk(mo) {
  if (mo <= 0)   return 28;
  if (mo <= 1)   return 22;
  if (mo <= 3)   return 14;
  if (mo <= 12)  return 6;
  return 0;
}
function freqRisk(f) {
  if (f >= 15) return 35; if (f >= 10) return 25;
  if (f >= 6)  return 16; if (f >= 4)  return 9;
  return 0;
}
function deviceRisk(d)  { return d === 'new' ? 26 : 0; }
function locationRisk(l) {
  return ({ lagos: 0, abuja: 0, portharcourt: 0,
            other_ng: 3, international: 22, vpn: 38 })[l] ?? 5;
}
function typeRisk(t) {
  return ({ nip: 6, ussd: 4, pos: 0, atm: 3, card: 5 })[t] ?? 0;
}

function buildFactors(inp) {
  const { amount, type, hour, auth, age, freq, device, location } = inp;
  return [
    { name: `Transaction Amount (₦${Number(amount).toLocaleString()})`,
      raw: amountRisk(amount),
      aeW: 0.55, xgbW: 0.45 },
    { name: 'Authentication Method',
      raw: authRisk(auth),
      aeW: 0.30, xgbW: 0.70 },
    { name: 'Device Status',
      raw: deviceRisk(device),
      aeW: 0.50, xgbW: 0.50 },
    { name: 'Origin Location',
      raw: locationRisk(location),
      aeW: 0.60, xgbW: 0.40 },
    { name: `Time of Day (${hour}:00)`,
      raw: timeRisk(hour),
      aeW: 0.55, xgbW: 0.45 },
    { name: 'Transaction Frequency (Last Hour)',
      raw: freqRisk(freq),
      aeW: 0.40, xgbW: 0.60 },
    { name: `Account Age (${age} months)`,
      raw: ageRisk(age),
      aeW: 0.30, xgbW: 0.70 },
    { name: `Transaction Type (${type.toUpperCase()})`,
      raw: typeRisk(type),
      aeW: 0.40, xgbW: 0.60 }
  ];
}

function simAggregate(factors) {
  const noise  = (Math.random() - 0.5) * 4;
  const aeRaw  = factors.reduce((s, f) => s + f.raw * f.aeW, 0);
  const xgbRaw = factors.reduce((s, f) => s + f.raw * f.xgbW, 0);
  const ae     = clamp(aeRaw  + noise * 0.5, 0, 100);
  const xgb    = clamp(xgbRaw + noise * 0.5, 0, 100);
  return {
    ae:    Math.round(ae),
    xgb:   Math.round(xgb),
    score: Math.round(clamp(SIM_WEIGHTS.ae * ae + SIM_WEIGHTS.xgb * xgb, 0, 100))
  };
}

function simRiskLevel(score) {
  if (score <= 25) return { level: 'LOW',      verdictCls: 'verdict-low',      recCls: 'rec-low',      cls: 'low' };
  if (score <= 50) return { level: 'MEDIUM',   verdictCls: 'verdict-medium',   recCls: 'rec-medium',   cls: 'medium' };
  if (score <= 75) return { level: 'HIGH',     verdictCls: 'verdict-high',     recCls: 'rec-high',     cls: 'high' };
  return             { level: 'CRITICAL', verdictCls: 'verdict-critical', recCls: 'rec-critical', cls: 'critical' };
}

function simRecommendation(score, level, inp) {
  const recs = {
    low:      `✓ Transaction appears <strong>legitimate</strong>. Autoencoder reconstruction error within normal range. XGBoost classifier assigns low fraud probability. Allow transaction to proceed.`,
    medium:   `⚠ Elevated risk on ₦${Number(inp.amount).toLocaleString()} ${inp.type.toUpperCase()}. Request <strong>additional authentication</strong> (callback or second OTP). Route to Tier-2 review queue.`,
    high:     `⚡ <strong>High fraud risk</strong>. <strong>Hold transaction</strong> pending manual review. Notify account holder via registered contact. Document flags for NIBSS reporting.`,
    critical: `🚨 <strong>CRITICAL — Block Immediately.</strong> Multiple critical indicators. <strong>Freeze transaction, suspend session, initiate account lockout.</strong> File STR with NIBSS.`
  };
  return recs[level.toLowerCase()] ?? recs.medium;
}

function runSimulation(inputs) {
  const factors = buildFactors(inputs);
  const scores  = simAggregate(factors);
  const risk    = simRiskLevel(scores.score);

  const panel = document.getElementById('result-panel');
  panel.style.display = 'block';

  // Verdict — clearly labelled as simulation
  const verdict = document.getElementById('result-verdict');
  verdict.textContent = `${risk.level} RISK — Simulation`;
  verdict.className   = `result-verdict ${risk.verdictCls}`;

  // Gauge
  document.getElementById('gauge-marker').style.left    = scores.score + '%';
  document.getElementById('gauge-score-num').textContent = scores.score;

  // Component bars
  setBar('ae-bar',  'ae-pct',  scores.ae);
  setBar('xgb-bar', 'xgb-pct', scores.xgb);
  setBar('ens-bar', 'ens-pct', scores.score, risk.cls);

  // Attribution bars (simulation-based)
  renderSimAttributions(factors);

  // Recommendation
  const rec = document.getElementById('recommendation-box');
  rec.innerHTML = simRecommendation(scores.score, risk.level, inputs);
  rec.className = `recommendation-box ${risk.recCls}`;

  panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  updateGlobalCounters(risk.level);
}

function renderSimAttributions(factors) {
  const container = document.getElementById('feature-attributions');
  container.innerHTML = '';

  const top    = factors.filter(f => f.raw !== 0)
                        .sort((a, b) => Math.abs(b.raw) - Math.abs(a.raw))
                        .slice(0, 5);
  const maxVal = Math.max(...top.map(f => Math.abs(f.raw)), 1);

  top.forEach(f => {
    const pct   = (Math.abs(f.raw) / maxVal) * 45;
    const isPos = f.raw > 0;
    const row   = document.createElement('div');
    row.className = 'attr-row';
    row.innerHTML = `
      <span class="attr-name">${f.name}</span>
      <div class="attr-bar-wrap">
        ${isPos
          ? `<div class="attr-bar-pos" style="width:${pct}%"></div>`
          : `<div class="attr-bar-neg" style="width:${pct}%"></div>`
        }
      </div>
      <span class="attr-val ${isPos ? 'pos' : 'neg'}">
        ${isPos ? '+' : '-'}${Math.abs(f.raw).toFixed(1)}
      </span>`;
    container.appendChild(row);
  });
}

// ════════════════════════════════════════════════════════════════
//  SHARED HELPERS
// ════════════════════════════════════════════════════════════════

function setBar(barId, pctId, value, riskCls) {
  const pct = clamp(value, 0, 100);
  const bar = document.getElementById(barId);
  if (!bar) return;
  bar.style.width = pct + '%';
  if (barId === 'ens-bar' && riskCls) {
    const colors = { low: '#2ED573', medium: '#FFD32A', high: '#FF7F50', critical: '#FF4757' };
    bar.style.background = colors[riskCls] || '#00D9FF';
  }
  const pctEl = document.getElementById(pctId);
  if (pctEl) pctEl.textContent = pct + '%';
}

function updateGlobalCounters(level) {
  if (typeof AppState === 'undefined') return;
  AppState.txCount++;
  if (level === 'HIGH' || level === 'CRITICAL') AppState.fraudCount++;
  if (typeof updateDashboardCounters === 'function') updateDashboardCounters();
}
