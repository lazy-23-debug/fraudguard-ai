/**
 * transactions.js
 * ─────────────────────────────────────────────────────────
 * Live Transaction Monitor — stream generator.
 *
 * Each transaction is now a COMPLETE record (amount, type,
 * hour, auth, age, freq, device, location) matching exactly
 * the fields the FastAPI /predict/demo endpoint expects.
 *
 * SCORING FLOW:
 *  ┌─ API online ──► POST /predict/demo → real AE + XGBoost scores
 *  │                  Row shown as "Scoring…" then updated live
 *  └─ API offline ──► JavaScript heuristic scoring (immediate)
 *
 * Caleb University · CSC 400 · 22/10407
 */

'use strict';

// ─── Nigerian payment context data ───────────────────────────────────────────

const NG_BANKS = [
  'GTBank', 'Access', 'Zenith', 'UBA', 'First Bank',
  'Polaris', 'Fidelity', 'FCMB', 'Sterling', 'Wema',
  'Stanbic', 'Keystone', 'Jaiz', 'OPay', 'Kuda',
  'PalmPay', 'Moniepoint', 'VBank'
];

const TX_TYPES = ['NIP', 'USSD', 'POS', 'ATM', 'Card'];

// API type values (lowercase, matches FastAPI schema)
const TX_TYPE_API = {
  NIP: 'nip', USSD: 'ussd', POS: 'pos', ATM: 'atm', Card: 'card'
};

const TX_TYPE_COLORS = {
  NIP: 'text-cyan', USSD: 'text-yellow',
  POS: 'text-green', ATM: 'text-orange', Card: 'text-secondary'
};

// Auth methods: weighted toward common secure methods
const AUTH_OPTIONS = [
  { api: 'biometric', label: 'Biometric',   weight: 25 },
  { api: 'pin_otp',   label: 'PIN + OTP',   weight: 35 },
  { api: 'otp',       label: 'OTP Only',    weight: 20 },
  { api: 'pin',       label: 'PIN Only',    weight: 15 },
  { api: 'none',      label: 'None',        weight: 5  }
];

// Location: weighted toward Lagos (63% per NIBSS)
const LOCATION_OPTIONS = [
  { api: 'lagos',         label: 'Lagos',         weight: 45 },
  { api: 'abuja',         label: 'Abuja',         weight: 18 },
  { api: 'portharcourt',  label: 'Port Harcourt', weight: 12 },
  { api: 'other_ng',      label: 'Other NG',      weight: 15 },
  { api: 'international', label: 'International', weight: 7  },
  { api: 'vpn',           label: 'VPN',           weight: 3  }
];

// Monitor state
let monitorInterval = null;
let monitorRunning  = false;
let monitorTotal    = 0;
let monitorFraud    = 0;
let monitorLegit    = 0;
const MAX_ROWS      = 80;
// Auto-matches whatever URL predictor.js resolved to
const API_URL_MON = (() => {
  const host    = window.location.hostname;
  const isLocal = host === 'localhost' || host === '127.0.0.1';
  return isLocal
    ? 'http://127.0.0.1:8000'
    : 'https://REPLACE-WITH-YOUR-RENDER-URL.onrender.com';
})();

// ─── Weighted random pick ─────────────────────────────────────────────────────
function weightedPick(options) {
  const total  = options.reduce((s, o) => s + o.weight, 0);
  let   rand   = Math.random() * total;
  for (const opt of options) {
    rand -= opt.weight;
    if (rand <= 0) return opt;
  }
  return options[options.length - 1];
}

function pick(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

// ─── Generate random Nigerian account number ──────────────────────────────────
function randomAcct() {
  return Array.from({ length: 10 }, () => Math.floor(Math.random() * 10)).join('');
}

// ─── Realistic NGN amount distribution ───────────────────────────────────────
function randomAmount() {
  const r = Math.random();
  if (r < 0.35) return Math.round((Math.random() * 4500  + 500)   / 100)   * 100;
  if (r < 0.60) return Math.round((Math.random() * 45000 + 5000)  / 500)   * 500;
  if (r < 0.80) return Math.round((Math.random() * 150000+50000)  / 1000)  * 1000;
  if (r < 0.95) return Math.round((Math.random() * 800000+200000) / 5000)  * 5000;
  return          Math.round((Math.random() * 4000000+1000000)     / 10000) * 10000;
}

// ─── Build a full transaction (all fields the API needs) ─────────────────────
function generateTransaction() {
  const now      = new Date();
  const hour     = now.getHours();
  const type     = pick(TX_TYPES);
  const authOpt  = weightedPick(AUTH_OPTIONS);
  const locOpt   = weightedPick(LOCATION_OPTIONS);
  const amount   = randomAmount();

  // Account age: most accounts are established; small % are brand new
  const age      = Math.random() < 0.08
    ? Math.floor(Math.random() * 3)          // 0–2 months (new account)
    : Math.floor(Math.random() * 58) + 3;    // 3–60 months (established)

  // Frequency: most 1–3 per hour; occasionally suspicious burst
  const freq     = Math.random() < 0.05
    ? Math.floor(Math.random() * 12) + 8     // 8–20 (burst)
    : Math.floor(Math.random() * 4);         // 0–3 (normal)

  // Device: mostly known, ~12% new/unrecognized
  const device   = Math.random() < 0.12 ? 'new' : 'known';

  return {
    // Display fields
    id:        'TXN' + now.getTime().toString().slice(-8),
    time:      now.toLocaleTimeString('en-GB', { hour12: false }),
    account:   pick(NG_BANKS) + ' ···' + randomAcct().slice(-4),
    typeLabel: type,

    // API payload fields (match FastAPI FormInput schema exactly)
    amount,
    type:     TX_TYPE_API[type],
    hour,
    auth:     authOpt.api,
    age,
    freq,
    device,
    location: locOpt.api
  };
}

// ─── Risk level helpers ───────────────────────────────────────────────────────
function riskClass(score) {
  if (score >= 76) return 'risk-critical';
  if (score >= 51) return 'risk-high';
  if (score >= 26) return 'risk-medium';
  return 'risk-low';
}

function statusFromScore(score) {
  if (score >= 76) return { label: 'FRAUD',  cls: 'pill-fraud',    isFraud: true  };
  if (score >= 51) return { label: 'HIGH',   cls: 'pill-flagged',  isFraud: false };
  if (score >= 26) return { label: 'REVIEW', cls: 'pill-review',   isFraud: false };
  return                   { label: 'LEGIT', cls: 'pill-legit',    isFraud: false };
}

// ─── JavaScript fallback scoring (used when API is offline) ──────────────────
function simulateScores(tx) {
  // Probability-based isFraud flag
  let prob = 0.002;
  if (tx.amount > 500000)   prob += 0.04;
  if (tx.amount > 1500000)  prob += 0.06;
  if (tx.type === 'nip' && (tx.hour <= 4 || tx.hour >= 23)) prob += 0.05;
  if (tx.type === 'card' && tx.amount > 200000) prob += 0.03;
  if (tx.type === 'ussd')   prob += 0.008;
  if (tx.amount < 600)      prob += 0.03;
  if (tx.auth === 'none')   prob += 0.12;
  if (tx.device === 'new')  prob += 0.04;
  if (tx.location === 'vpn') prob += 0.08;
  if (tx.location === 'international') prob += 0.04;
  if (tx.freq >= 8)         prob += 0.05;
  if (tx.age <= 1)          prob += 0.05;
  prob = Math.min(prob, 0.40);

  const isFraud  = Math.random() < prob;
  const aeScore  = isFraud
    ? Math.round(50 + Math.random() * 45)
    : Math.round(Math.random() * 20);
  const xgbScore = isFraud
    ? Math.round(55 + Math.random() * 42)
    : Math.round(Math.random() * 18);
  const risk     = Math.round(0.35 * aeScore + 0.65 * xgbScore);

  return { aeScore, xgbScore, riskScore: risk, isFraud };
}

// ─── Call real API for a single transaction ───────────────────────────────────
async function scoreWithAPI(tx) {
  try {
    const ctrl = new AbortController();
    const tid  = setTimeout(() => ctrl.abort(), 5000);

    const payload = {
      amount:   tx.amount,
      type:     tx.type,
      hour:     tx.hour,
      auth:     tx.auth,
      age:      tx.age,
      freq:     tx.freq,
      device:   tx.device,
      location: tx.location
    };

    const res = await fetch(`${API_URL_MON}/predict/demo`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
      signal:  ctrl.signal
    });
    clearTimeout(tid);

    if (!res.ok) throw new Error(`API ${res.status}`);
    const data = await res.json();

    return {
      aeScore:   Math.round(data.ae_score),
      xgbScore:  Math.round(data.xgb_score),
      riskScore: data.risk_score,
      isFraud:   data.risk_score >= 76,
      fromModel: true
    };
  } catch {
    return null;   // fall back to simulation
  }
}

// ─── Add a row to the table ───────────────────────────────────────────────────
function buildRow(tx, scores) {
  const status = statusFromScore(scores.riskScore);
  const source = scores.fromModel ? '★' : '';   // ★ = real model scored this row

  const tr = document.createElement('tr');
  tr.id = 'row-' + tx.id;
  if (scores.isFraud || scores.riskScore >= 51) tr.classList.add('fraud-row');

  tr.innerHTML = `
    <td class="text-secondary">${tx.id}</td>
    <td>${tx.time}</td>
    <td class="text-primary"><strong>₦${tx.amount.toLocaleString()}</strong></td>
    <td class="${TX_TYPE_COLORS[tx.typeLabel] || ''}">${tx.typeLabel}</td>
    <td>${tx.account}</td>
    <td class="${riskClass(scores.aeScore)}">${scores.aeScore}${source}</td>
    <td class="${riskClass(scores.xgbScore)}">${scores.xgbScore}${source}</td>
    <td class="${riskClass(scores.riskScore)} risk-score-cell">${scores.riskScore}</td>
    <td><span class="status-pill ${status.cls}">${status.label}</span></td>
  `;
  return { tr, status };
}

// ─── Placeholder row shown while API is scoring ───────────────────────────────
function buildPendingRow(tx) {
  const tr = document.createElement('tr');
  tr.id = 'row-' + tx.id;
  tr.innerHTML = `
    <td class="text-secondary">${tx.id}</td>
    <td>${tx.time}</td>
    <td class="text-primary"><strong>₦${tx.amount.toLocaleString()}</strong></td>
    <td class="${TX_TYPE_COLORS[tx.typeLabel] || ''}">${tx.typeLabel}</td>
    <td>${tx.account}</td>
    <td class="text-muted">…</td>
    <td class="text-muted">…</td>
    <td class="text-muted">…</td>
    <td><span class="status-pill" style="background:rgba(255,255,255,0.06);color:#3D4466">SCORING</span></td>
  `;
  return tr;
}

// ─── Update an existing row with real scores ──────────────────────────────────
function updateRow(tx, scores) {
  const tr     = document.getElementById('row-' + tx.id);
  if (!tr) return;

  const status = statusFromScore(scores.riskScore);
  const cells  = tr.querySelectorAll('td');

  if (scores.isFraud || scores.riskScore >= 51) tr.classList.add('fraud-row');

  cells[5].className   = riskClass(scores.aeScore);
  cells[5].textContent = scores.aeScore + '★';   // ★ = scored by real model

  cells[6].className   = riskClass(scores.xgbScore);
  cells[6].textContent = scores.xgbScore + '★';

  cells[7].className   = riskClass(scores.riskScore) + ' risk-score-cell';
  cells[7].textContent = scores.riskScore;

  cells[8].innerHTML = `<span class="status-pill ${status.cls}">${status.label}</span>`;

  // Brief highlight to show the row was just updated
  tr.style.outline = '1px solid var(--cyan)';
  setTimeout(() => { tr.style.outline = ''; }, 800);

  // Update counters for this newly-scored row
  if (status.isFraud || scores.riskScore >= 76) {
    monitorFraud++;
    if (typeof pushAlert === 'function') pushAlert({ ...tx, riskScore: scores.riskScore });
    if (typeof AppState !== 'undefined') AppState.fraudCount++;
  } else {
    monitorLegit++;
  }
  updateMonitorCounters();
  if (typeof updateDashboardCounters === 'function') updateDashboardCounters();
}

// ─── Main: process one generated transaction ──────────────────────────────────
async function processTransaction() {
  const tx     = generateTransaction();
  const tbody  = document.getElementById('tx-tbody');
  if (!tbody) return;

  // Trim table to MAX_ROWS
  while (tbody.rows.length >= MAX_ROWS) {
    tbody.deleteRow(tbody.rows.length - 1);
  }

  // Check if the real API is available (use predictor.js flag)
  const useAPI = (typeof apiOnline !== 'undefined') && apiOnline;

  monitorTotal++;

  if (useAPI) {
    // Show a "Scoring…" placeholder row immediately
    const pendingTr = buildPendingRow(tx);
    tbody.insertBefore(pendingTr, tbody.firstChild);
    updateMonitorCounters();

    // Call the real model asynchronously
    const apiResult = await scoreWithAPI(tx);

    if (apiResult) {
      // Real model responded — update the row with genuine scores
      updateRow(tx, apiResult);
    } else {
      // API failed mid-stream — fall back to simulation for this row
      const simResult = simulateScores(tx);
      updateRow(tx, simResult);
    }

  } else {
    // API offline — use simulation immediately, no pending state needed
    const simResult = simulateScores(tx);
    const { tr, status } = buildRow(tx, simResult);
    tbody.insertBefore(tr, tbody.firstChild);

    if (simResult.isFraud || simResult.riskScore >= 76) {
      monitorFraud++;
      if (typeof pushAlert === 'function') {
        pushAlert({ ...tx, riskScore: simResult.riskScore });
      }
      if (typeof AppState !== 'undefined') AppState.fraudCount++;
    } else {
      monitorLegit++;
    }

    if (typeof AppState !== 'undefined') AppState.txCount++;
    updateMonitorCounters();
    if (typeof updateDashboardCounters === 'function') updateDashboardCounters();
  }

  // Always update txCount on the AppState (one per transaction)
  if (useAPI && typeof AppState !== 'undefined') {
    AppState.txCount++;
    if (typeof updateDashboardCounters === 'function') updateDashboardCounters();
  }

  // Update trend chart
  if (typeof updateTrendData === 'function') {
    updateTrendData({ isFraud: false, riskScore: 0 }); // tick the chart
  }
}

// ─── Monitor controls ─────────────────────────────────────────────────────────
function updateMonitorCounters() {
  const el = (id) => document.getElementById(id);
  if (el('mon-total')) el('mon-total').textContent = monitorTotal.toLocaleString();
  if (el('mon-fraud')) el('mon-fraud').textContent = monitorFraud.toLocaleString();
  if (el('mon-legit')) el('mon-legit').textContent = monitorLegit.toLocaleString();
}

function startMonitor() {
  if (monitorRunning) return;
  monitorRunning = true;

  const btn = document.getElementById('btn-pause-monitor');
  if (btn) { btn.textContent = '⏸ Pause'; btn.classList.add('active'); }

  monitorInterval = setInterval(() => {
    processTransaction();   // async — fires and doesn't block the interval
  }, 1800);                 // slightly longer gap to allow API response time
}

function stopMonitor() {
  if (!monitorRunning) return;
  monitorRunning = false;
  clearInterval(monitorInterval);
  monitorInterval = null;

  const btn = document.getElementById('btn-pause-monitor');
  if (btn) { btn.textContent = '▶ Resume'; btn.classList.remove('active'); }
}

function toggleMonitor() {
  if (monitorRunning) stopMonitor(); else startMonitor();
}

function clearMonitor() {
  const tbody = document.getElementById('tx-tbody');
  if (tbody) tbody.innerHTML = '';
  monitorTotal = 0;
  monitorFraud = 0;
  monitorLegit = 0;
  updateMonitorCounters();
}
