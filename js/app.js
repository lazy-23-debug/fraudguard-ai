/**
 * app.js
 * ─────────────────────────────────────────────────────────
 * Main application controller.
 * Handles: navigation, clock, global state, dashboard
 * counters, fraud alerts panel, and section initialization.
 *
 * Load order: predictor.js → transactions.js → charts.js → app.js
 *
 * Caleb University · CSC 400 · 22/10407
 */

'use strict';

// ─── Global application state ─────────────────────────────────────────────────
const AppState = {
  txCount:     0,
  fraudCount:  0,
  alertCount:  0,
  alerts:      [],                // recent fraud alert objects
  currentSection: 'dashboard',
  chartsInitialized: {}           // tracks which sections have had charts drawn
};

// ─── Section metadata (page title / subtitle) ─────────────────────────────────
const SECTION_META = {
  dashboard:   { title: 'Overview Dashboard',        sub: 'Hybrid ML Framework — Autoencoder + XGBoost Ensemble' },
  analyzer:    { title: 'Transaction Analyzer',      sub: 'Input a transaction and receive a SHAP-attributed fraud score' },
  monitor:     { title: 'Live Transaction Monitor',  sub: 'Real-time NIP / USSD / POS / ATM / Card feed' },
  performance: { title: 'Model Performance',         sub: 'ULB Credit Card Dataset — 284,807 transactions, 492 fraud cases' },
  shap:        { title: 'SHAP Explainability',       sub: 'SHapley Additive exPlanations — feature attribution analysis' },
  about:       { title: 'About This Project',        sub: 'Hybrid ML Framework for Fraud Detection — Caleb University, 2025' }
};

// ════════════════════════════════════════════════════════════════
//  INITIALIZATION
// ════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
  setupNavigation();
  startClock();
  showSection('dashboard');   // init dashboard (including its charts)
  startMonitor();             // kick off live transaction stream
  updateDashboardCounters();  // zero out counter elements
});

// ════════════════════════════════════════════════════════════════
//  MODE NOTICE — syncs with API health check in predictor.js
// ════════════════════════════════════════════════════════════════

function updateModeNotice(online) {
  const bar  = document.getElementById('mode-notice-bar');
  const icon = document.getElementById('mode-notice-icon');
  const text = document.getElementById('mode-notice-text');
  if (!bar) return;

  if (online) {
    bar.classList.add('online');
    icon.textContent = '🟢';
    text.innerHTML   = '<strong>Real Model Active</strong> — predictions are coming from the trained ' +
      'Autoencoder + XGBoost pipeline via FastAPI. ' +
      'SHAP values shown are from the actual trained model.';
  } else {
    bar.classList.remove('online');
    icon.textContent = '🟡';
    text.innerHTML   = '<strong>Simulation Mode</strong> — JavaScript heuristic scoring active. ' +
      'To use the real model: run <code>python ml/train.py</code> then ' +
      '<code>uvicorn ml.api:app --reload</code>.';
  }
}

// Expose so predictor.js can call it after each health check
window.__updateModeNotice = updateModeNotice;

// ════════════════════════════════════════════════════════════════
//  SIDEBAR TOGGLE (mobile)
// ════════════════════════════════════════════════════════════════

function toggleSidebar() {
  const sidebar  = document.querySelector('.sidebar');
  const overlay  = document.getElementById('sidebar-overlay');
  const isOpen   = sidebar.classList.contains('open');
  if (isOpen) {
    closeSidebar();
  } else {
    sidebar.classList.add('open');
    overlay.classList.add('open');
    document.body.style.overflow = 'hidden'; // prevent bg scroll
  }
}

function closeSidebar() {
  const sidebar = document.querySelector('.sidebar');
  const overlay = document.getElementById('sidebar-overlay');
  sidebar.classList.remove('open');
  overlay.classList.remove('open');
  document.body.style.overflow = '';
}

// ════════════════════════════════════════════════════════════════
//  NAVIGATION
// ════════════════════════════════════════════════════════════════

function setupNavigation() {
  document.querySelectorAll('.nav-item').forEach(btn => {
    btn.addEventListener('click', () => {
      const section = btn.dataset.section;
      if (section) {
        showSection(section);
        // Auto-close sidebar on mobile after navigation
        if (window.innerWidth <= 768) closeSidebar();
      }
    });
  });
}

function showSection(name) {
  // ── Update active nav item ────────────────────────────────────────────────
  document.querySelectorAll('.nav-item').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.section === name);
  });

  // ── Swap visible section ──────────────────────────────────────────────────
  document.querySelectorAll('.section').forEach(sec => {
    sec.classList.remove('active');
  });
  const target = document.getElementById('section-' + name);
  if (target) target.classList.add('active');

  // ── Update page title ─────────────────────────────────────────────────────
  const meta = SECTION_META[name];
  if (meta) {
    const titleEl = document.getElementById('page-title');
    const subEl   = document.getElementById('page-subtitle');
    if (titleEl) titleEl.textContent = meta.title;
    if (subEl)   subEl.textContent   = meta.sub;
  }

  // ── Initialize charts for this section (once only) ────────────────────────
  if (!AppState.chartsInitialized[name]) {
    AppState.chartsInitialized[name] = true;
    if (typeof initChartsForSection === 'function') {
      // Small delay so the section is visible and has correct dimensions
      setTimeout(() => initChartsForSection(name), 60);
    }
  }

  AppState.currentSection = name;
}

// ════════════════════════════════════════════════════════════════
//  LIVE CLOCK
// ════════════════════════════════════════════════════════════════

function startClock() {
  function tick() {
    const el = document.getElementById('system-clock');
    if (el) {
      el.textContent = new Date().toLocaleTimeString('en-GB', {
        hour12: false,
        timeZoneName: 'short'
      });
    }
  }
  tick();
  setInterval(tick, 1000);
}

// ════════════════════════════════════════════════════════════════
//  DASHBOARD COUNTERS
// ════════════════════════════════════════════════════════════════

function updateDashboardCounters() {
  const set = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  };

  const fraudRate = AppState.txCount > 0
    ? ((AppState.fraudCount / AppState.txCount) * 100).toFixed(2)
    : '0.00';

  set('dash-tx-count',    AppState.txCount.toLocaleString());
  set('dash-fraud-count', AppState.fraudCount.toLocaleString());
  set('dash-fraud-rate',  'Rate: ' + fraudRate + '%');
  set('topbar-alert-count', AppState.alertCount);
}

// ════════════════════════════════════════════════════════════════
//  FRAUD ALERTS PANEL
// ════════════════════════════════════════════════════════════════

/**
 * Called from transactions.js whenever a transaction is
 * classified as FRAUD (risk score ≥ 76).
 */
function pushAlert(tx) {
  AppState.alertCount++;
  AppState.alerts.unshift(tx);
  if (AppState.alerts.length > 20) AppState.alerts.pop();

  renderAlerts();
  updateDashboardCounters();
}

function renderAlerts() {
  const list  = document.getElementById('alerts-list');
  const badge = document.getElementById('alert-total-badge');
  if (!list) return;

  if (AppState.alerts.length === 0) {
    list.innerHTML = '<div class="empty-state">No alerts yet. System is actively monitoring transactions...</div>';
    if (badge) badge.textContent = '0 Active';
    return;
  }

  if (badge) badge.textContent = AppState.alerts.length + ' Active';

  list.innerHTML = AppState.alerts.slice(0, 8).map(tx => `
    <div class="alert-item">
      <span class="alert-dot"></span>
      <div class="alert-body">
        <strong>${tx.id}</strong> &mdash;
        ₦${tx.amount.toLocaleString()} ${tx.type} from <strong>${tx.account}</strong>
        <span class="text-red"> · Risk Score: ${tx.riskScore}/100</span>
      </div>
      <span class="alert-time">${tx.time}</span>
    </div>
  `).join('');
}
