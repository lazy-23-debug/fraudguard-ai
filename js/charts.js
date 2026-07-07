/**
 * charts.js
 * ─────────────────────────────────────────────────────────
 * Defines and initializes every Chart.js chart in the app.
 * Charts are initialized lazily the first time their
 * parent section is visited, so hidden canvases never
 * render with incorrect dimensions.
 *
 * Caleb University · CSC 400 · 22/10407
 */

'use strict';

// ─── Chart instances (stored so they can be updated / destroyed) ──────────────
const Charts = {};

// ─── Shared Chart.js theme (matches CSS variables) ───────────────────────────
const THEME = {
  cyan:    '#00D9FF',
  green:   '#2ED573',
  red:     '#FF4757',
  yellow:  '#FFD32A',
  orange:  '#FF7F50',
  purple:  '#A29BFE',
  bg:      'rgba(0,0,0,0)',
  grid:    'rgba(255,255,255,0.04)',
  tick:    '#3D4466',
  label:   '#7B85A8',
  font:    "'Inter', sans-serif"
};

// Default options reused across multiple charts
const BASE_OPTS = {
  responsive: true,
  maintainAspectRatio: true,
  plugins: {
    legend: {
      labels: {
        color: THEME.label,
        font: { family: THEME.font, size: 12 },
        boxWidth: 10,
        padding: 14
      }
    },
    tooltip: {
      backgroundColor: '#101530',
      borderColor: 'rgba(255,255,255,0.08)',
      borderWidth: 1,
      titleColor: '#E8EBF5',
      bodyColor: '#7B85A8',
      padding: 10,
      titleFont: { family: THEME.font, size: 12, weight: 'bold' },
      bodyFont:  { family: THEME.font, size: 11.5 }
    }
  },
  scales: {
    x: {
      ticks: { color: THEME.tick, font: { family: THEME.font, size: 11 } },
      grid:  { color: THEME.grid }
    },
    y: {
      ticks: { color: THEME.tick, font: { family: THEME.font, size: 11 } },
      grid:  { color: THEME.grid }
    }
  }
};

// ─── Model names (shared by performance charts) ───────────────────────────────
const MODEL_LABELS = [
  'Logistic Reg.', 'Random Forest', 'XGBoost',
  'ANN', 'LSTM', 'Isolation Forest', 'Hybrid ⟨E⟩'
];

// ════════════════════════════════════════════════════════════════
//  DASHBOARD CHARTS
// ════════════════════════════════════════════════════════════════

// ─── Rolling 24-hour trend data ──────────────────────────────────────────────
const HOURS = Array.from({ length: 24 }, (_, i) => {
  const h = (new Date().getHours() - 23 + i + 24) % 24;
  return String(h).padStart(2, '0') + ':00';
});

// Simulated baseline data (legit & fraud per hour)
const trendLegit = Array.from({ length: 24 }, () => Math.round(2000 + Math.random() * 3000));
const trendFraud = Array.from({ length: 24 }, () => Math.round(Math.random() * 8));

function initDashboardCharts() {
  // ── 1. Fraud Detection Trend (line) ──────────────────────────────────────
  const trendCtx = document.getElementById('trendChart');
  if (!trendCtx || Charts.trend) return;

  Charts.trend = new Chart(trendCtx, {
    type: 'line',
    data: {
      labels: HOURS,
      datasets: [
        {
          label: 'Legitimate Transactions',
          data: [...trendLegit],
          borderColor: THEME.cyan,
          backgroundColor: 'rgba(0,217,255,0.06)',
          fill: true,
          tension: 0.45,
          pointRadius: 2,
          pointHoverRadius: 5,
          borderWidth: 2,
          yAxisID: 'yLeft'
        },
        {
          label: 'Fraud Detected',
          data: [...trendFraud],
          borderColor: THEME.red,
          backgroundColor: 'rgba(255,71,87,0.10)',
          fill: true,
          tension: 0.45,
          pointRadius: 3,
          pointHoverRadius: 6,
          borderWidth: 2,
          yAxisID: 'yRight'
        }
      ]
    },
    options: {
      ...BASE_OPTS,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        ...BASE_OPTS.plugins,
        legend: { ...BASE_OPTS.plugins.legend }
      },
      scales: {
        x: { ...BASE_OPTS.scales.x, ticks: { ...BASE_OPTS.scales.x.ticks, maxTicksLimit: 8 } },
        yLeft: {
          type: 'linear',
          position: 'left',
          ticks: { color: THEME.tick, font: { family: THEME.font, size: 10.5 } },
          grid:  { color: THEME.grid },
          title: { display: true, text: 'Legitimate', color: THEME.cyan, font: { size: 11 } }
        },
        yRight: {
          type: 'linear',
          position: 'right',
          ticks: { color: THEME.tick, font: { family: THEME.font, size: 10.5 } },
          grid: { drawOnChartArea: false },
          title: { display: true, text: 'Fraud', color: THEME.red, font: { size: 11 } }
        }
      }
    }
  });

  // ── 2. Transaction Type Distribution (doughnut) ───────────────────────────
  const distCtx = document.getElementById('distChart');
  if (!distCtx || Charts.dist) return;

  Charts.dist = new Chart(distCtx, {
    type: 'doughnut',
    data: {
      labels: ['NIP Transfer', 'USSD', 'POS', 'ATM', 'Card'],
      datasets: [{
        data: [38, 22, 18, 12, 10],
        backgroundColor: [
          THEME.cyan,
          THEME.yellow,
          THEME.green,
          THEME.orange,
          'rgba(162,155,254,0.8)'
        ],
        borderColor: '#0C1027',
        borderWidth: 3,
        hoverOffset: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      cutout: '65%',
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            color: THEME.label,
            font: { family: THEME.font, size: 11 },
            boxWidth: 10,
            padding: 10
          }
        },
        tooltip: BASE_OPTS.plugins.tooltip
      }
    }
  });
}

// ─── Called by transactions.js each time a new tx comes in ───────────────────
function updateTrendData(tx) {
  if (!Charts.trend) return;

  const ds = Charts.trend.data.datasets;
  const lastIdx = ds[0].data.length - 1;

  if (tx.isFraud || tx.riskScore >= 76) {
    ds[1].data[lastIdx] = (ds[1].data[lastIdx] || 0) + 1;
  } else {
    ds[0].data[lastIdx] = (ds[0].data[lastIdx] || 0) + 1;
  }

  Charts.trend.update('none'); // 'none' = no animation for live updates
}

// ════════════════════════════════════════════════════════════════
//  MODEL PERFORMANCE CHARTS
// ════════════════════════════════════════════════════════════════

function initPerformanceCharts() {
  // ── 3. AUC-ROC Bar Chart ─────────────────────────────────────────────────
  const aucCtx = document.getElementById('aucChart');
  if (aucCtx && !Charts.auc) {
    const aucValues = [0.975, 0.981, 0.996, 0.992, 0.994, 0.971, 0.998];
    const aucColors = MODEL_LABELS.map((_, i) =>
      i === 6 ? THEME.cyan : 'rgba(0,217,255,0.35)'
    );

    Charts.auc = new Chart(aucCtx, {
      type: 'bar',
      data: {
        labels: MODEL_LABELS,
        datasets: [{
          label: 'AUC-ROC',
          data: aucValues,
          backgroundColor: aucColors,
          borderColor: MODEL_LABELS.map((_, i) => i === 6 ? THEME.cyan : 'transparent'),
          borderWidth: 1.5,
          borderRadius: 4
        }]
      },
      options: {
        ...BASE_OPTS,
        indexAxis: 'y',
        plugins: { ...BASE_OPTS.plugins, legend: { display: false } },
        scales: {
          x: {
            ...BASE_OPTS.scales.x,
            min: 0.95,
            max: 1.0,
            ticks: {
              ...BASE_OPTS.scales.x.ticks,
              callback: (v) => v.toFixed(3)
            }
          },
          y: { ...BASE_OPTS.scales.y, grid: { display: false } }
        }
      }
    });
  }

  // ── 4. F1-Score Bar Chart ─────────────────────────────────────────────────
  const f1Ctx = document.getElementById('f1Chart');
  if (f1Ctx && !Charts.f1) {
    const f1Values = [83.1, 88.5, 93.0, 90.5, 91.2, 83.8, 94.6];
    const f1Colors = MODEL_LABELS.map((_, i) =>
      i === 6 ? THEME.green : 'rgba(46,213,115,0.35)'
    );

    Charts.f1 = new Chart(f1Ctx, {
      type: 'bar',
      data: {
        labels: MODEL_LABELS,
        datasets: [{
          label: 'F1-Score (%)',
          data: f1Values,
          backgroundColor: f1Colors,
          borderColor: MODEL_LABELS.map((_, i) => i === 6 ? THEME.green : 'transparent'),
          borderWidth: 1.5,
          borderRadius: 4
        }]
      },
      options: {
        ...BASE_OPTS,
        indexAxis: 'y',
        plugins: { ...BASE_OPTS.plugins, legend: { display: false } },
        scales: {
          x: {
            ...BASE_OPTS.scales.x,
            min: 78,
            max: 100,
            ticks: {
              ...BASE_OPTS.scales.x.ticks,
              callback: (v) => v + '%'
            }
          },
          y: { ...BASE_OPTS.scales.y, grid: { display: false } }
        }
      }
    });
  }

  // ── 5. Precision vs Recall Scatter ────────────────────────────────────────
  const prCtx = document.getElementById('prChart');
  if (prCtx && !Charts.pr) {
    const prData = [
      { x: 76.4, y: 91.2,  label: 'Log. Reg.' },
      { x: 82.7, y: 95.1,  label: 'Rand. Forest' },
      { x: 88.6, y: 97.8,  label: 'XGBoost' },
      { x: 85.3, y: 96.4,  label: 'ANN' },
      { x: 86.1, y: 96.9,  label: 'LSTM' },
      { x: 79.6, y: 88.4,  label: 'Iso. Forest' },
      { x: 91.2, y: 98.3,  label: 'Hybrid ⟨E⟩' }
    ];

    Charts.pr = new Chart(prCtx, {
      type: 'scatter',
      data: {
        datasets: [{
          label: 'Models',
          data: prData.map((d, i) => ({ x: d.x, y: d.y })),
          backgroundColor: prData.map((d, i) =>
            i === 6 ? THEME.cyan : 'rgba(0,217,255,0.4)'
          ),
          pointRadius: prData.map((_, i) => i === 6 ? 10 : 7),
          pointHoverRadius: 12
        }]
      },
      options: {
        ...BASE_OPTS,
        plugins: {
          ...BASE_OPTS.plugins,
          legend: { display: false },
          tooltip: {
            ...BASE_OPTS.plugins.tooltip,
            callbacks: {
              label: (ctx) => {
                const d = prData[ctx.dataIndex];
                return ` ${d.label}: Recall ${d.x}% | Precision ${d.y}%`;
              }
            }
          }
        },
        scales: {
          x: {
            ...BASE_OPTS.scales.x,
            min: 73, max: 95,
            title: {
              display: true,
              text: 'Recall (%)',
              color: THEME.label,
              font: { family: THEME.font, size: 11 }
            },
            ticks: { ...BASE_OPTS.scales.x.ticks, callback: (v) => v + '%' }
          },
          y: {
            ...BASE_OPTS.scales.y,
            min: 85, max: 100,
            title: {
              display: true,
              text: 'Precision (%)',
              color: THEME.label,
              font: { family: THEME.font, size: 11 }
            },
            ticks: { ...BASE_OPTS.scales.y.ticks, callback: (v) => v + '%' }
          }
        }
      }
    });
  }
}

// ════════════════════════════════════════════════════════════════
//  SHAP CHARTS
// ════════════════════════════════════════════════════════════════

function initShapCharts() {
  // ── 6. Global Feature Importance (horizontal bar) ─────────────────────────
  const shapCtx = document.getElementById('shapBarChart');
  if (shapCtx && !Charts.shap) {
    // Features ordered by mean |SHAP| (from published ULB analyses + V31 from AE)
    const shapFeatures = [
      'V31 (AE Error)', 'V14', 'V10', 'Amount', 'V4',
      'V11', 'V17', 'V12', 'V3', 'Time'
    ];
    const shapValues = [0.48, 0.42, 0.35, 0.29, 0.26, 0.24, 0.18, 0.15, 0.12, 0.08];
    const shapColors = shapValues.map((_, i) => {
      if (i === 0) return THEME.cyan;       // V31 — highlight the AE contribution
      if (i <= 2)  return 'rgba(0,217,255,0.65)';
      if (i <= 4)  return 'rgba(0,217,255,0.45)';
      return 'rgba(0,217,255,0.28)';
    });

    Charts.shap = new Chart(shapCtx, {
      type: 'bar',
      data: {
        labels: shapFeatures,
        datasets: [{
          label: 'Mean |SHAP| value',
          data: shapValues,
          backgroundColor: shapColors,
          borderColor: shapColors.map((c, i) => i === 0 ? THEME.cyan : 'transparent'),
          borderWidth: 1.5,
          borderRadius: 4
        }]
      },
      options: {
        ...BASE_OPTS,
        indexAxis: 'y',
        plugins: {
          ...BASE_OPTS.plugins,
          legend: { display: false },
          tooltip: {
            ...BASE_OPTS.plugins.tooltip,
            callbacks: {
              label: (ctx) => ` Mean |SHAP| = ${ctx.parsed.x.toFixed(3)}`
            }
          }
        },
        scales: {
          x: {
            ...BASE_OPTS.scales.x,
            title: {
              display: true, text: 'Mean |SHAP| Value',
              color: THEME.label, font: { family: THEME.font, size: 11 }
            }
          },
          y: { ...BASE_OPTS.scales.y, grid: { display: false } }
        }
      }
    });
  }

  // ── 7. Feature Direction — Positive vs Negative SHAP ─────────────────────
  const dirCtx = document.getElementById('shapDirChart');
  if (dirCtx && !Charts.shapDir) {
    Charts.shapDir = new Chart(dirCtx, {
      type: 'bar',
      data: {
        labels: ['V31 (AE)', 'V14', 'Amount (hi)', 'V10', 'V11', 'Amount (lo)', 'Time (night)', 'V12'],
        datasets: [
          {
            label: 'Pushes Toward FRAUD',
            data: [0.44, 0.38, 0.27, 0.32, 0.21, 0.19, 0.14, 0.12],
            backgroundColor: 'rgba(255,71,87,0.65)',
            borderRadius: 4
          },
          {
            label: 'Pushes Toward LEGIT',
            data: [-0.06, -0.05, -0.03, -0.04, -0.03, 0, -0.01, -0.02],
            backgroundColor: 'rgba(46,213,115,0.55)',
            borderRadius: 4
          }
        ]
      },
      options: {
        ...BASE_OPTS,
        indexAxis: 'y',
        plugins: {
          ...BASE_OPTS.plugins,
          tooltip: BASE_OPTS.plugins.tooltip
        },
        scales: {
          x: {
            ...BASE_OPTS.scales.x,
            stacked: false,
            title: {
              display: true, text: 'SHAP Contribution',
              color: THEME.label, font: { family: THEME.font, size: 11 }
            }
          },
          y: { ...BASE_OPTS.scales.y, grid: { display: false } }
        }
      }
    });
  }
}

// ─── Called when a section is first activated ────────────────────────────────
function initChartsForSection(section) {
  switch (section) {
    case 'dashboard':   initDashboardCharts();   break;
    case 'performance': initPerformanceCharts(); break;
    case 'shap':        initShapCharts();        break;
  }
}
