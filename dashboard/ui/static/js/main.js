/* FJSSP Research Workbench — shared JS utilities */
'use strict';

// ── API helpers ──────────────────────────────────────────────
const API = {
  async get(url) {
    const r = await fetch(url);
    return r.json();
  },
  async post(url, body) {
    const r = await fetch(url, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    return r.json();
  },
};

// ── Task streaming ────────────────────────────────────────────
function streamTask(taskId, {onLog, onProgress, onDone, onError} = {}) {
  const es = new EventSource(`/api/tasks/${taskId}/stream`);
  es.onmessage = (e) => {
    const d = JSON.parse(e.data);
    if (d.error) { es.close(); onError?.(d.error); return; }
    const logs = d.log || [];
    if (logs.length > 0) onLog?.(logs[logs.length - 1]);
    onProgress?.(d.progress || 0);
    if (d.status === 'done') { es.close(); onDone?.(d.result); }
    if (d.status === 'error') { es.close(); onError?.(d.error); }
  };
  es.onerror = () => { es.close(); onError?.('Connection lost'); };
  return es;
}

// ── Progress card helper ──────────────────────────────────────
function makeProgressCard(container, title) {
  container.innerHTML = `
    <div class="wb-card">
      <div class="wb-card-header">
        <i class="fa fa-spinner fa-spin text-accent"></i>
        <span class="wb-card-title">${title}</span>
        <span id="prog-pct" class="badge bg-secondary ms-auto">0%</span>
      </div>
      <div class="progress mb-3"><div id="prog-bar" class="progress-bar" style="width:0%"></div></div>
      <div id="prog-log" class="log-console"></div>
    </div>`;
  return {
    setProgress(p) {
      document.getElementById('prog-bar').style.width = p + '%';
      document.getElementById('prog-pct').textContent = p + '%';
    },
    appendLog(msg) {
      const el = document.getElementById('prog-log');
      el.textContent += msg + '\n';
      el.scrollTop = el.scrollHeight;
    },
  };
}

// ── Plotly dark layout defaults ───────────────────────────────
const DARK_LAYOUT = {
  paper_bgcolor: '#1c2128',
  plot_bgcolor:  '#1c2128',
  font: { color: '#e6edf3', size: 12 },
  xaxis: { gridcolor: '#30363d', zerolinecolor: '#30363d' },
  yaxis: { gridcolor: '#30363d', zerolinecolor: '#30363d' },
  margin: { t: 40, r: 20, b: 50, l: 60 },
  legend: { bgcolor: 'rgba(0,0,0,0)', bordercolor: '#30363d' },
};

function plotLayout(extra = {}) {
  return Object.assign({}, DARK_LAYOUT, extra);
}

const COLORS = ['#4493f8','#3fb950','#f85149','#d29922','#bc8cff',
                 '#ff7b72','#56d364','#79c0ff','#ffa657','#e3b341'];

// ── Tables helper (plain, without DataTables) ─────────────────
function buildTable(headers, rows) {
  let html = '<table class="table table-hover"><thead><tr>';
  html += headers.map(h => `<th>${h}</th>`).join('');
  html += '</tr></thead><tbody>';
  rows.forEach(row => {
    html += '<tr>' + row.map(c => `<td>${c}</td>`).join('') + '</tr>';
  });
  html += '</tbody></table>';
  return html;
}

// ── Number formatting ─────────────────────────────────────────
function fmt(n, dec = 3) {
  if (n == null || !isFinite(n)) return '—';
  return Number(n).toFixed(dec);
}

function fmtPct(n) {
  if (n == null || !isFinite(n)) return '—';
  const s = Number(n).toFixed(1);
  return (n >= 0 ? '+' : '') + s + '%';
}

function fmtP(p) {
  if (p == null) return '—';
  if (p < 0.001) return '<0.001';
  return p.toFixed(3);
}

// ── Status indicator ──────────────────────────────────────────
function setStatus(msg, type = 'success') {
  const el = document.getElementById('status-indicator');
  if (!el) return;
  el.textContent = msg;
  el.className = `badge bg-${type}`;
}

// ── Toast notification ────────────────────────────────────────
function toast(msg, type = 'info') {
  const id = 'toast-' + Date.now();
  const colors = {info:'#4493f8', success:'#3fb950', error:'#f85149', warning:'#d29922'};
  const div = document.createElement('div');
  div.id = id;
  div.style.cssText = `
    position:fixed; bottom:24px; right:24px; z-index:9999;
    background:#1c2128; border:1px solid ${colors[type]||colors.info};
    color:#e6edf3; padding:12px 18px; border-radius:8px;
    font-size:13px; max-width:360px; box-shadow:0 4px 20px rgba(0,0,0,.5);
    animation: fadeIn .2s ease;
  `;
  div.textContent = msg;
  document.body.appendChild(div);
  setTimeout(() => div.remove(), 4000);
}

// ── Export CSV ────────────────────────────────────────────────
function exportCSV(headers, rows, filename = 'export.csv') {
  const lines = [headers.join(',')];
  rows.forEach(r => lines.push(r.map(c => `"${c}"`).join(',')));
  const blob = new Blob([lines.join('\n')], {type: 'text/csv'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
}
