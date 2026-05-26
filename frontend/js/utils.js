// ============================================
// VRITTIFY ASTRA – Shared Utilities v3.4
// Fixes:
//   - apiCall() does NOT redirect on 401 for /auth/ endpoints
//     (wrong password showed "File not found" because redirect fired)
//   - redirectToHome() fixed for file:// and http:// at any nesting depth
//   - All template literals use proper backticks
//   - showExpiredBanner CSS string fixed
// ============================================

const API_BASE = 'http://localhost:5000/api';

// ── Token expiry check ────────────────────────────────────────────────────
function isTokenExpired(token) {
  try {
    const payload = JSON.parse(atob(token.split('.')[1]));
    return payload.exp * 1000 < Date.now();
  } catch {
    return true;
  }
}

// ── API Helper ────────────────────────────────────────────────────────────
// FIX: Auth endpoints (/auth/login, /auth/register) must NOT trigger
// redirectToHome() on 401 — that caused "File not found" on wrong password.
// Only protected-route 401s (expired session) should redirect.
async function apiCall(endpoint, method = 'GET', data = null, isFormData = false) {
  const isAuthEndpoint = endpoint.startsWith('/auth/');

  const token = getToken();
  // Only check expiry for non-auth calls (no token needed for auth endpoints)
  if (!isAuthEndpoint && token && isTokenExpired(token)) {
    clearAuth();
    redirectToHome();
    throw new Error('Session expired. Please log in again.');
  }

  const opts = { method, headers: {} };
  if (token) opts.headers['Authorization'] = `Bearer ${token}`;
  if (data && !isFormData) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(data);
  } else if (data && isFormData) {
    opts.body = data;
  }

  const res  = await fetch(`${API_BASE}${endpoint}`, opts);
  const json = await res.json().catch(() => ({ error: 'Invalid server response' }));

  // FIX: Only redirect on 401 for protected routes, NOT auth routes.
  // Auth routes return 401 for bad credentials — that's a normal error,
  // not a session expiry. Redirecting here caused "File not found".
  if (res.status === 401 && !isAuthEndpoint) {
    clearAuth();
    redirectToHome();
    throw new Error('Session expired. Please log in again.');
  }

  if (!res.ok) throw new Error(json.error || `HTTP ${res.status}`);
  return json;
}

// ── Auth helpers ──────────────────────────────────────────────────────────
function getToken() { return localStorage.getItem('vr_token'); }
function getUser()  { return JSON.parse(localStorage.getItem('vr_user') || 'null'); }

function setAuth(token, user) {
  localStorage.setItem('vr_token', token);
  localStorage.setItem('vr_user', JSON.stringify(user));
}

function clearAuth() {
  localStorage.removeItem('vr_token');
  localStorage.removeItem('vr_user');
}

// ── Redirect to home/login ────────────────────────────────────────────────
// Project structure:
//   vrittify-astra/
//     frontend/
//       pages/
//         index.html               ← landing page (SAME folder as dashboards)
//         student-login.html
//         teacher-login.html
//         student-dashboard.html
//         teacher-dashboard.html
//
// All HTML files are in the SAME pages/ directory.
// So "go home" = just navigate to index.html in the same folder.
function redirectToHome() {
  // All pages are siblings inside frontend/pages/ — index.html is right there.
  window.location.href = 'index.html';
}

function logout() {
  clearAuth();
  redirectToHome();
}

// ── requireAuth ───────────────────────────────────────────────────────────
function requireAuth(role) {
  const user  = getUser();
  const token = getToken();

  if (!user || !token) {
    clearAuth();
    redirectToHome();
    return null;
  }

  if (isTokenExpired(token)) {
    clearAuth();
    showExpiredBanner();
    setTimeout(redirectToHome, 1500);
    return null;
  }

  if (role && user.role !== role) {
    redirectToHome();
    return null;
  }

  return user;
}

function showExpiredBanner() {
  const div = document.createElement('div');
  div.style.cssText = [
    'position:fixed', 'top:0', 'left:0', 'right:0', 'z-index:9999',
    'background:#ef4444', 'color:#fff', 'text-align:center',
    'padding:16px', 'font-family:sans-serif', 'font-size:14px',
  ].join(';');
  div.textContent = '⏱ Session expired. Redirecting to login…';
  document.body.prepend(div);
}

// ── Toast notifications ───────────────────────────────────────────────────
function showToast(message, type = 'info', duration = 3500) {
  let c = document.querySelector('.toast-container');
  if (!c) {
    c = document.createElement('div');
    c.className = 'toast-container';
    document.body.appendChild(c);
  }
  const icons = { success: '✅', error: '❌', info: 'ℹ️', warning: '⚠️' };
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.innerHTML = `<span>${icons[type] || 'ℹ️'}</span><span>${message}</span>`;
  c.appendChild(t);
  setTimeout(() => {
    t.style.cssText = 'opacity:0;transform:translateY(10px);transition:all 0.3s';
    setTimeout(() => t.remove(), 300);
  }, duration);
}

// ── Modal helpers ─────────────────────────────────────────────────────────
function openModal(id)  { document.getElementById(id)?.classList.add('show'); }
function closeModal(id) { document.getElementById(id)?.classList.remove('show'); }
document.addEventListener('click', e => {
  if (e.target.classList.contains('modal-overlay')) e.target.classList.remove('show');
});

// ── Format helpers ────────────────────────────────────────────────────────
function formatDate(iso) {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleString('en-IN', {
      day: '2-digit', month: 'short', year: 'numeric',
      hour: '2-digit', minute: '2-digit',
    });
  } catch { return iso; }
}

function scoreColor(s) {
  return s >= 8 ? 'var(--success)' : s >= 5 ? 'var(--warning)' : 'var(--danger)';
}
function scoreLabel(s) {
  if (s >= 9) return 'Outstanding';
  if (s >= 8) return 'Excellent';
  if (s >= 7) return 'Good';
  if (s >= 6) return 'Satisfactory';
  if (s >= 5) return 'Average';
  if (s >= 3) return 'Below Average';
  return 'Needs Improvement';
}

// ── Tab switching ─────────────────────────────────────────────────────────
function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const g = btn.dataset.group, t = btn.dataset.tab;
      document.querySelectorAll(`.tab-btn[data-group="${g}"]`).forEach(b => b.classList.remove('active'));
      document.querySelectorAll(`.tab-panel[data-group="${g}"]`).forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.querySelector(`.tab-panel[data-group="${g}"][data-tab="${t}"]`)?.classList.add('active');
    });
  });
}

// ── Sidebar user fill ─────────────────────────────────────────────────────
function fillSidebarUser(user) {
  const n = document.querySelector('.user-name');
  const r = document.querySelector('.user-role');
  const a = document.querySelector('.user-avatar');
  if (n && user) n.textContent = user.name || user.email;
  if (r && user) r.textContent = user.role === 'teacher' ? '👩‍🏫 Teacher' : '🎓 Student';
  if (a && user) a.textContent = (user.name || user.email || 'U')[0].toUpperCase();
}

// ── Animate numbers ───────────────────────────────────────────────────────
function animateNumber(el, target, duration = 800) {
  if (!el) return;
  let s = 0;
  const step = target / (duration / 16);
  const tick = () => {
    s = Math.min(s + step, target);
    el.textContent = Number.isInteger(target) ? Math.round(s) : s.toFixed(1);
    if (s < target) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

// ── Active nav ────────────────────────────────────────────────────────────
function setActiveNav(id) {
  document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
  document.getElementById(id)?.classList.add('active');
}

// ── Shared badge builders ─────────────────────────────────────────────────
function statusBadge(status) {
  const map = {
    submitted: '<span class="badge badge-submitted">● Submitted</span>',
    reviewing: '<span class="badge badge-reviewing">● Reviewing</span>',
    scored:    '<span class="badge badge-scored">● Scored</span>',
  };
  return map[status] || `<span class="badge">${status || 'unknown'}</span>`;
}

function renderAnalysisBars(analysis) {
  if (!analysis || !Object.keys(analysis).length) return '';
  const items = [
    { label: 'Originality',             key: 'originality',    invert: false },
    { label: 'AI Content Probability',  key: 'ai_probability', invert: true  },
    { label: 'Copy Similarity',         key: 'copy_similarity',invert: true  },
    { label: 'Handwriting Consistency', key: 'hw_consistency', invert: false },
    { label: 'Style Authenticity',      key: 'style_score',    invert: false },
    { label: 'OCR Confidence',          key: 'ocr_confidence', invert: false },
  ];
  return `
    <div class="divider"></div>
    <div style="font-family:var(--font-display);font-size:14px;font-weight:700;margin-bottom:16px">
      AI Analysis Breakdown
    </div>
    <div class="analysis-grid">
      ${items.map(item => {
        const val = analysis[item.key];
        if (val === undefined || val === null) return '';
        const pct = Math.round(val * 100);
        const cls = item.invert
          ? (pct > 60 ? 'high'  : pct > 30 ? 'medium' : 'low')
          : (pct > 60 ? 'low'   : pct > 30 ? 'medium' : 'high');
        return `
          <div class="analysis-item">
            <div class="analysis-header">
              <span class="analysis-label">${item.label}</span>
              <span class="analysis-value ${cls}">${pct}%</span>
            </div>
            <div class="progress-track">
              <div class="progress-fill" style="width:${pct}%"></div>
            </div>
          </div>`;
      }).join('')}
    </div>`;
}

// ── Export ────────────────────────────────────────────────────────────────
window.VSTRA = {
  apiCall, getToken, getUser, setAuth, clearAuth, logout, requireAuth,
  showToast, openModal, closeModal, formatDate, scoreColor, scoreLabel,
  initTabs, fillSidebarUser, animateNumber, setActiveNav,
  statusBadge, renderAnalysisBars,
};