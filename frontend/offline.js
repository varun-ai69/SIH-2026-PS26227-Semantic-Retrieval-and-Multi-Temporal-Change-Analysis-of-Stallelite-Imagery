/**
 * CANOPUS - AIR-GAPPED OFFLINE / ONLINE SOVEREIGN NETWORK MANAGER
 * ===============================================================
 * Provides robust dual-mode management:
 *   1. ONLINE INGESTION MODE: Connects to Sentinel-2 STAC and Maxar Wayback to ingest new AOIs.
 *   2. AIR-GAPPED OFFLINE MODE: Halts all external HTTP requests and web basemaps.
 *      RemoteCLIP Semantic Retrieval, HDBSCAN Clustering, Multi-Temporal Change Detection,
 *      and Analyst Review continue operating 100% locally from the database.
 * 
 * Includes:
 *   - Live Synchronized Satellite UTC Mission Clock (1-second tick)
 *   - Header Network Mode Pill & Instant Toggle Switch
 *   - First-Time Air-Gapped Operations Briefing Modal
 *   - Tactical Air-Gapped Map Cover for index.html
 *   - Automatic browser network change listeners
 */

(function () {
  'use strict';

  // ============================================================
  // 1. BULLETPROOF LIVE UTC MISSION CLOCK
  // ============================================================
  function tickMissionClock() {
    const now = new Date();
    const h = String(now.getUTCHours()).padStart(2, '0');
    const m = String(now.getUTCMinutes()).padStart(2, '0');
    const s = String(now.getUTCSeconds()).padStart(2, '0');
    const timeStr = `${h}:${m}:${s} UTC`;

    const clockEls = document.querySelectorAll('#liveUtcClock, .mission-clock-time');
    clockEls.forEach(el => {
      el.textContent = timeStr;
    });
  }

  // Execute immediately so --:--:-- UTC never shows
  tickMissionClock();
  setInterval(tickMissionClock, 1000);
  window.updateLiveMissionClock = tickMissionClock;

  // ============================================================
  // 2. NETWORK MODE STATE MANAGEMENT
  // ============================================================
  const STORAGE_KEY_OFFLINE = 'canopus_offline_mode';
  const STORAGE_KEY_BRIEFING = 'canopus_offline_briefing_seen';

  function getLocalOfflineState() {
    return localStorage.getItem(STORAGE_KEY_OFFLINE) === 'true';
  }

  function setLocalOfflineState(val) {
    localStorage.setItem(STORAGE_KEY_OFFLINE, val ? 'true' : 'false');
  }

  // Sync mode with FastAPI Backend
  async function syncBackendMode(isOffline) {
    try {
      const res = await fetch('/api/v1/system/mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ offline_mode: Boolean(isOffline) })
      });
      if (res.ok) {
        const data = await res.json();
        return data;
      }
    } catch (e) {
      console.warn('[Canopus Network Manager] Backend sync warning:', e);
    }
    return null;
  }

  async function fetchBackendMode() {
    try {
      const res = await fetch('/api/v1/system/mode');
      if (res.ok) {
        const data = await res.json();
        return data.offline_mode;
      }
    } catch (e) {
      console.warn('[Canopus Network Manager] Could not query backend mode:', e);
    }
    return null;
  }

  // Apply visual state across header, map, and modals
  function applyNetworkModeUI(isOffline) {
    // 1. Header Pill Update
    const pill = document.getElementById('networkModeBtn');
    if (pill) {
      if (isOffline) {
        pill.className = 'network-mode-pill offline';
        pill.innerHTML = `
          <div class="mode-dot"></div>
          <span>OFFLINE</span>
        `;
        pill.title = 'Air-Gapped Mode: External tiles paused. Click to Go Online.';
      } else {
        pill.className = 'network-mode-pill';
        pill.innerHTML = `
          <div class="mode-dot"></div>
          <span>ONLINE</span>
        `;
        pill.title = 'Online Ingest Mode. Click to switch to Air-Gapped Mode.';
      }
    }

    // 2. Map Cover in index.html
    const mapCover = document.getElementById('offlineMapCover');
    if (mapCover) {
      mapCover.style.display = isOffline ? 'flex' : 'none';
      if (!isOffline && window.map && typeof window.map.invalidateSize === 'function') {
        setTimeout(() => window.map.invalidateSize(), 250);
      }
    }

    // 3. Broadcast Event for other modules
    window.dispatchEvent(new CustomEvent('canopus:network-mode-change', {
      detail: { offline: isOffline }
    }));
  }

  // Toggle Function exposed globally
  window.toggleNetworkMode = async function () {
    const currentState = getLocalOfflineState();
    const newState = !currentState;
    setLocalOfflineState(newState);
    applyNetworkModeUI(newState);

    if (window.playTacticalClick) window.playTacticalClick();
    if (window.showTacticalToast) {
      if (newState) {
        window.showTacticalToast('AIR-GAPPED OFFLINE DEFENSE MODE ACTIVATED', 'info');
      } else {
        window.showTacticalToast('ONLINE INGESTION MODE ACTIVATED', 'success');
      }
    }

    await syncBackendMode(newState);
  };

  window.setNetworkMode = async function (isOffline) {
    setLocalOfflineState(isOffline);
    applyNetworkModeUI(isOffline);
    if (window.playTacticalClick) window.playTacticalClick();
    await syncBackendMode(isOffline);
  };

  // ============================================================
  // 3. HEADER NETWORK PILL INJECTION / ATTACHMENT
  // ============================================================
  function ensureHeaderNetworkPill() {
    const headerStatus = document.querySelector('.header-status');
    if (!headerStatus) return;

    let pill = document.getElementById('networkModeBtn');
    if (!pill) {
      pill = document.createElement('div');
      pill.id = 'networkModeBtn';
      pill.className = 'network-mode-pill';
      pill.onclick = () => window.toggleNetworkMode();
      // Insert right after the mission clock pill or at beginning of headerStatus
      const clockPill = headerStatus.querySelector('.mission-clock-pill');
      if (clockPill && clockPill.nextSibling) {
        headerStatus.insertBefore(pill, clockPill.nextSibling);
      } else {
        headerStatus.prepend(pill);
      }
    }
    applyNetworkModeUI(getLocalOfflineState());
  }

  // ============================================================
  // 4. MAP COVER HUD INJECTION (index.html)
  // ============================================================
  function ensureMapCover() {
    const mapEl = document.getElementById('leafletMap');
    if (!mapEl) return; // Not on the map page

    let cover = document.getElementById('offlineMapCover');
    if (!cover) {
      cover = document.createElement('div');
      cover.id = 'offlineMapCover';
      cover.style.display = 'none';
      cover.innerHTML = `
        <div class="offline-cover-card">
          <div class="offline-shield-icon">
            <svg class="ui-icon" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
              <line x1="8" y1="12" x2="16" y2="12"/>
            </svg>
          </div>
          <div class="offline-cover-title">AIR-GAPPED OFFLINE MODE ACTIVE</div>
          <div class="offline-cover-subtitle">
            External web tile basemaps (Carto/OSM/Esri) and live STAC satellite catalog discovery are paused to ensure zero outbound network emissions.
            All core neural intelligence modules remain <strong>100% operational locally</strong> from your sovereign archive.
          </div>

          <div class="offline-module-matrix">
            <div class="offline-module-item active">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>
              <span>Semantic Retrieval (RemoteCLIP) &bull; Local</span>
            </div>
            <div class="offline-module-item active">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>
              <span>Terrain Clustering (HDBSCAN) &bull; Local</span>
            </div>
            <div class="offline-module-item active">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>
              <span>Multi-Temporal Change Engine &bull; Local</span>
            </div>
            <div class="offline-module-item active">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>
              <span>Analyst Review &amp; Provenance &bull; Local</span>
            </div>
            <div class="offline-module-item paused">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
              <span>Cloud STAC Sentinel-2 Ingest &bull; Paused</span>
            </div>
            <div class="offline-module-item paused">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
              <span>Maxar Wayback WMTS &bull; Paused</span>
            </div>
          </div>

          <div class="offline-actions-row">
            <button class="offline-btn-primary" onclick="window.setNetworkMode(false)">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12.55a11 11 0 0 1 14.08 0"/><path d="M1.42 9a16 16 0 0 1 21.16 0"/><path d="M8.53 16.11a6 6 0 0 1 6.95 0"/><line x1="12" y1="20" x2="12.01" y2="20"/></svg>
              Connect to Internet / Go Online to Ingest AOI
            </button>
            <a href="/retrieval.html" class="offline-btn-secondary">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
              Semantic Retrieval
            </a>
            <a href="/clustering.html" class="offline-btn-secondary">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="4"/><circle cx="6" cy="6" r="2"/><circle cx="18" cy="6" r="2"/></svg>
              Clustering
            </a>
          </div>
        </div>
      `;

      // Insert cover inside map container
      const container = mapEl.parentElement || mapEl;
      container.style.position = 'relative';
      container.appendChild(cover);
    }

    applyNetworkModeUI(getLocalOfflineState());
  }

  // ============================================================
  // 5. FIRST-TIME OPERATIONAL BRIEFING MODAL
  // ============================================================
  function ensureBriefingModal() {
    const hasSeenBriefing = localStorage.getItem(STORAGE_KEY_BRIEFING);
    if (hasSeenBriefing === 'true') return; // User already acknowledged

    let modal = document.getElementById('offlineBriefingModal');
    if (!modal) {
      modal = document.createElement('div');
      modal.id = 'offlineBriefingModal';
      modal.innerHTML = `
        <div class="offline-briefing-card">
          <div class="offline-briefing-header">
            <div style="display: flex; align-items: center; gap: 10px;">
              <svg class="ui-icon icon-cyan" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#06b6d4" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="m4.93 4.93 4.24 4.24"/><path d="m14.83 9.17 4.24-4.24"/><path d="m14.83 14.83 4.24 4.24"/><path d="m9.17 14.83-4.24 4.24"/></svg>
              <div>
                <div style="font-size: 14px; font-weight: 800; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.05em;">CANOPUS OPERATIONAL NETWORK BRIEFING</div>
                <div style="font-size: 11px; color: #94a3b8;">Strategic Air-Gapped vs Online Ingestion Architecture</div>
              </div>
            </div>
            <button onclick="dismissBriefingModal(false)" style="background: transparent; border: none; color: #64748b; font-size: 20px; cursor: pointer; line-height: 1;">&times;</button>
          </div>

          <div class="offline-briefing-body">
            <p style="margin-top: 0; color: #e2e8f0; font-size: 13px;">
              Canopus is currently operating in <strong>ONLINE MODE</strong> with active constellation telemetry:
            </p>

            <div class="offline-mode-comparison">
              <div class="mode-comparison-box online-box">
                <div class="mode-comparison-title">
                  <span style="display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #10b981;"></span>
                  Online Mode (Default)
                </div>
                <div style="font-size: 11px; color: #94a3b8; line-height: 1.5;">
                  Enables live on-demand ingestion of Areas of Interest (AOIs) from <strong>Sentinel-2 STAC</strong> and <strong>Maxar Wayback WMTS</strong> directly into your local vector database.
                </div>
              </div>

              <div class="mode-comparison-box offline-box">
                <div class="mode-comparison-title">
                  <span style="display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #f59e0b;"></span>
                  Air-Gapped Offline Mode
                </div>
                <div style="font-size: 11px; color: #94a3b8; line-height: 1.5;">
                  For secure defense facilities without internet. External tile requests and cloud catalogs are paused. <strong>All local AI retrieval, clustering, and change detection remain 100% active</strong>.
                </div>
              </div>
            </div>

            <div style="font-size: 11px; color: #94a3b8; background: rgba(6, 182, 212, 0.08); border: 1px solid rgba(6, 182, 212, 0.25); border-radius: 8px; padding: 10px 14px;">
              <strong style="color: #38bdf8;">Tip:</strong> You can toggle between Online and Air-Gapped modes at any time via the network mode pill in the top navigation bar.
            </div>
          </div>

          <div class="offline-briefing-footer">
            <label class="briefing-remember-check">
              <input type="checkbox" id="briefingRememberCheckbox" checked style="accent-color: #06b6d4;">
              <span>Don't show this briefing on startup</span>
            </label>
            <div style="display: flex; gap: 10px;">
              <button class="offline-btn-secondary" onclick="dismissBriefingModal(true)" style="font-size: 11px; padding: 8px 14px; border-color: rgba(245, 158, 11, 0.4); color: #fbbf24;">
                Switch to Offline Mode
              </button>
              <button class="offline-btn-primary" onclick="dismissBriefingModal(false)" style="font-size: 11px; padding: 8px 16px; background: #06b6d4; color: #000;">
                Stay in Online Mode (Ingest AOIs)
              </button>
            </div>
          </div>
        </div>
      `;
      document.body.appendChild(modal);
    }

    // Show after slight delay for smooth page presentation
    setTimeout(() => {
      modal.style.display = 'flex';
    }, 400);
  }

  window.dismissBriefingModal = function (setOffline) {
    const modal = document.getElementById('offlineBriefingModal');
    if (modal) modal.style.display = 'none';

    const chk = document.getElementById('briefingRememberCheckbox');
    if (chk && chk.checked) {
      localStorage.setItem(STORAGE_KEY_BRIEFING, 'true');
    }

    if (setOffline) {
      window.setNetworkMode(true);
    }
  };

  // ============================================================
  // 6. AUTOMATIC BROWSER NETWORK LISTENERS
  // ============================================================
  window.addEventListener('offline', () => {
    if (window.showTacticalToast) {
      window.showTacticalToast('Physical network connection lost. Operating in Air-Gapped Mode.', 'warning');
    }
    window.setNetworkMode(true);
  });

  window.addEventListener('online', () => {
    if (window.showTacticalToast) {
      window.showTacticalToast('Physical network connection restored.', 'info');
    }
    // If user was offline, refresh map
    const map = window.map;
    if (map && typeof map.invalidateSize === 'function') {
      setTimeout(() => map.invalidateSize(), 300);
    }
  });

  // ============================================================
  // 7. INITIALIZATION
  // ============================================================
  async function init() {
    tickMissionClock();
    ensureHeaderNetworkPill();
    ensureMapCover();
    ensureBriefingModal();

    // Check backend mode and sync
    const backendOffline = await fetchBackendMode();
    if (backendOffline !== null) {
      // If backend says offline or local says offline
      const shouldBeOffline = getLocalOfflineState() || backendOffline;
      setLocalOfflineState(shouldBeOffline);
      applyNetworkModeUI(shouldBeOffline);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
