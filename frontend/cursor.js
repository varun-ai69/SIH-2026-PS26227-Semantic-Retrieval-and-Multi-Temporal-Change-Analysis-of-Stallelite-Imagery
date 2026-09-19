/**
 * CANOPUS — ORBITAL MISSION CONTROL SATELLITE CURSOR ENGINE
 * ==========================================================
 * Signature Satellite Cursor with Lerped Inertia, Dynamic Heading Rotation,
 * Interactive Target Lock-On Reticles, and Click Shockwaves.
 *
 * Additive Only: Zero changes to backend, APIs, or database.
 */

(function () {
  'use strict';

  // Check hardware capability: fine pointer & hover required
  const isTouchDevice = () => window.matchMedia('(pointer: coarse)').matches;
  const prefersReducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  if (isTouchDevice() || prefersReducedMotion()) {
    return; // Calm fallback for touch / reduced motion
  }

  // State
  let enabled = localStorage.getItem('canopus_tactical_cursor') !== 'false';
  let isLoopRunning = false;
  let mouseX = window.innerWidth / 2;
  let mouseY = window.innerHeight / 2;
  let currentX = mouseX;
  let currentY = mouseY;
  let currentAngle = 0;
  let targetAngle = 0;
  let isIdle = false;
  let idleTimer = null;
  let activeFetches = 0;
  let currentHoverType = 'normal'; // 'normal', 'locked', 'threat', 'map', 'text'

  // Elements
  let rootEl = null;
  let satelliteEl = null;
  let reticleEl = null;

  // Apply Enabled / Disabled Visual & Physics State
  function applyCursorState() {
    if (enabled) {
      document.body.classList.add('tactical-cursor-active');
      if (rootEl) rootEl.style.display = 'block';
      if (!isLoopRunning) {
        isLoopRunning = true;
        requestAnimationFrame(renderLoop);
      }
    } else {
      document.body.classList.remove('tactical-cursor-active');
      if (rootEl) rootEl.style.display = 'none';
      isLoopRunning = false;
    }
    updateCursorToggleButton();
  }

  // Build DOM Structure
  function initDOM() {
    rootEl = document.createElement('div');
    rootEl.id = 'tactical-cursor-root';
    if (!enabled) {
      rootEl.style.display = 'none';
    }

    // Satellite Probe Follower
    satelliteEl = document.createElement('div');
    satelliteEl.className = 'satellite-probe';
    satelliteEl.innerHTML = `
      <svg class="satellite-svg" viewBox="0 0 48 48">
        <!-- Solar Array Left -->
        <rect x="2" y="19" width="13" height="10" rx="1.5" class="sat-solar" />
        <line x1="2" y1="24" x2="15" y2="24" class="sat-grid" />
        <line x1="8" y1="19" x2="8" y2="29" class="sat-grid" />
        <line x1="15" y1="24" x2="19" y2="24" class="sat-body" />

        <!-- Solar Array Right -->
        <rect x="33" y="19" width="13" height="10" rx="1.5" class="sat-solar" />
        <line x1="33" y1="24" x2="46" y2="24" class="sat-grid" />
        <line x1="39" y1="19" x2="39" y2="29" class="sat-grid" />
        <line x1="29" y1="24" x2="33" y2="24" class="sat-body" />

        <!-- Satellite Body Chassis -->
        <rect x="18" y="15" width="12" height="18" rx="2" class="sat-body" />
        <rect x="21" y="20" width="6" height="8" rx="1" class="sat-core" />

        <!-- Forward Sensor / High-Gain Dish -->
        <circle cx="24" cy="11" r="3.5" class="sat-body" />
        <line x1="24" y1="15" x2="24" y2="8" class="sat-antenna" />
        <circle cx="24" cy="7.5" r="1.5" class="sat-core" />

        <!-- Rear Nozzle -->
        <path d="M 21 33 L 27 33 L 26 36 L 22 36 Z" class="sat-body" />
      </svg>
    `;

    // Targeting Reticle Component (Clean rings & crosshairs, no text badges)
    reticleEl = document.createElement('div');
    reticleEl.className = 'targeting-reticle';
    reticleEl.innerHTML = `
      <div class="reticle-ring"></div>
      <div class="reticle-inner-ring"></div>
      <div class="reticle-crosshair"></div>
    `;
    satelliteEl.appendChild(reticleEl);

    rootEl.appendChild(satelliteEl);
    document.body.appendChild(rootEl);

    if (enabled) {
      document.body.classList.add('tactical-cursor-active');
    } else {
      document.body.classList.remove('tactical-cursor-active');
    }
  }

  // Hover target categorization
  function checkHoverTarget(target) {
    if (!target) return 'normal';

    // 1. Text input
    if (target.matches && target.matches('input[type=text], textarea, select, .fast-travel-input')) {
      return 'text';
    }

    // 2. Destructive / Threat actions
    if (target.closest && target.closest('.btn-danger, [data-action="reject"], .reject-btn, .btn-drop, .clear-btn')) {
      return 'threat';
    }

    // 3. Interactive clickable elements
    if (target.closest && target.closest('a, button, .result-card, .cluster-card, .nav-item, [onclick], .aoi-card, .timeline-tick, .leaflet-control, .badge-action')) {
      return 'locked';
    }

    // 4. Over Leaflet Map Canvas
    if (target.closest && target.closest('.leaflet-container')) {
      return 'map';
    }

    return 'normal';
  }

  // Event Listeners
  function initListeners() {
    window.addEventListener('mousemove', (e) => {
      if (!enabled) return;
      mouseX = e.clientX;
      mouseY = e.clientY;

      // Reset Idle Timer
      if (isIdle) {
        isIdle = false;
        if (satelliteEl) satelliteEl.classList.remove('standby');
      }
      clearTimeout(idleTimer);
      idleTimer = setTimeout(() => {
        isIdle = true;
        if (satelliteEl) satelliteEl.classList.add('standby');
      }, 2500);

      // Determine hover state via event delegation
      const hoverType = checkHoverTarget(e.target);
      if (hoverType !== currentHoverType) {
        currentHoverType = hoverType;
        updateCursorClasses();
      }
    }, { passive: true });

    // Click Shockwave Feedback
    window.addEventListener('mousedown', (e) => {
      if (!enabled) return;
      spawnClickShockwave(e.clientX, e.clientY);

      // Trigger Leaflet Map-Space Sonar if clicking inside map
      if (e.target && e.target.closest && e.target.closest('.leaflet-container') && window.map && window.L) {
        triggerLeafletMapSonar(e);
      }
    });

    // Alt + C toggle shortcut
    window.addEventListener('keydown', (e) => {
      if (e.altKey && (e.key === 'c' || e.key === 'C')) {
        toggleTacticalCursor();
      }
    });
  }

  function updateCursorClasses() {
    if (!satelliteEl) return;
    satelliteEl.classList.remove('locked', 'threat', 'map-recon', 'text-mode');

    if (currentHoverType === 'locked') {
      satelliteEl.classList.add('locked');
      if (window.playTacticalLockChirp) window.playTacticalLockChirp();
    } else if (currentHoverType === 'threat') {
      satelliteEl.classList.add('locked', 'threat');
      if (window.playTacticalLockChirp) window.playTacticalLockChirp();
    } else if (currentHoverType === 'map') {
      satelliteEl.classList.add('map-recon');
    } else if (currentHoverType === 'text') {
      satelliteEl.classList.add('text-mode');
    }
  }

  // Click Feedback: Expanding Shockwave + Chromatic Aberration
  function spawnClickShockwave(x, y) {
    if (!rootEl) return;

    // Shockwave ring
    const wave = document.createElement('div');
    wave.className = 'cursor-shockwave';
    wave.style.left = `${x}px`;
    wave.style.top = `${y}px`;
    wave.style.width = '30px';
    wave.style.height = '30px';
    rootEl.appendChild(wave);

    // Chromatic burst
    const burst = document.createElement('div');
    burst.className = 'chromatic-flash';
    burst.style.left = `${x}px`;
    burst.style.top = `${y}px`;
    rootEl.appendChild(burst);

    setTimeout(() => {
      wave.remove();
      burst.remove();
    }, 500);
  }

  // Map Sonar Marker Ping
  function triggerLeafletMapSonar(e) {
    if (!window.map || !window.L) return;
    try {
      const mapContainer = window.map.getContainer();
      const rect = mapContainer.getBoundingClientRect();
      const containerPt = window.L.point(e.clientX - rect.left, e.clientY - rect.top);
      const latlng = window.map.containerPointToLatLng(containerPt);

      const sonarIcon = window.L.divIcon({
        className: 'leaflet-map-sonar-ping',
        html: '<div class="sonar-core"></div><div class="sonar-ring"></div><div class="sonar-ring"></div>',
        iconSize: [40, 40],
        iconAnchor: [20, 20]
      });

      const marker = window.L.marker(latlng, { icon: sonarIcon, interactive: false }).addTo(window.map);
      setTimeout(() => {
        if (window.map && marker) window.map.removeLayer(marker);
      }, 1500);
    } catch (err) {
      // Non-blocking catch
    }
  }

  // Global Fetch Interceptor for Ambient Loading Halo
  function hookFetch() {
    const originalFetch = window.fetch;
    window.fetch = async function (...args) {
      activeFetches++;
      if (satelliteEl) satelliteEl.classList.add('fetching');
      try {
        const res = await originalFetch.apply(this, args);
        return res;
      } finally {
        activeFetches = Math.max(0, activeFetches - 1);
        if (activeFetches === 0 && satelliteEl) {
          satelliteEl.classList.remove('fetching');
        }
      }
    };
  }

  // Animation Loop (Physics Lerp Inertia + Heading Angle)
  function renderLoop() {
    if (!enabled) {
      isLoopRunning = false;
      return;
    }
    isLoopRunning = true;

    // Lerp translation with inertia (0.20 spring factor = silky smooth trailing inertia)
    const dx = mouseX - currentX;
    const dy = mouseY - currentY;
    const speed = Math.hypot(dx, dy);

    currentX += dx * 0.20;
    currentY += dy * 0.20;

    // Heading Angle calculation
    if (speed > 1.2) {
      // Nose turns toward movement vector
      const targetRad = Math.atan2(dy, dx);
      targetAngle = (targetRad * 180 / Math.PI) + 90; // +90 aligns SVG forward
    } else {
      // Gentle orbital tumble when resting
      targetAngle += 0.08;
    }

    // Smooth angular rotation interpolation
    let diffAngle = (targetAngle - currentAngle) % 360;
    if (diffAngle > 180) diffAngle -= 360;
    if (diffAngle < -180) diffAngle += 360;
    currentAngle += diffAngle * 0.15;

    // Apply translation & rotation to satellite
    if (satelliteEl) {
      satelliteEl.style.transform = `translate3d(${currentX}px, ${currentY}px, 0)`;

      const svgEl = satelliteEl.querySelector('.satellite-svg');
      if (svgEl) {
        svgEl.style.transform = `rotate(${currentAngle}deg)`;
      }
    }

    requestAnimationFrame(renderLoop);
  }

  function updateCursorToggleButton() {
    const btn = document.getElementById('cursorToggleBtn');
    if (!btn) return;
    if (enabled) {
      btn.innerHTML = `<svg class="ui-icon icon-cyan" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M5 3v4"/><path d="M19 17v4"/><path d="M3 5h4"/><path d="M17 19h4"/><path d="M15 9l5-5"/><path d="M9 15l-5 5"/></svg> <span>TAC CURSOR [ON]</span>`;
      btn.style.borderColor = 'rgba(6, 182, 212, 0.4)';
      btn.style.color = '#38bdf8';
    } else {
      btn.innerHTML = `<svg class="ui-icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="8"/><line x1="12" y1="2" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="22"/><line x1="2" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="22" y2="12"/></svg> <span>TAC CURSOR [OFF]</span>`;
      btn.style.borderColor = 'rgba(100, 116, 139, 0.3)';
      btn.style.color = '#64748b';
    }
  }

  // Toggle Function
  window.toggleTacticalCursor = function () {
    enabled = !enabled;
    localStorage.setItem('canopus_tactical_cursor', enabled ? 'true' : 'false');
    applyCursorState();
  };

  // Launch Engine
  function init() {
    initDOM();
    initListeners();
    hookFetch();
    applyCursorState();

    // Multi-window / tab sync
    window.addEventListener('storage', (e) => {
      if (e.key === 'canopus_tactical_cursor') {
        const shouldBeEnabled = e.newValue !== 'false';
        if (enabled !== shouldBeEnabled) {
          enabled = shouldBeEnabled;
          applyCursorState();
        }
      }
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
