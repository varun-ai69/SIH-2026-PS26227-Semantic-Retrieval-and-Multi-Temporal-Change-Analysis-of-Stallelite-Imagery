/**
 * CANOPUS - ORBITAL MISSION CONTROL AMBIENT EFFECTS & SYSTEM UTILITIES
 * =====================================================================
 * Phase 2: Living Parallax Starfield, Shooting Stars, Terminal Boot Sequence,
 * Matrix Text Scrambler, Number Odometer, and Page Transition Shutter.
 *
 * Additive Only: Zero changes to backend, APIs, or database.
 */

(function () {
  'use strict';

  const prefersReducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // Immediate Live UTC Clock (runs on script parse)
  function updateLiveClockNow() {
    const now = new Date();
    const h = String(now.getUTCHours()).padStart(2, '0');
    const m = String(now.getUTCMinutes()).padStart(2, '0');
    const s = String(now.getUTCSeconds()).padStart(2, '0');
    const timeStr = `${h}:${m}:${s} UTC`;
    document.querySelectorAll('#liveUtcClock, .mission-clock-time').forEach(el => {
      el.textContent = timeStr;
    });
  }
  updateLiveClockNow();
  setInterval(updateLiveClockNow, 1000);
  window.updateLiveMissionClock = updateLiveClockNow;

  // ============================================================
  // 1. LIVING PARALLAX STARFIELD & RADAR SWEEP
  // ============================================================

  let starCanvas = null;
  let starCtx = null;
  let stars = [];
  const STAR_COUNT = 110;
  let shootingStar = null;
  let shootingStarTimer = null;
  let radarAngle = 0;

  class Star {
    constructor(w, h) {
      this.reset(w, h, true);
    }

    reset(w, h, randomY = false) {
      this.x = Math.random() * w;
      this.y = randomY ? Math.random() * h : -5;
      // 3 depth layers: 0 (distant), 1 (mid), 2 (near)
      this.layer = Math.random() < 0.65 ? 0 : (Math.random() < 0.85 ? 1 : 2);
      this.size = this.layer === 0 ? 0.8 : (this.layer === 1 ? 1.4 : 2.2);
      this.speed = (0.05 + this.layer * 0.08) * 0.8;
      this.baseAlpha = 0.2 + this.layer * 0.25;
      this.twinkleSpeed = 0.02 + Math.random() * 0.03;
      this.twinklePhase = Math.random() * Math.PI * 2;
      this.hue = Math.random() < 0.25 ? 195 : (Math.random() < 0.1 ? 215 : 0); // Mostly white/cyan
    }

    update(w, h) {
      this.y += this.speed;
      this.twinklePhase += this.twinkleSpeed;
      if (this.y > h + 5) {
        this.reset(w, h, false);
      }
    }

    draw(ctx) {
      const alpha = Math.max(0.1, this.baseAlpha + Math.sin(this.twinklePhase) * 0.25);
      ctx.save();
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
      if (this.hue === 0) {
        ctx.fillStyle = `rgba(240, 246, 255, ${alpha})`;
      } else {
        ctx.fillStyle = `hsla(${this.hue}, 90%, 75%, ${alpha})`;
        ctx.shadowColor = '#06b6d4';
        ctx.shadowBlur = this.layer === 2 ? 4 : 1;
      }
      ctx.fill();
      ctx.restore();
    }
  }

  class ShootingStar {
    constructor(w, h) {
      this.x = Math.random() * (w * 0.75);
      this.y = Math.random() * (h * 0.35);
      this.length = 90 + Math.random() * 60;
      this.speed = 12 + Math.random() * 8;
      this.angle = (25 + Math.random() * 20) * (Math.PI / 180);
      this.vx = Math.cos(this.angle) * this.speed;
      this.vy = Math.sin(this.angle) * this.speed;
      this.life = 0;
      this.maxLife = 28;
      this.active = true;
    }

    update() {
      this.x += this.vx;
      this.y += this.vy;
      this.life++;
      if (this.life >= this.maxLife) {
        this.active = false;
      }
    }

    draw(ctx) {
      if (!this.active) return;
      const progress = this.life / this.maxLife;
      const alpha = (1 - progress) * 0.85;

      const tailX = this.x - Math.cos(this.angle) * this.length;
      const tailY = this.y - Math.sin(this.angle) * this.length;

      const grad = ctx.createLinearGradient(tailX, tailY, this.x, this.y);
      grad.addColorStop(0, 'rgba(6, 182, 212, 0)');
      grad.addColorStop(0.7, `rgba(56, 189, 248, ${alpha * 0.6})`);
      grad.addColorStop(1, `rgba(255, 255, 255, ${alpha})`);

      ctx.save();
      ctx.beginPath();
      ctx.moveTo(tailX, tailY);
      ctx.lineTo(this.x, this.y);
      ctx.strokeStyle = grad;
      ctx.lineWidth = 1.8;
      ctx.shadowColor = '#38bdf8';
      ctx.shadowBlur = 6;
      ctx.stroke();
      ctx.restore();
    }
  }

  function initStarfield() {
    if (prefersReducedMotion()) return;

    starCanvas = document.createElement('canvas');
    starCanvas.id = 'ambient-starfield-canvas';
    document.body.prepend(starCanvas);
    starCtx = starCanvas.getContext('2d');

    const resize = () => {
      starCanvas.width = window.innerWidth;
      starCanvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize, { passive: true });

    stars = [];
    for (let i = 0; i < STAR_COUNT; i++) {
      stars.push(new Star(starCanvas.width, starCanvas.height));
    }

    // Schedule shooting star easter eggs (every 22 - 42 seconds)
    function scheduleShootingStar() {
      const delay = 22000 + Math.random() * 20000;
      shootingStarTimer = setTimeout(() => {
        if (!shootingStar || !shootingStar.active) {
          shootingStar = new ShootingStar(starCanvas.width, starCanvas.height);
        }
        scheduleShootingStar();
      }, delay);
    }
    scheduleShootingStar();

    // Render loop
    function loop() {
      const w = starCanvas.width;
      const h = starCanvas.height;
      starCtx.clearRect(0, 0, w, h);

      // Draw and update stars
      for (let i = 0; i < stars.length; i++) {
        stars[i].update(w, h);
        stars[i].draw(starCtx);
      }

      // Draw shooting star
      if (shootingStar && shootingStar.active) {
        shootingStar.update();
        shootingStar.draw(starCtx);
      }

      // Draw Distant Corner Radar Sweep (Atmospheric Subtlety)
      radarAngle = (radarAngle + 0.006) % (Math.PI * 2);
      const radarX = w - 80;
      const radarY = h - 80;
      const radarR = 70;

      starCtx.save();
      starCtx.beginPath();
      starCtx.arc(radarX, radarY, radarR, 0, Math.PI * 2);
      starCtx.strokeStyle = 'rgba(6, 182, 212, 0.08)';
      starCtx.lineWidth = 1;
      starCtx.stroke();

      // Sweep line
      starCtx.beginPath();
      starCtx.moveTo(radarX, radarY);
      starCtx.lineTo(radarX + Math.cos(radarAngle) * radarR, radarY + Math.sin(radarAngle) * radarR);
      starCtx.strokeStyle = 'rgba(56, 189, 248, 0.18)';
      starCtx.lineWidth = 1.2;
      starCtx.stroke();
      starCtx.restore();

      requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
  }

  // ============================================================
  // 2. SUBTLE CRT SCANLINE & VIGNETTE LAYER
  // ============================================================

  function initCRTScanlines() {
    if (document.getElementById('crt-scanline-overlay')) return;
    const overlay = document.createElement('div');
    overlay.id = 'crt-scanline-overlay';
    document.body.appendChild(overlay);
  }

  // ============================================================
  // 3. GLOBAL UPLINK BOOT SEQUENCE (Runs once per session)
  // ============================================================

  function initBootSequence() {
    const isBooted = sessionStorage.getItem('canopus_uplink_booted');
    if (isBooted || prefersReducedMotion()) {
      return; // Already booted in this session
    }

    const overlay = document.createElement('div');
    overlay.id = 'boot-sequence-overlay';
    overlay.innerHTML = `
      <div class="boot-terminal-box">
        <div class="boot-terminal-header">
          <span>CANOPUS DEFENSE RECONNAISSANCE MATRIX</span>
          <span>DGIS-DOD-SECURE</span>
        </div>
        <div class="boot-terminal-lines" id="bootLinesContainer"></div>
        <div class="boot-progress-track">
          <div class="boot-progress-fill" id="bootProgressBar"></div>
        </div>
        <div class="boot-skip-prompt">PRESS [ESC] OR CLICK TO BYPASS HANDSHAKE</div>
      </div>
    `;
    document.body.appendChild(overlay);

    const lines = [
      { time: '[00.012]', msg: 'INITIALIZING CANOPUS TACTICAL UPLINK...', ok: 'READY' },
      { time: '[00.180]', msg: 'MOUNTING SENTINEL-2 L2A HARMONIZED PIPELINE...', ok: 'ONLINE' },
      { time: '[00.410]', msg: 'SYNCHRONIZING QDRANT 512-D VECTOR LATENT SPACE...', ok: 'LOCKED' },
      { time: '[00.680]', msg: 'AIR-GAP INTEGRITY CONFIRMED // DGIS CLEARANCE...', ok: 'SOVEREIGN' },
      { time: '[00.920]', msg: 'SATELLITE UPLINK ESTABLISHED. ACCESS GRANTED.', ok: 'OPTIMAL' }
    ];

    const container = overlay.querySelector('#bootLinesContainer');
    const progressBar = overlay.querySelector('#bootProgressBar');
    let currentIndex = 0;

    function renderLine() {
      if (currentIndex >= lines.length) {
        if (progressBar) progressBar.style.width = '100%';
        setTimeout(dismissBoot, 450);
        return;
      }

      const item = lines[currentIndex];
      const lineEl = document.createElement('div');
      lineEl.className = 'boot-line';
      lineEl.innerHTML = `
        <span class="time">${item.time}</span>
        <span class="msg">${item.msg}</span>
        <span class="status-ok">${item.ok}</span>
      `;
      container.appendChild(lineEl);

      // Force layout then animate
      requestAnimationFrame(() => {
        lineEl.classList.add('active');
      });

      const pct = Math.round(((currentIndex + 1) / lines.length) * 100);
      if (progressBar) progressBar.style.width = `${pct}%`;

      currentIndex++;
      setTimeout(renderLine, 220);
    }

    function dismissBoot() {
      sessionStorage.setItem('canopus_uplink_booted', 'true');
      overlay.classList.add('closing');
      setTimeout(() => overlay.remove(), 550);
      window.removeEventListener('keydown', handleKey);
      overlay.removeEventListener('click', dismissBoot);
    }

    function handleKey(e) {
      if (e.key === 'Escape' || e.key === ' ') {
        dismissBoot();
      }
    }

    window.addEventListener('keydown', handleKey);
    overlay.addEventListener('click', dismissBoot);

    setTimeout(renderLine, 120);
  }

  // ============================================================
  // 4. REUSABLE CYPHER TEXT SCRAMBLE / DECRYPT UTILITY
  // ============================================================

  const CYPHER_GLYPHS = '01#%*+-=_<>[]{}XYZ10';

  window.decodeScrambleText = function (element, finalText, durationMs = 450) {
    if (!element || prefersReducedMotion()) {
      if (element) element.innerText = finalText;
      return;
    }

    const startTime = performance.now();
    const length = finalText.length;

    function step(now) {
      const elapsed = now - startTime;
      const progress = Math.min(1, elapsed / durationMs);

      // Number of characters already resolved
      const resolvedCount = Math.floor(progress * length);

      let scrambled = '';
      for (let i = 0; i < length; i++) {
        if (i < resolvedCount) {
          scrambled += finalText[i];
        } else {
          // Random cypher glyph
          scrambled += CYPHER_GLYPHS[Math.floor(Math.random() * CYPHER_GLYPHS.length)];
        }
      }

      element.innerText = scrambled;

      if (progress < 1) {
        requestAnimationFrame(step);
      } else {
        element.innerText = finalText;
      }
    }

    requestAnimationFrame(step);
  };

  // ============================================================
  // 5. REUSABLE NUMBER ODOMETER COUNT-UP UTILITY
  // ============================================================

  window.animateNumberCount = function (element, start, end, durationMs = 600, suffix = '') {
    if (!element || prefersReducedMotion()) {
      if (element) element.innerText = `${end}${suffix}`;
      return;
    }

    const startTime = performance.now();
    const range = end - start;

    function step(now) {
      const elapsed = now - startTime;
      const progress = Math.min(1, elapsed / durationMs);
      // Ease out cubic
      const ease = 1 - Math.pow(1 - progress, 3);
      const current = Math.round(start + range * ease);

      element.innerText = `${current}${suffix}`;

      if (progress < 1) {
        requestAnimationFrame(step);
      } else {
        element.innerText = `${end}${suffix}`;
      }
    }

    requestAnimationFrame(step);
  };

  // ============================================================
  // 6. TACTICAL HUD TOAST NOTIFICATION UTILITY
  // ============================================================

  window.showTacticalToast = function (msg, durationMs = 2800) {
    let container = document.getElementById('tactical-toast-container');
    if (!container) {
      container = document.createElement('div');
      container.id = 'tactical-toast-container';
      document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    toast.className = 'tactical-toast';
    toast.innerHTML = `<span style="color:#06b6d4; font-weight:700; font-family:monospace;">[SYS]</span> <span>${msg}</span>`;
    container.appendChild(toast);

    if (window.playTacticalClick) window.playTacticalClick();

    setTimeout(() => {
      toast.classList.add('fade-out');
      setTimeout(() => toast.remove(), 300);
    }, durationMs);
  };

  // ============================================================
  // 7. CHANNEL-CHANGE SHUTTER ON NAVIGATION CLICKS
  // ============================================================

  function initPageTransitionShutter() {
    const shutter = document.createElement('div');
    shutter.className = 'page-channel-shutter';
    document.body.appendChild(shutter);

    document.addEventListener('click', (e) => {
      const link = e.target.closest('a[href]');
      if (!link) return;

      const href = link.getAttribute('href');
      // Only intercept internal HTML navigation
      if (!href || href.startsWith('#') || href.startsWith('javascript:') || href.startsWith('http')) return;

      // Quick 100ms CRT shutter flash before navigation
      shutter.classList.add('active');
    }, true);
  }

  // ============================================================
  // 8. LIVE UTC MISSION CLOCK
  // ============================================================

  function updateLiveMissionClock() {
    const el = document.getElementById('liveUtcClock');
    if (el) {
      const now = new Date();
      const h = String(now.getUTCHours()).padStart(2, '0');
      const m = String(now.getUTCMinutes()).padStart(2, '0');
      const s = String(now.getUTCSeconds()).padStart(2, '0');
      el.textContent = `${h}:${m}:${s} UTC`;
    }
  }

  // ============================================================
  // INITIALIZATION
  // ============================================================

  function init() {
    initStarfield();
    initCRTScanlines();
    initBootSequence();
    initPageTransitionShutter();
    updateLiveMissionClock();
    setInterval(updateLiveMissionClock, 1000);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
