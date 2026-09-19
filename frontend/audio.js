/**
 * CANOPUS — ORBITAL MISSION CONTROL TACTICAL WEB AUDIO SYNTHESIZER
 * =================================================================
 * Phase 3: Zero-Asset Pure Procedural Audio Synthesizer (Web Audio API)
 *
 * Synthesizes subtle aerospace UI audio using oscillators and filters:
 *  - Target Lock-on Chirp
 *  - Mechanical Relay Button Click
 *  - Deep Leaflet Sonar Ping
 *  - Cypher Decryption Chatter Tick
 *
 * Muted by default. Toggleable via UI button or Alt + M.
 * Additive Only: Zero external audio files or dependencies.
 */

(function () {
  'use strict';

  let audioCtx = null;
  let isEnabled = localStorage.getItem('canopus_audio_enabled') === 'true'; // Default muted
  let masterGain = null;

  function initAudioContext() {
    if (audioCtx) return;
    try {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass) return;
      audioCtx = new AudioContextClass();

      masterGain = audioCtx.createGain();
      masterGain.gain.setValueAtTime(0.25, audioCtx.currentTime);
      masterGain.connect(audioCtx.destination);
    } catch (e) {
      // Non-blocking fallback
    }
  }

  function ensureContextResumed() {
    initAudioContext();
    if (audioCtx && audioCtx.state === 'suspended') {
      audioCtx.resume();
    }
  }

  // ------------------------------------------------------------
  // Sound 1: Micro Target Lock Chirp (35ms dual-tone)
  // ------------------------------------------------------------
  window.playTacticalLockChirp = function () {
    if (!isEnabled || !audioCtx) return;
    try {
      const now = audioCtx.currentTime;
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();

      osc.type = 'sine';
      osc.frequency.setValueAtTime(850, now);
      osc.frequency.exponentialRampToValueAtTime(1400, now + 0.035);

      gain.gain.setValueAtTime(0.08, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.035);

      osc.connect(gain);
      gain.connect(masterGain);

      osc.start(now);
      osc.stop(now + 0.036);
    } catch (e) {}
  };

  // ------------------------------------------------------------
  // Sound 2: Mechanical Relay Button Click (Tactile 25ms blip)
  // ------------------------------------------------------------
  window.playTacticalClick = function () {
    if (!isEnabled || !audioCtx) return;
    try {
      const now = audioCtx.currentTime;
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();

      osc.type = 'triangle';
      osc.frequency.setValueAtTime(160, now);
      osc.frequency.exponentialRampToValueAtTime(40, now + 0.025);

      gain.gain.setValueAtTime(0.12, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.025);

      osc.connect(gain);
      gain.connect(masterGain);

      osc.start(now);
      osc.stop(now + 0.026);
    } catch (e) {}
  };

  // ------------------------------------------------------------
  // Sound 3: Deep Leaflet Map Sonar Ping (Submarine / Deep-Space)
  // ------------------------------------------------------------
  window.playTacticalSonar = function () {
    if (!isEnabled || !audioCtx) return;
    try {
      const now = audioCtx.currentTime;
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();

      osc.type = 'sine';
      osc.frequency.setValueAtTime(440, now);
      osc.frequency.exponentialRampToValueAtTime(180, now + 0.65);

      gain.gain.setValueAtTime(0.14, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.65);

      osc.connect(gain);
      gain.connect(masterGain);

      osc.start(now);
      osc.stop(now + 0.66);
    } catch (e) {}
  };

  // ------------------------------------------------------------
  // Sound 4: Cypher Decryption Chatter Tick
  // ------------------------------------------------------------
  window.playTacticalCypherTick = function () {
    if (!isEnabled || !audioCtx) return;
    try {
      const now = audioCtx.currentTime;
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();

      osc.type = 'square';
      osc.frequency.setValueAtTime(500 + Math.random() * 300, now);

      gain.gain.setValueAtTime(0.02, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.015);

      osc.connect(gain);
      gain.connect(masterGain);

      osc.start(now);
      osc.stop(now + 0.016);
    } catch (e) {}
  };

  // ------------------------------------------------------------
  // Global Audio Toggle & UI Binding
  // ------------------------------------------------------------
  window.toggleTacticalAudio = function () {
    ensureContextResumed();
    isEnabled = !isEnabled;
    localStorage.setItem('canopus_audio_enabled', isEnabled ? 'true' : 'false');
    updateAudioToggleButton();
    if (isEnabled) {
      window.playTacticalClick();
    }
  };

  // Unified tacticalAudio interface used across views
  window.tacticalAudio = {
    playTacticalClick: () => window.playTacticalClick(),
    playLockOnChirp: () => window.playTacticalLockChirp(),
    playTargetLock: () => window.playTacticalLockChirp(),
    playSonarPing: () => window.playTacticalSonar(),
    playCypherTick: () => window.playTacticalCypherTick(),
    toggleAudio: () => window.toggleTacticalAudio()
  };

  function updateAudioToggleButton() {
    const btn = document.getElementById('audioToggleBtn');
    if (!btn) return;
    if (isEnabled) {
      btn.innerHTML = `<svg class="ui-icon icon-emerald" viewBox="0 0 24 24"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"/></svg> <span>TAC AUDIO [LIVE]</span>`;
      btn.classList.add('audio-active');
      btn.style.color = '#10b981';
      btn.style.borderColor = 'rgba(16, 185, 129, 0.4)';
    } else {
      btn.innerHTML = `<svg class="ui-icon" viewBox="0 0 24 24"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/></svg> <span>TAC AUDIO [MUTE]</span>`;
      btn.classList.remove('audio-active');
      btn.style.color = '#64748b';
      btn.style.borderColor = 'rgba(100, 116, 139, 0.3)';
    }
  }

  // Automatically hook clicks on interactive buttons
  function attachEventHooks() {
    // Keyboard Alt+M shortcut
    window.addEventListener('keydown', (e) => {
      if (e.altKey && (e.key === 'm' || e.key === 'M')) {
        window.toggleTacticalAudio();
      }
    });

    // Button clicks
    document.addEventListener('click', (e) => {
      if (e.target.closest('button, .nav-item, [onclick], .aoi-card')) {
        ensureContextResumed();
        window.playTacticalClick();
      }
    });

    // Map clicks trigger Sonar
    document.addEventListener('click', (e) => {
      if (e.target.closest('.leaflet-container')) {
        ensureContextResumed();
        window.playTacticalSonar();
      }
    });
  }

  function init() {
    attachEventHooks();
    updateAudioToggleButton();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
