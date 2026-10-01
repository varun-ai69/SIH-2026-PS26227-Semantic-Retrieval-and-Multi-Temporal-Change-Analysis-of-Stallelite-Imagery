/**
 * Canopus Frontend Application Logic
 * ====================================
 * Manages:
 *  1. Top navigation switching across pages (Map, Semantic Retrieval, Change Detection, Clustering, Review Tool)
 *  2. Leaflet world map initialization with high-resolution satellite layers
 *  3. Ingestion coverage polygon overlay from PostgreSQL database
 *  4. Interactive Leaflet Draw polygon creation with auto-closure
 *  5. Multi-year timeline selection & dynamic ingestion pipeline execution
 */

// Configurable backend base: uses custom URL if set, or current origin (works with Netlify proxy or direct backend)
const API_BASE = (typeof window !== 'undefined' && (
  window.CANOPUS_BACKEND_URL ||
  localStorage.getItem('canopus_backend_url') ||
  window.location.origin
)).replace(/\/+$/, '');
window.API_BASE = API_BASE;

// Global Map State
let map = null;
let currentBasemap = null;
let basemapLayers = {};
let coverageLayerGroup = null;
let discoveryMapLayerGroup = null;
let drawControl = null;
let currentDrawLayer = null;
let coverageData = null;

// ============================================================
// 1. PAGE NAVIGATION SWITCHER
// ============================================================

window.switchPage = function (pageId) {
  // Update Nav Buttons
  document.querySelectorAll('.nav-item').forEach(btn => btn.classList.remove('active'));
  const activeBtn = document.getElementById(`nav-${pageId}`);
  if (activeBtn) activeBtn.classList.add('active');

  // Update View Panels
  document.querySelectorAll('.view-panel').forEach(panel => panel.classList.remove('active'));
  const activePanel = document.getElementById(`view-${pageId}`);
  if (activePanel) {
    activePanel.classList.add('active');
    if (pageId === 'map' && map) {
      setTimeout(() => map.invalidateSize(), 200);
    }
  }
};

// ============================================================
// 2. LEAFLET MAP & HIGH-RES SATELLITE LAYERS
// ============================================================

function initMap() {
  // Define High-Resolution Basemaps
  basemapLayers = {
    satellite: L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      attribution: 'Esri World Imagery, Maxar, Earthstar Geographics',
      maxZoom: 19
    }),
    hybrid: L.tileLayer('https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', {
      attribution: 'Google Satellite Hybrid Imagery',
      maxZoom: 20
    }),
    dark: L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; CartoDB & OpenStreetMap',
      subdomains: 'abcd',
      maxZoom: 19
    }),
    osm: L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors',
      maxZoom: 19
    })
  };

  // Initialize Map with whole world view centered on India/Central Asia
  map = L.map('leafletMap', {
    center: [23.5, 75.0],
    zoom: 5,
    minZoom: 2,
    maxZoom: 20,
    worldCopyJump: true,
    zoomControl: true
  });

  // Add default satellite layer
  currentBasemap = basemapLayers.satellite;
  currentBasemap.addTo(map);

  // Initialize Coverage Layer Group
  coverageLayerGroup = L.featureGroup().addTo(map);

  // Setup Leaflet Draw Handler
  setupLeafletDraw();

  // Setup Live Mouse Coordinates HUD
  setupMouseCoordsHud();

  // Load Database Ingestion Coverage Polygons
  loadCoverageRegions();
}

// ============================================================
// 2.5 COORDINATE TELEPORT & MOUSE HUD SYSTEM
// ============================================================
let activeTeleportMarker = null;
let activeTeleportBox = null;

function setupMouseCoordsHud() {
  if (!map) return;
  const hudLatLon = document.getElementById('hudLatLon');
  const hudZoom = document.getElementById('hudZoom');

  map.on('mousemove', function (e) {
    if (hudLatLon) {
      hudLatLon.innerText = `Lat: ${e.latlng.lat.toFixed(4)}° | Lon: ${e.latlng.lng.toFixed(4)}°`;
    }
  });

  map.on('zoomend', function () {
    if (hudZoom) {
      hudZoom.innerText = map.getZoom();
    }
  });
}

window.parseCoordinateInput = function (raw) {
  if (!raw) return null;
  const clean = raw.trim();

  // Check if JSON or bracket array e.g. [minLon, minLat, maxLon, maxLat]
  if (clean.startsWith('[') && clean.endsWith(']')) {
    try {
      const arr = JSON.parse(clean);
      if (Array.isArray(arr) && arr.length === 4) {
        return { type: 'bbox', bbox: arr.map(Number) };
      }
      if (Array.isArray(arr) && arr.length === 2) {
        return { type: 'point', lat: Number(arr[0]), lon: Number(arr[1]) };
      }
    } catch (e) { }
  }

  // Comma or space separated numbers
  const nums = clean.replace(/[^\d.\-+,]/g, ' ')
    .split(/[\s,]+/)
    .filter(Boolean)
    .map(Number);

  if (nums.length === 4) {
    // BBOX: [minLon, minLat, maxLon, maxLat]
    return { type: 'bbox', bbox: nums };
  }

  if (nums.length >= 2) {
    let lat = nums[0];
    let lon = nums[1];

    // Smart detection: if first coordinate is > 90 or < -90, it's [lon, lat], swap them
    if ((lat > 90 || lat < -90) && (lon >= -90 && lon <= 90)) {
      const temp = lat;
      lat = lon;
      lon = temp;
    }

    return { type: 'point', lat, lon };
  }

  return null;
};

window.executeTeleport = function () {
  const topInput = document.getElementById('teleportCoordInputTop');
  const sideInput = document.getElementById('teleportCoordInput');
  const val = (topInput && topInput.value.trim()) || (sideInput && sideInput.value.trim());

  if (!val) {
    alert('Please enter coordinates (e.g. 34.455, 73.315 or [minLat, minLon, maxLat, maxLon])');
    return;
  }

  if (topInput) topInput.value = val;
  if (sideInput) sideInput.value = val;

  const parsed = parseCoordinateInput(val);
  if (!parsed) {
    alert('Invalid coordinate format.\n\nSupported examples:\n• "28.6139, 77.2090" (Lat, Lon)\n• "34.455 73.315"\n• "28°36\'50\\"N, 77°12\'32\\"E" (DMS)\n• "77.10, 28.50, 77.30, 28.70" (BBox)');
    return;
  }

  if (parsed.type === 'point') {
    teleportToPoint(parsed.lat, parsed.lon, 15, `Target Point: ${parsed.lat.toFixed(4)}°, ${parsed.lon.toFixed(4)}°`);
  } else if (parsed.type === 'bbox') {
    teleportToBbox(parsed.bbox);
  }
};

window.executeFastTravel = window.executeTeleport;

window.teleportToPreset = function (presetStr) {
  if (!presetStr) return;
  const parts = presetStr.split(',');
  if (parts.length < 3) return;

  const lat = parseFloat(parts[0]);
  const lon = parseFloat(parts[1]);
  const zoom = parseInt(parts[2], 10) || 15;
  const label = parts.slice(3).join(',') || `Preset (${lat.toFixed(4)}, ${lon.toFixed(4)})`;

  const coordText = `${lat.toFixed(4)}, ${lon.toFixed(4)}`;
  const topInput = document.getElementById('teleportCoordInputTop');
  const sideInput = document.getElementById('teleportCoordInput');
  if (topInput) topInput.value = coordText;
  if (sideInput) sideInput.value = coordText;

  const topSelect = document.getElementById('teleportPresetSelectTop');
  const sideSelect = document.getElementById('teleportPresetSelect');
  if (topSelect) topSelect.value = presetStr;
  if (sideSelect) sideSelect.value = presetStr;

  teleportToPoint(lat, lon, zoom, label);
};

window.teleportToPoint = function (lat, lon, zoom = 15, label = '') {
  if (!map) return;

  // Hyperspace warp animation on map container & tactical toast
  try {
    const mapContainer = map.getContainer();
    if (mapContainer) {
      mapContainer.classList.add('hyperspace-warp');
      setTimeout(() => mapContainer.classList.remove('hyperspace-warp'), 450);
    }
    if (window.showTacticalToast) {
      window.showTacticalToast(`COORDINATES LOCKED // ORBIT TRANSIT: [${lat.toFixed(4)}°, ${lon.toFixed(4)}°]`);
    }
    window.tacticalAudio?.playLockOnChirp?.();
  } catch (err) { }

  // Fly animation
  map.flyTo([lat, lon], zoom, {
    animate: true,
    duration: 1.8
  });

  // Remove previous teleport marker & box
  if (activeTeleportMarker) map.removeLayer(activeTeleportMarker);
  if (activeTeleportBox) map.removeLayer(activeTeleportBox);

  // Custom pulsing beacon icon
  const pulseIcon = L.divIcon({
    className: 'teleport-pulse-icon',
    html: '<div class="teleport-pulse-dot"></div><div class="teleport-pulse-ring"></div>',
    iconSize: [24, 24],
    iconAnchor: [12, 12]
  });

  activeTeleportMarker = L.marker([lat, lon], { icon: pulseIcon }).addTo(map);

  const popupContent = `
    <div style="font-family: 'Inter', sans-serif; font-size: 12px; color: #fff; min-width: 200px; padding: 4px;">
      <div style="font-weight: 700; color: #38bdf8; margin-bottom: 4px; font-size: 13px;">${label || 'Target Location'}</div>
      <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #94a3b8; margin-bottom: 10px;">
        Lat: ${lat.toFixed(5)}°<br>Lon: ${lon.toFixed(5)}°
      </div>
      <div style="display: flex; flex-direction: column; gap: 6px;">
        <button onclick="prepareIngestFromPoint(${lat}, ${lon})" style="background: #06b6d4; color: #000; border: none; padding: 6px 10px; border-radius: 4px; font-weight: 600; cursor: pointer; font-size: 11px;">
          Ingest This Location (5km AOI)
        </button>
        <button onclick="clearTeleportMarker()" style="background: rgba(255,255,255,0.1); color: #cbd5e1; border: 1px solid rgba(255,255,255,0.2); padding: 5px 10px; border-radius: 4px; cursor: pointer; font-size: 11px;">
          Clear Marker
        </button>
      </div>
    </div>
  `;

  activeTeleportMarker.bindPopup(popupContent).openPopup();
};

window.teleportToBbox = function (bbox) {
  if (!map || !bbox || bbox.length !== 4) return;
  const [minLon, minLat, maxLon, maxLat] = bbox;

  // Fly to BBOX bounds
  map.flyToBounds([[minLat, minLon], [maxLat, maxLon]], {
    animate: true,
    duration: 1.8,
    padding: [40, 40]
  });

  if (activeTeleportMarker) map.removeLayer(activeTeleportMarker);
  if (activeTeleportBox) map.removeLayer(activeTeleportBox);

  // Draw highlight polygon for the BBOX
  activeTeleportBox = L.rectangle([[minLat, minLon], [maxLat, maxLon]], {
    color: '#06b6d4',
    weight: 2,
    dashArray: '6, 6',
    fillColor: '#06b6d4',
    fillOpacity: 0.15
  }).addTo(map);

  activeTeleportBox.bindPopup(`
    <div style="font-family: 'Inter', sans-serif; font-size: 12px; color: #fff; min-width: 220px; padding: 4px;">
      <div style="font-weight: 700; color: #38bdf8; margin-bottom: 4px;">Target Bounding Box</div>
      <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #94a3b8; margin-bottom: 8px;">
        SW: ${minLat.toFixed(4)}°, ${minLon.toFixed(4)}°<br>
        NE: ${maxLat.toFixed(4)}°, ${maxLon.toFixed(4)}°
      </div>
      <button onclick="prepareIngestFromBbox(${minLon}, ${minLat}, ${maxLon}, ${maxLat})" style="background: #06b6d4; color: #000; border: none; padding: 6px 10px; border-radius: 4px; font-weight: 600; cursor: pointer; font-size: 11px; width: 100%;">
        Ingest This Bounding Box
      </button>
    </div>
  `).openPopup();
};

window.clearTeleportMarker = function () {
  if (activeTeleportMarker && map) {
    map.removeLayer(activeTeleportMarker);
    activeTeleportMarker = null;
  }
  if (activeTeleportBox && map) {
    map.removeLayer(activeTeleportBox);
    activeTeleportBox = null;
  }
};

window.prepareIngestFromPoint = function (lat, lon) {
  // Generate ~5km x 5km box around point (approx 0.045 deg)
  const d = 0.0225;
  prepareIngestFromBbox(lon - d, lat - d, lon + d, lat + d);
};

window.prepareIngestFromBbox = function (minLon, minLat, maxLon, maxLat) {
  const geojsonPoly = {
    type: "Polygon",
    coordinates: [[
      [minLon, minLat],
      [maxLon, minLat],
      [maxLon, maxLat],
      [minLon, maxLat],
      [minLon, minLat]
    ]]
  };

  const geoInput = document.getElementById('geojsonInput');
  if (geoInput) {
    geoInput.value = JSON.stringify(geojsonPoly, null, 2);
  }

  const centerLat = (minLat + maxLat) / 2;
  const centerLon = (minLon + maxLon) / 2;
  const regionIdInput = document.getElementById('aoiRegionId');
  const regionNameInput = document.getElementById('aoiRegionName');
  if (regionIdInput) {
    regionIdInput.value = `sector_${centerLat.toFixed(2)}_${centerLon.toFixed(2)}`;
  }
  if (regionNameInput) {
    regionNameInput.value = `Sector (${centerLat.toFixed(2)}°N, ${centerLon.toFixed(2)}°E)`;
  }

  openIngestModal();
};

window.switchBasemap = function (type) {
  if (!basemapLayers[type] || !map) return;

  // Switch Active Button Style
  document.querySelectorAll('.basemap-option').forEach(btn => btn.classList.remove('active'));
  const activeBtn = document.getElementById(`bm-${type}`);
  if (activeBtn) activeBtn.classList.add('active');

  // Replace Layer
  map.removeLayer(currentBasemap);
  currentBasemap = basemapLayers[type];
  currentBasemap.addTo(map);
  window.tacticalAudio?.playTacticalClick?.();
};

// ============================================================
// 3. DATABASE INGESTION COVERAGE OVERLAY
// ============================================================

window.loadCoverageRegions = async function () {
  try {
    const res = await fetch(`${API_BASE}/api/v1/coverage`);
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    coverageData = await res.json();

    // Clear previous coverage
    coverageLayerGroup.clearLayers();

    // Populate Region Dropdown
    const select = document.getElementById('regionSelect');
    if (select) {
      select.innerHTML = '<option value="all">[ALL] Ingested Sectors</option>';
    }

    // Update Header Tile Counter
    const archiveCountEl = document.getElementById('archiveTileCount');
    if (archiveCountEl) {
      const targetCount = coverageData.total_tiles || 0;
      if (window.animateNumberCount) {
        window.animateNumberCount(archiveCountEl, 0, targetCount, 850, ' TILES ONLINE');
      } else {
        archiveCountEl.innerText = `${targetCount} TILES ONLINE`;
      }
    }

    if (!coverageData.features || coverageData.features.length === 0) {
      return;
    }

    // Colors for distinct regions
    const sectorColors = ['#06b6d4', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6', '#3b82f6'];

    coverageData.features.forEach((feat, idx) => {
      const color = sectorColors[idx % sectorColors.length];
      const props = feat.properties;

      // Add to Region Select Dropdown
      if (select) {
        const opt = document.createElement('option');
        opt.value = props.region_id;
        opt.innerText = `[${props.region_id || 'AOI'}] ${props.region_name} (${props.tile_count} Tiles)`;
        select.appendChild(opt);
      }

      // Render GeoJSON Polygon with Tactical Laser Border
      const geoLayer = L.geoJSON(feat.geometry, {
        style: {
          className: 'tactical-laser-polygon',
          color: color,
          weight: 2.5,
          opacity: 0.95,
          fillColor: color,
          fillOpacity: 0.18,
          dashArray: '6, 6'
        }
      });

      // Hover Tooltip
      geoLayer.bindTooltip(`
        <strong>${props.region_name}</strong><br/>
        <span style="font-family: monospace; font-size: 11px;">Tiles: ${props.tile_count} | Status: ${props.status}</span>
      `, { sticky: true });

      // Click Popup
      geoLayer.bindPopup(`
        <div class="popup-title"><span class="status-pill status-confirmed">SECTOR</span> ${props.region_name}</div>
        <div class="popup-stat"><strong>Region ID:</strong> ${props.region_id}</div>
        <div class="popup-stat"><strong>Embedded Tiles:</strong> ${props.tile_count}</div>
        <div class="popup-stat"><strong>Database Status:</strong> <span style="color: #34d399; font-weight: 600;">${props.status.toUpperCase()}</span></div>
        <div class="popup-stat"><strong>Updated:</strong> ${props.last_updated ? props.last_updated.substring(0, 10) : 'N/A'}</div>
      `);

      coverageLayerGroup.addLayer(geoLayer);
    });

    // Fit Map to Ingested Regions on initial load
    if (coverageLayerGroup.getLayers().length > 0) {
      map.fitBounds(coverageLayerGroup.getBounds(), { padding: [40, 40], maxZoom: 13 });
    }

  } catch (err) {
    console.error("Failed to load coverage regions:", err);
  }
};

window.onSelectRegion = function (regionId) {
  if (!coverageData || !coverageData.features) return;

  if (regionId === 'all') {
    fitAllAois();
    return;
  }

  const targetFeat = coverageData.features.find(f => f.properties.region_id === regionId);
  if (targetFeat) {
    const tempLayer = L.geoJSON(targetFeat.geometry);
    map.fitBounds(tempLayer.getBounds(), { padding: [60, 60], maxZoom: 14 });
  }
};

window.fitAllAois = function () {
  if (coverageLayerGroup && coverageLayerGroup.getLayers().length > 0) {
    map.fitBounds(coverageLayerGroup.getBounds(), { padding: [40, 40], maxZoom: 13 });
  }
};

// ============================================================
// 4. LEAFLET DRAW INTERACTIVE POLYGON CREATION
// ============================================================

function setupLeafletDraw() {
  map.on(L.Draw.Event.CREATED, function (e) {
    const layer = e.layer;
    if (currentDrawLayer) {
      map.removeLayer(currentDrawLayer);
    }
    currentDrawLayer = layer;
    map.addLayer(currentDrawLayer);

    const geojson = layer.toGeoJSON();

    // Auto-populate GeoJSON Input in Ingest Modal
    const geoInput = document.getElementById('geojsonInput');
    if (geoInput) {
      geoInput.value = JSON.stringify(geojson.geometry, null, 2);
    }

    // Auto-generate suggested Region ID
    const center = layer.getBounds().getCenter();
    const regionIdInput = document.getElementById('aoiRegionId');
    const regionNameInput = document.getElementById('aoiRegionName');
    if (regionIdInput && !regionIdInput.value) {
      regionIdInput.value = `region_${center.lat.toFixed(2)}_${center.lng.toFixed(2)}`;
    }
    if (regionNameInput && !regionNameInput.value) {
      regionNameInput.value = `Sector (${center.lat.toFixed(2)}°N, ${center.lng.toFixed(2)}°E)`;
    }

    // Open Ingest Modal for Confirmation
    openIngestModal();
  });
}

window.startLeafletDraw = function () {
  closeIngestModal();
  const polygonDrawer = new L.Draw.Polygon(map, {
    shapeOptions: {
      color: '#06b6d4',
      weight: 3,
      fillColor: '#06b6d4',
      fillOpacity: 0.25
    },
    allowIntersection: false,
    showArea: true
  });
  polygonDrawer.enable();
};

window.triggerDrawFromModal = function () {
  closeIngestModal();
  startLeafletDraw();
};

// ============================================================
// 5. INGESTION MODAL & TIMELINE CONTROLS
// ============================================================

let currentIngestMode = 'aoi'; // 'aoi' or 'file'
let currentIngestionSensor = 'sentinel2'; // 'sentinel2' or 'maxar'
let selectedYears = 2;
let selectedGeoTiffFiles = [];

window.setIngestionSensor = function (sensor) {
  currentIngestionSensor = sensor;
  const cardSentinel = document.getElementById('sensorCardSentinel');
  const cardMaxar = document.getElementById('sensorCardMaxar');
  const sentinelGroup = document.getElementById('sentinelTimelineGroup');
  const maxarGroup = document.getElementById('maxarTimelineGroup');
  const descEl = document.getElementById('aoiDescriptionText');
  const bucketGroup = document.getElementById('bucketSelectGroup');

  if (sensor === 'maxar') {
    if (cardSentinel) {
      cardSentinel.style.border = '2px solid var(--border-color)';
      cardSentinel.style.background = 'rgba(255,255,255,0.02)';
    }
    if (cardMaxar) {
      cardMaxar.style.border = '2px solid var(--accent-cyan)';
      cardMaxar.style.background = 'rgba(6,182,212,0.15)';
    }
    if (sentinelGroup) sentinelGroup.style.display = 'none';
    if (maxarGroup) maxarGroup.style.display = 'block';
    if (bucketGroup) bucketGroup.style.display = 'none';
    if (descEl) {
      descEl.innerHTML = `<strong>Maxar High-Resolution Optical Ingestion:</strong> Fetches sub-meter orthorectified imagery from the Maxar/Esri Wayback archive (2020-2026), slices 512x512 georeferenced GeoTIFFs, computes the VARI index, and stores embeddings in dedicated Qdrant collection <code>maxar_tile_embeddings</code>.`;
    }
    setMaxarEpochPreset('2020_2026');
  } else {
    if (cardSentinel) {
      cardSentinel.style.border = '2px solid var(--accent-blue)';
      cardSentinel.style.background = 'rgba(56,189,248,0.12)';
    }
    if (cardMaxar) {
      cardMaxar.style.border = '2px solid var(--border-color)';
      cardMaxar.style.background = 'rgba(255,255,255,0.02)';
    }
    if (sentinelGroup) sentinelGroup.style.display = 'block';
    if (maxarGroup) maxarGroup.style.display = 'none';
    if (bucketGroup) bucketGroup.style.display = 'block';
    if (descEl) {
      descEl.innerHTML = `Define an AOI polygon and choose the historical timeline (e.g. 1-10 years). The pipeline fetches all covering Sentinel-2 scenes, cleans cloud/shadow masks, normalizes, slices 512x512 tiles, computes indices (NDVI/NDWI/NDBI), and embeds into Qdrant & PostgreSQL.`;
    }
    setTimelineYears(2);
  }
};

window.setMaxarEpochPreset = function (preset) {
  document.querySelectorAll('#maxarTimelineGroup .timeline-chip').forEach(btn => btn.classList.remove('active'));
  const activeBtn = document.getElementById(
    preset === '2020_2026' ? 'chipMaxar2020_2026' :
      preset === '2022_2026' ? 'chipMaxar2022_2026' :
        preset === 'all' ? 'chipMaxarAll' : 'chipMaxarCustom'
  );
  if (activeBtn) activeBtn.classList.add('active');

  const dateFrom = document.getElementById('aoiDateFrom');
  const dateTo = document.getElementById('aoiDateTo');

  if (preset === '2020_2026') {
    if (dateFrom) dateFrom.value = '2020-08-12';
    if (dateTo) dateTo.value = '2026-08-05';
  } else if (preset === '2022_2026') {
    if (dateFrom) dateFrom.value = '2022-10-19';
    if (dateTo) dateTo.value = '2026-08-05';
  } else if (preset === 'all') {
    if (dateFrom) dateFrom.value = '2020-08-12';
    if (dateTo) dateTo.value = '2026-08-05';
  }
};

window.selectSensorFilter = function (val) {
  const pSent = document.getElementById('pillSensorSentinel');
  const pMax = document.getElementById('pillSensorMaxar');
  const pAll = document.getElementById('pillSensorAll');

  if (pSent) {
    pSent.style.background = (val === 'Sentinel-2') ? 'rgba(6,182,212,0.2)' : 'rgba(255,255,255,0.03)';
    pSent.style.borderColor = (val === 'Sentinel-2') ? 'var(--accent-cyan)' : 'var(--border-color)';
    pSent.style.color = (val === 'Sentinel-2') ? '#fff' : 'var(--text-secondary)';
    pSent.style.boxShadow = (val === 'Sentinel-2') ? '0 0 10px rgba(6,182,212,0.2)' : 'none';
  }
  if (pMax) {
    pMax.style.background = (val === 'Maxar') ? 'rgba(6,182,212,0.2)' : 'rgba(255,255,255,0.03)';
    pMax.style.borderColor = (val === 'Maxar') ? 'var(--accent-cyan)' : 'var(--border-color)';
    pMax.style.color = (val === 'Maxar') ? '#fff' : 'var(--text-secondary)';
    pMax.style.boxShadow = (val === 'Maxar') ? '0 0 10px rgba(6,182,212,0.2)' : 'none';
  }
  if (pAll) {
    pAll.style.background = (val === '') ? 'rgba(6,182,212,0.2)' : 'rgba(255,255,255,0.03)';
    pAll.style.borderColor = (val === '') ? 'var(--accent-cyan)' : 'var(--border-color)';
    pAll.style.color = (val === '') ? '#fff' : 'var(--text-secondary)';
    pAll.style.boxShadow = (val === '') ? '0 0 10px rgba(6,182,212,0.2)' : 'none';
  }

  const filterSensor = document.getElementById('filterSensor');
  if (filterSensor) filterSensor.value = val;
  const directSelects = document.querySelectorAll('#directSensorSelect');
  directSelects.forEach(s => s.value = val);
};

window.onDirectSensorChange = function (val) {
  selectSensorFilter(val);
};

window.syncDirectSensor = function (val) {
  selectSensorFilter(val || '');
};

window.handleGeoTiffFilesSelected = function (files) {
  if (!files || files.length === 0) return;
  selectedGeoTiffFiles = Array.from(files);
  const summaryEl = document.getElementById('selectedFilesSummary');
  if (summaryEl) {
    summaryEl.style.display = 'block';
    const names = selectedGeoTiffFiles.map(f => f.name).join(', ');
    summaryEl.innerHTML = `<span class="status-pill status-confirmed">[READY]</span> <strong>${selectedGeoTiffFiles.length} file(s) selected:</strong> ${names}`;
  }
};

window.openIngestModal = function () {
  const modal = document.getElementById('ingestModal');
  if (!modal) return;
  modal.classList.add('open');

  // Show air-gapped alert banner if currently in offline mode
  const isOffline = localStorage.getItem('canopus_offline_mode') === 'true';
  const existingBanner = document.getElementById('ingestOfflineNotice');
  if (isOffline) {
    if (!existingBanner) {
      const banner = document.createElement('div');
      banner.id = 'ingestOfflineNotice';
      banner.style.cssText = 'background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.4); border-radius: 8px; padding: 12px 16px; margin: 0 0 16px 0; display: flex; align-items: center; justify-content: space-between; gap: 12px; font-size: 12px; color: #fbbf24;';
      banner.innerHTML = `
        <div>
          <strong>AIR-GAPPED OFFLINE MODE:</strong>
          <span>External STAC & Maxar satellite downloading is paused. Click Go Online or use Direct GeoTIFF tab.</span>
        </div>
        <button onclick="window.setNetworkMode(false); const n = document.getElementById('ingestOfflineNotice'); if (n) n.remove();" style="background: #f59e0b; color: #000; border: none; border-radius: 6px; padding: 6px 12px; font-size: 11px; font-weight: 700; cursor: pointer; white-space: nowrap;">
          Go Online Now
        </button>
      `;
      const body = modal.querySelector('.modal-body') || modal;
      body.insertBefore(banner, body.firstChild);
    }
  } else if (existingBanner) {
    existingBanner.remove();
  }
};

window.closeIngestModal = function () {
  document.getElementById('ingestModal').classList.remove('open');
};

window.switchIngestMode = function (mode) {
  currentIngestMode = mode;
  document.getElementById('tabAoiBtn').classList.toggle('active', mode === 'aoi');
  document.getElementById('tabFileBtn').classList.toggle('active', mode === 'file');
  document.getElementById('aoiFormSection').style.display = (mode === 'aoi') ? 'flex' : 'none';
  document.getElementById('fileFormSection').style.display = (mode === 'file') ? 'flex' : 'none';
};

window.setTimelineYears = function (years) {
  selectedYears = years;
  document.querySelectorAll('#sentinelTimelineGroup .timeline-chip').forEach(btn => btn.classList.remove('active'));
  if (event && event.target && event.target.classList.contains('timeline-chip')) {
    event.target.classList.add('active');
  }

  const today = new Date();
  const startYear = today.getFullYear() - years;
  const fromDate = `${startYear}-01-01`;
  const toDate = today.toISOString().split('T')[0];

  document.getElementById('aoiDateFrom').value = fromDate;
  document.getElementById('aoiDateTo').value = toDate;

  // Auto-adjust suggested time buckets based on years
  const bucketSelect = document.getElementById('aoiBuckets');
  if (bucketSelect) {
    if (years === 1) bucketSelect.value = "2";
    else if (years <= 3) bucketSelect.value = "2";
    else if (years <= 5) bucketSelect.value = "4";
    else bucketSelect.value = "10";
  }
};

window.setCustomTimeline = function () {
  document.querySelectorAll('#sentinelTimelineGroup .timeline-chip').forEach(btn => btn.classList.remove('active'));
  if (event && event.target) event.target.classList.add('active');
};

// ============================================================
// 6. PIPELINE EXECUTION & REAL-TIME PROGRESS BAR
// ============================================================

let progressInterval = null;

function animateProgressBar() {
  const card = document.getElementById('ingestProgressCard');
  const bar = document.getElementById('ingestProgressBar');
  const label = document.getElementById('ingestStepLabel');
  const timer = document.getElementById('ingestTimer');
  const subtext = document.getElementById('ingestSubtext');

  card.style.display = 'flex';
  bar.style.width = '10%';

  const isMaxar = currentIngestionSensor === 'maxar';
  label.innerText = isMaxar
    ? '[PHASE 1/4] Connecting to Maxar Wayback Archive & Selecting Epochs...'
    : '[PHASE 1/5] Querying STAC Catalog & Selecting Granules...';
  subtext.innerText = isMaxar
    ? 'Selecting 2020 baseline and contemporary sub-meter orthorectified releases...'
    : 'Scanning AWS Earth Search STAC for cloud-free Sentinel-2 scenes...';

  let seconds = 0;
  timer.innerText = '0s';

  const steps = isMaxar ? [
    { pct: 25, label: '[PHASE 2/4] Streaming High-Res Maxar WMTS Tiles & Stitching...', sub: 'Fetching sub-meter orthorectified image canvas across epochs...' },
    { pct: 50, label: '[PHASE 3/4] Slicing 512x512 Georeferenced Tiles & Computing VARI...', sub: 'Generating EPSG:4326 GeoTIFFs with Visible Atmospherically Resistant Index...' },
    { pct: 75, label: '[PHASE 4/4] RemoteCLIP ViT-B-32 Vector Embeddings & Indexing...', sub: 'Embedding optical features into Qdrant maxar_tile_embeddings & PostgreSQL...' },
    { pct: 90, label: '[FINALIZE] Database Registration & Map Coverage...', sub: 'Writing georeferenced sector polygons to PostgreSQL...' }
  ] : [
    { pct: 25, label: '[PHASE 2/5] Streaming Cloud-Optimized GeoTIFFs (B2, B3, B4, B8, B11)...', sub: 'Reprojecting rasters to EPSG:4326 working canvas...' },
    { pct: 50, label: '[PHASE 3/5] Computing Cloud & Shadow Masks + 2-98% Normalization...', sub: 'Filtering bad pixels and scaling dynamic range across 5 bands...' },
    { pct: 70, label: '[PHASE 4/5] Slicing 512x512 Tiles & Calculating NDVI, NDWI, NDBI...', sub: 'Computing multi-spectral vegetation, water, and built-up indices...' },
    { pct: 90, label: '[PHASE 5/5] Generating RemoteCLIP ViT-B-32 Vector Embeddings...', sub: 'Batch upserting 512-dim vectors into Qdrant and metadata into PostgreSQL...' }
  ];

  let stepIdx = 0;
  progressInterval = setInterval(() => {
    seconds++;
    timer.innerText = `${seconds}s`;

    if (seconds % 4 === 0 && stepIdx < steps.length) {
      bar.style.width = `${steps[stepIdx].pct}%`;
      label.innerText = steps[stepIdx].label;
      subtext.innerText = steps[stepIdx].sub;
      stepIdx++;
    }
  }, 1000);
}

function stopProgressBar(success, message) {
  if (progressInterval) clearInterval(progressInterval);
  const bar = document.getElementById('ingestProgressBar');
  const label = document.getElementById('ingestStepLabel');
  const alertBox = document.getElementById('ingestSuccessAlert');
  const submitBtn = document.getElementById('submitIngestBtn');

  submitBtn.disabled = false;

  if (success) {
    bar.style.width = '100%';
    label.innerText = '[STATUS: OK] Ingestion Pipeline Complete!';
    label.style.color = '#34d399';
    alertBox.style.display = 'block';
    alertBox.innerHTML = `<strong>SUCCESS:</strong> ${message}`;
  } else {
    label.innerText = '[STATUS: ERR] Ingestion Failed';
    label.style.color = '#f43f5e';
    alertBox.style.display = 'block';
    alertBox.style.background = 'rgba(244, 63, 94, 0.15)';
    alertBox.style.borderColor = 'rgba(244, 63, 94, 0.4)';
    alertBox.style.color = '#fb7185';
    alertBox.innerHTML = `<strong>ERROR:</strong> ${message}`;
  }
}

window.startIngestion = async function () {
  const submitBtn = document.getElementById('submitIngestBtn');
  const alertBox = document.getElementById('ingestSuccessAlert');
  alertBox.style.display = 'none';
  submitBtn.disabled = true;

  animateProgressBar();

  try {
    if (currentIngestMode === 'aoi') {
      const isOffline = localStorage.getItem('canopus_offline_mode') === 'true';
      if (isOffline) {
        throw new Error("System is currently in Air-Gapped Offline Mode. External STAC / Sentinel-2 requests are blocked. Click 'GO ONLINE' in the top header to switch to Online Mode, or use the 'Direct GeoTIFF Ingestion' tab for local files.");
      }

      // Entry Point A: AOI Polygon
      const geojsonStr = document.getElementById('geojsonInput').value.trim();
      if (!geojsonStr) {
        throw new Error("Please draw a region on the map or paste GeoJSON polygon coordinates.");
      }

      let parsedGeojson;
      try {
        parsedGeojson = JSON.parse(geojsonStr);
      } catch (e) {
        throw new Error("Invalid GeoJSON JSON syntax. Please verify coordinates format.");
      }

      const dateFromVal = document.getElementById('aoiDateFrom')?.value?.trim();
      const dateToVal = document.getElementById('aoiDateTo')?.value?.trim();

      if (currentIngestionSensor === 'maxar') {
        // -- Maxar Async Job: POST ? job_id ? poll GET /job/{id} --
        // Fixes ERR_EMPTY_RESPONSE for large AOIs that exceed browser timeout
        const startYr = dateFromVal ? parseInt(dateFromVal.slice(0, 4), 10) : 2020;
        const endYr = dateToVal ? parseInt(dateToVal.slice(0, 4), 10) : 2026;
        const yearsList = (startYr === endYr) ? [startYr] : [startYr, endYr];

        const maxarPayload = {
          geojson_polygon: parsedGeojson,
          region_id: document.getElementById('aoiRegionId')?.value?.trim() || undefined,
          region_name: document.getElementById('aoiRegionName')?.value?.trim() || undefined,
          years: yearsList,
          populate_db: true
        };

        console.log("[Canopus] Submitting Maxar async job:", maxarPayload);

        // Step 1: Submit - returns job_id INSTANTLY (no timeout risk)
        const submitRes = await fetch(`${API_BASE}/api/v1/ingest/maxar`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(maxarPayload)
        });

        if (!submitRes.ok) {
          const errData = await submitRes.json().catch(() => ({ detail: `HTTP ${submitRes.status}` }));
          throw new Error(errData.detail || `Server returned error ${submitRes.status}`);
        }

        const submitResult = await submitRes.json();
        const jobId = submitResult.job_id;
        const regionId = submitResult.region_id;
        console.log(`[Canopus] Maxar job queued: ${jobId} for region: ${regionId}`);

        if (!jobId) {
          throw new Error("No job_id received from server.");
        }

        // Step 2: Poll every 3 seconds for job completion
        const progressStages = [
          { minPct: 0, label: '[PHASE 1/4] Connecting to Maxar Wayback Archive...', sub: 'Resolving release IDs for selected epochs...' },
          { minPct: 10, label: '[PHASE 2/4] Streaming WMTS Tiles & Stitching Canvas...', sub: 'Downloading sub-meter optical imagery from Esri Wayback...' },
          { minPct: 30, label: '[PHASE 3/4] Slicing 512x512 GeoTIFF Tiles...', sub: 'Polygon intersection & VARI index computation...' },
          { minPct: 60, label: '[PHASE 4/4] RemoteCLIP Embeddings -> Qdrant & PostgreSQL...', sub: 'Indexing tile vectors into maxar_tile_embeddings...' },
        ];

        const finalResult = await new Promise((resolve, reject) => {
          let fakeProgress = 5;
          const pollInterval = setInterval(async () => {
            try {
              const pollRes = await fetch(`${API_BASE}/api/v1/ingest/job/${jobId}`);
              if (!pollRes.ok) {
                clearInterval(pollInterval);
                reject(new Error(`Poll failed: HTTP ${pollRes.status}`));
                return;
              }
              const job = await pollRes.json();
              console.log(`[Canopus] Job ${jobId}:`, job.status, job.progress + '%', job.message);

              // Animate progress bar based on server progress
              fakeProgress = Math.max(fakeProgress, job.progress || fakeProgress + 3);
              fakeProgress = Math.min(fakeProgress, 95); // cap at 95 until done
              const bar = document.getElementById('ingestProgressBar');
              const labelEl = document.getElementById('ingestStepLabel');
              const subEl = document.getElementById('ingestSubtext');
              if (bar) bar.style.width = fakeProgress + '%';

              // Pick UI stage label
              const stage = progressStages.slice().reverse().find(s => fakeProgress >= s.minPct);
              if (stage) {
                if (labelEl) labelEl.innerText = stage.label;
                if (subEl) subEl.innerText = stage.sub;
              }

              if (job.status === 'done') {
                clearInterval(pollInterval);
                resolve(job.result || {});
              } else if (job.status === 'failed') {
                clearInterval(pollInterval);
                reject(new Error(job.message || 'Maxar pipeline failed'));
              }
            } catch (pollErr) {
              clearInterval(pollInterval);
              reject(pollErr);
            }
          }, 3000);
        });

        // Handle final result
        if (!finalResult || finalResult.total_tiles_generated === 0) {
          const errDetail = (finalResult && finalResult.errors && finalResult.errors.length > 0)
            ? finalResult.errors.join("; ")
            : "No tiles extracted. Check polygon coverage area.";
          stopProgressBar(false, `Maxar returned 0 tiles: ${errDetail}`);
          return;
        }

        const epochsCount = (finalResult.epochs_processed || []).length;
        stopProgressBar(true,
          `[SUCCESS] Maxar Ingestion Complete! ${finalResult.total_tiles_generated} sub-meter tiles, ` +
          `${epochsCount} epoch(s), ${finalResult.elapsed_seconds?.toFixed(1)}s. ` +
          `Indexed in 'maxar_tile_embeddings'.`
        );

        if (currentDrawLayer) { map.removeLayer(currentDrawLayer); currentDrawLayer = null; }
        await loadCoverageRegions();
        if (finalResult.region_id) {
          const sel = document.getElementById('regionSelect');
          if (sel) sel.value = finalResult.region_id;
          onSelectRegion(finalResult.region_id);
        }
        setTimeout(() => closeIngestModal(), 1800);

      } else {
        // Standard Sentinel-2 Multi-Spectral Ingestion
        const payload = {
          geojson_polygon: parsedGeojson,
          sensor: "sentinel2",
          region_id: document.getElementById('aoiRegionId')?.value?.trim() || undefined,
          region_name: document.getElementById('aoiRegionName')?.value?.trim() || undefined,
          date_from: dateFromVal && dateFromVal !== "" ? dateFromVal : undefined,
          date_to: dateToVal && dateToVal !== "" ? dateToVal : undefined,
          num_time_buckets: parseInt(document.getElementById('aoiBuckets')?.value, 10) || 2,
          populate_db: true
        };

        const res = await fetch(`${API_BASE}/api/v1/ingest/aoi`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
          throw new Error(errData.detail || `Server returned error ${res.status}`);
        }

        const result = await res.json();
        stopProgressBar(true, `Ingested ${result.total_tiles_generated} tiles across ${result.scenes_processed ? result.scenes_processed.length : 1} scenes in ${result.elapsed_seconds}s. Vector embeddings populated in Qdrant.`);

        if (currentDrawLayer) {
          map.removeLayer(currentDrawLayer);
          currentDrawLayer = null;
        }

        // Reload Coverage and Zoom to New Region
        await loadCoverageRegions();
        if (result.region_id) {
          const sel = document.getElementById('regionSelect');
          if (sel) sel.value = result.region_id;
          onSelectRegion(result.region_id);
        }

        setTimeout(() => {
          closeIngestModal();
        }, 1800);
      }

    } else {
      // Entry Point B: Offline File / Multi-File Ingestion
      const filePath = document.getElementById('filePathInput')?.value?.trim();
      const sensorVal = document.getElementById('fileSensor')?.value || 'auto';
      const bandOrderVal = document.getElementById('fileBandOrder')?.value;
      const customBandOrder = (bandOrderVal && bandOrderVal !== 'auto') ? bandOrderVal : undefined;
      const acqDate = document.getElementById('fileAcqDate')?.value || undefined;
      const regionId = document.getElementById('fileRegionId')?.value?.trim() || undefined;

      if (selectedGeoTiffFiles && selectedGeoTiffFiles.length > 0) {
        // Upload via FormData to /api/v1/ingest/upload
        const formData = new FormData();
        for (const f of selectedGeoTiffFiles) {
          formData.append('files', f);
        }
        if (sensorVal) formData.append('sensor', sensorVal);
        if (customBandOrder) formData.append('custom_band_order', customBandOrder);
        if (acqDate) formData.append('acquisition_date', acqDate);
        if (regionId) formData.append('region_id', regionId);

        const res = await fetch(`${API_BASE}/api/v1/ingest/upload`, {
          method: 'POST',
          body: formData
        });

        if (!res.ok) {
          const errData = await res.json();
          throw new Error(errData.detail || `Server returned error ${res.status}`);
        }

        const result = await res.json();
        stopProgressBar(true, `Uploaded and ingested ${result.total_tiles_generated} tiles from ${result.files_uploaded} file(s) in ${result.elapsed_seconds}s.`);
        await loadCoverageRegions();
        if (result.region_id) {
          document.getElementById('regionSelect').value = result.region_id;
          onSelectRegion(result.region_id);
        }
      } else if (filePath) {
        // Server Local Path / Directory
        const payload = {
          file_path: filePath,
          region_id: regionId,
          sensor: sensorVal,
          custom_band_order: (customBandOrder ? customBandOrder.split(',') : undefined),
          acquisition_date: acqDate,
          populate_db: true
        };

        const res = await fetch(`${API_BASE}/api/v1/ingest/file`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        if (!res.ok) {
          const errData = await res.json();
          throw new Error(errData.detail || `Server returned error ${res.status}`);
        }

        const result = await res.json();
        stopProgressBar(true, `Ingested ${result.total_tiles_generated} tiles from offline path in ${result.elapsed_seconds}s.`);
        await loadCoverageRegions();
        if (result.region_id) {
          document.getElementById('regionSelect').value = result.region_id;
          onSelectRegion(result.region_id);
        }
      } else {
        throw new Error("Please select one or more GeoTIFF files to upload, or specify a server local path.");
      }
    }

  } catch (err) {
    stopProgressBar(false, err.message);
  }
};

// ============================================================
// 8. PHASE 2.7 — SEMANTIC RETRIEVAL CHAT & TILE INSPECTOR
// ============================================================

let attachedSearchFile = null;
let currentSearchResults = [];
let currentInspectingTile = null;
let mapTileHighlightLayer = null;
let activeSensorFilter = 'Sentinel-2';

window.selectSensorFilter = function (sensorName) {
  if (window.playTacticalClick) window.playTacticalClick();
  activeSensorFilter = (sensorName !== undefined && sensorName !== null) ? sensorName : '';

  // Update Pills visual styling
  const pSentinel = document.getElementById('pillSensorSentinel');
  const pMaxar = document.getElementById('pillSensorMaxar');
  const pAll = document.getElementById('pillSensorAll');

  const inactiveStyle = "background: rgba(255,255,255,0.03); border: 1.5px solid var(--border-color); color: var(--text-secondary); font-size: 12px; font-weight: 600; border-radius: 20px; padding: 6px 16px; cursor: pointer; display: flex; align-items: center; gap: 8px; transition: all 0.2s;";
  const activeSentinelStyle = "background: rgba(6,182,212,0.18); border: 1.5px solid var(--accent-cyan); color: #fff; font-size: 12px; font-weight: 600; border-radius: 20px; padding: 6px 16px; cursor: pointer; display: flex; align-items: center; gap: 8px; transition: all 0.2s; box-shadow: 0 0 12px rgba(6,182,212,0.25);";
  const activeMaxarStyle = "background: rgba(245,158,11,0.18); border: 1.5px solid var(--accent-amber); color: #fff; font-size: 12px; font-weight: 600; border-radius: 20px; padding: 6px 16px; cursor: pointer; display: flex; align-items: center; gap: 8px; transition: all 0.2s; box-shadow: 0 0 12px rgba(245,158,11,0.25);";
  const activeAllStyle = "background: rgba(99,102,241,0.18); border: 1.5px solid var(--accent-indigo); color: #fff; font-size: 12px; font-weight: 600; border-radius: 20px; padding: 6px 16px; cursor: pointer; display: flex; align-items: center; gap: 8px; transition: all 0.2s; box-shadow: 0 0 12px rgba(99,102,241,0.25);";

  if (pSentinel) pSentinel.style.cssText = (activeSensorFilter === 'Sentinel-2') ? activeSentinelStyle : inactiveStyle;
  if (pMaxar) pMaxar.style.cssText = (activeSensorFilter === 'Maxar') ? activeMaxarStyle : inactiveStyle;
  if (pAll) pAll.style.cssText = (!activeSensorFilter || activeSensorFilter === '') ? activeAllStyle : inactiveStyle;

  // Synchronize inputs
  const filterSensor = document.getElementById('filterSensor');
  if (filterSensor) filterSensor.value = activeSensorFilter;
  const directSelect = document.getElementById('directSensorSelect');
  if (directSelect) directSelect.value = activeSensorFilter;
};

window.syncDirectSensor = function (val) {
  selectSensorFilter(val);
};

window.onDirectSensorChange = function (val) {
  selectSensorFilter(val);
};

window.toggleSearchFilterPopover = function () {
  if (window.playTacticalClick) window.playTacticalClick();
  const popover = document.getElementById('searchFilterPopover');
  if (!popover) return;
  const isHidden = popover.style.display === 'none' || popover.style.display === '';
  popover.style.display = isHidden ? 'block' : 'none';
};

window.resetSearchFilters = function () {
  if (window.playTacticalClick) window.playTacticalClick();
  const topK = document.getElementById('filterTopK');
  if (topK) topK.value = 5;
  const topKLabel = document.getElementById('topKValueLabel');
  if (topKLabel) topKLabel.innerText = '5';

  selectSensorFilter('Sentinel-2');

  const sDate = document.getElementById('filterStartDate');
  if (sDate) sDate.value = '';
  const eDate = document.getElementById('filterEndDate');
  if (eDate) eDate.value = '';
  const minQ = document.getElementById('filterMinQuality');
  if (minQ) minQ.value = '0.0';
  const maxC = document.getElementById('filterMaxCloud');
  if (maxC) maxC.value = '100';
  const badge = document.getElementById('filterCountBadge');
  if (badge) badge.style.display = 'none';
};

window.onSearchImageSelected = function (event) {
  const file = event.target.files[0];
  if (!file) return;

  attachedSearchFile = file;
  const preview = document.getElementById('searchImageAttachmentPreview');
  const nameLabel = document.getElementById('attachedImageName');
  if (preview && nameLabel) {
    nameLabel.innerText = file.name;
    preview.style.display = 'inline-flex';
  }
};

window.clearAttachedSearchImage = function () {
  attachedSearchFile = null;
  const fileInput = document.getElementById('searchImageFileInput');
  if (fileInput) fileInput.value = '';
  const preview = document.getElementById('searchImageAttachmentPreview');
  if (preview) preview.style.display = 'none';
};

window.applyQuickPrompt = function (promptText) {
  const input = document.getElementById('searchPromptInput');
  if (input) {
    if (window.playTacticalLockChirp) window.playTacticalLockChirp();
    input.value = promptText;
    submitSemanticSearch();
  }
};

// ============================================================
// CHAT SYSTEM STATE & PERSISTENCE (PHASE 2.8)
// ============================================================

let currentConversationId = null;
let cachedConversations = [];
let defaultWelcomeFeedHtml = '';

function getCurrentUserId() {
  let uid = localStorage.getItem('canopus_user_id');
  if (!uid) {
    uid = 'analyst_' + Math.random().toString(36).substring(2, 10);
    localStorage.setItem('canopus_user_id', uid);
  }
  return uid;
}

window.regenerateUserIdentity = function () {
  const newUid = 'analyst_' + Math.random().toString(36).substring(2, 10);
  if (confirm(`Switch analyst identity to "${newUid}"?\nThis starts an isolated session separate from current chats.`)) {
    localStorage.setItem('canopus_user_id', newUid);
    const label = document.getElementById('analystIdLabel');
    if (label) label.textContent = `Analyst: ${newUid}`;
    startNewChat();
    loadUserConversations();
  }
};

window.toggleChatSidebar = function () {
  const sidebar = document.getElementById('chatSidebar');
  const wrapper = document.querySelector('.retrieval-layout-wrapper');
  const floatBtn = document.getElementById('floatingSidebarBtn');
  if (!sidebar) return;

  if (window.playTacticalClick) window.playTacticalClick();
  const isCollapsed = sidebar.classList.toggle('collapsed');
  if (wrapper) wrapper.classList.toggle('sidebar-collapsed', isCollapsed);
  if (floatBtn) {
    floatBtn.style.display = isCollapsed ? 'inline-flex' : 'none';
  }
  localStorage.setItem('canopus_sidebar_collapsed', isCollapsed ? 'true' : 'false');
};

function formatRelativeTime(dateStr) {
  if (!dateStr) return '';
  const now = new Date();
  const d = new Date(dateStr);
  const diffMs = now - d;
  const diffSec = Math.floor(diffMs / 1000);
  const diffMin = Math.floor(diffSec / 60);
  const diffHours = Math.floor(diffMin / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffMin < 1) return 'Just now';
  if (diffMin < 60) return `${diffMin}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays === 1) return 'Yesterday';
  if (diffDays < 7) return `${diffDays}d ago`;
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

window.loadUserConversations = async function () {
  const uid = getCurrentUserId();
  const listEl = document.getElementById('chatHistoryList');
  const countBadge = document.getElementById('chatCountBadge');
  if (!listEl) return;

  try {
    const res = await fetch(`${API_BASE}/api/v1/chat/conversations?user_id=${encodeURIComponent(uid)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const convs = await res.json();
    cachedConversations = convs || [];

    if (countBadge) countBadge.textContent = cachedConversations.length;
    renderConversationsList(cachedConversations);
  } catch (err) {
    console.error('Failed to load conversations:', err);
  }
};

function renderConversationsList(conversations) {
  const listEl = document.getElementById('chatHistoryList');
  if (!listEl) return;

  if (!conversations || conversations.length === 0) {
    listEl.innerHTML = `
      <div class="chat-history-empty" id="chatHistoryEmpty">
        <svg class="ui-icon icon-cyan" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#06b6d4" stroke-width="1.5"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
        <div>No saved searches yet.</div>
        <div style="font-size: 11px; opacity: 0.7;">Submit your first query to begin a persistent session.</div>
      </div>
    `;
    return;
  }

  let html = '';
  conversations.forEach(conv => {
    const isActive = conv.conversation_id === currentConversationId;
    const timeStr = formatRelativeTime(conv.updated_at || conv.created_at);
    const titleEscaped = escapeHtml(conv.title || 'Untitled Search');

    html += `
      <div class="chat-history-item ${isActive ? 'active' : ''}" 
           id="conv_item_${conv.conversation_id}"
           onclick="selectConversation('${conv.conversation_id}')"
           title="${titleEscaped}">
        <div class="chat-history-item-icon">
          <svg class="ui-icon icon-sm" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
        </div>
        <div class="chat-history-item-content">
          <div class="chat-history-item-title">${titleEscaped}</div>
          <div class="chat-history-item-meta">
            <span>${timeStr}</span>
            <span>&bull;</span>
            <span>${conv.message_count || 0} msgs</span>
          </div>
        </div>
        <div class="chat-item-actions">
          <button class="chat-item-action-btn" onclick="renameConversation(event, '${conv.conversation_id}')" title="Rename Conversation">
            <svg class="ui-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>
          </button>
          <button class="chat-item-action-btn delete" onclick="deleteConversation(event, '${conv.conversation_id}')" title="Delete Conversation">
            <svg class="ui-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
          </button>
        </div>
      </div>
    `;
  });

  listEl.innerHTML = html;
}

window.filterConversationsList = function (q) {
  const query = (q || '').trim().toLowerCase();
  if (!query) {
    renderConversationsList(cachedConversations);
    return;
  }
  const filtered = cachedConversations.filter(c => (c.title || '').toLowerCase().includes(query));
  renderConversationsList(filtered);
};

window.startNewChat = function () {
  if (window.playTacticalClick) window.playTacticalClick();
  currentConversationId = null;
  const feed = document.getElementById('retrievalChatFeed');
  if (feed && defaultWelcomeFeedHtml) {
    feed.innerHTML = defaultWelcomeFeedHtml;
    feed.scrollTop = 0;
  }
  // Clear inputs
  const textInput = document.getElementById('searchPromptInput');
  if (textInput) textInput.value = '';
  clearAttachedSearchImage();

  // Remove active styling on sidebar items
  document.querySelectorAll('.chat-history-item').forEach(el => el.classList.remove('active'));
};

window.selectConversation = async function (conversationId) {
  if (!conversationId) return;
  if (window.playTacticalLockChirp) window.playTacticalLockChirp();
  const uid = getCurrentUserId();
  const feed = document.getElementById('retrievalChatFeed');
  if (!feed) return;

  currentConversationId = conversationId;

  // Update sidebar active highlights
  document.querySelectorAll('.chat-history-item').forEach(el => el.classList.remove('active'));
  const activeEl = document.getElementById(`conv_item_${conversationId}`);
  if (activeEl) activeEl.classList.add('active');

  // Loading indicator
  feed.innerHTML = `
    <div class="chat-message assistant" style="animation: fadeIn 0.2s ease;">
      <div class="chat-avatar">
        <svg class="ui-icon icon-md" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M5 3v4"/><path d="M19 17v4"/><path d="M3 5h4"/><path d="M17 19h4"/><path d="M15 9l5-5"/><path d="M9 15l-5 5"/></svg>
      </div>
      <div class="chat-bubble">
        <div style="display: flex; align-items: center; gap: 10px; color: var(--accent-cyan); font-size: 13px;">
          <span class="status-dot"></span>
          <span>Restoring conversation state and candidate tiles...</span>
        </div>
      </div>
    </div>
  `;

  try {
    const res = await fetch(`${API_BASE}/api/v1/chat/conversations/${conversationId}?user_id=${encodeURIComponent(uid)}`);
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `Failed to load conversation (${res.status})`);
    }
    const convData = await res.json();

    // Clear feed to reconstruct exact message turns
    feed.innerHTML = '';
    const messages = convData.messages || [];

    if (messages.length === 0) {
      feed.innerHTML = defaultWelcomeFeedHtml;
      return;
    }

    // Reset current search results store with all retrieved tiles across turns
    currentSearchResults = [];

    messages.forEach(msg => {
      if (msg.role === 'user') {
        renderUserChatMessage(msg.content, null, msg.attached_image_name);
      } else if (msg.role === 'assistant') {
        const queryContext = msg.query_context || {};
        const tileResults = msg.results || [];
        // Accumulate in currentSearchResults so tile inspection works immediately
        tileResults.forEach(t => {
          if (!currentSearchResults.find(x => x.tile_id === t.tile_id)) {
            currentSearchResults.push(t);
          }
        });
        const fakeResponse = {
          results: tileResults,
          execution_time_ms: queryContext.execution_time_ms || 0,
          total_found: tileResults.length,
          search_log_id: msg.search_log_id || queryContext.search_log_id || null
        };
        renderAssistantResultsBubble(fakeResponse, queryContext);
      }
    });

    feed.scrollTop = feed.scrollHeight;
  } catch (err) {
    feed.innerHTML = '';
    renderAssistantErrorBubble(`Could not restore past chat: ${err.message}`);
  }
};

window.renameConversation = async function (e, conversationId) {
  if (e) e.stopPropagation();
  const conv = cachedConversations.find(c => c.conversation_id === conversationId);
  const currentTitle = conv ? conv.title : '';
  const newTitle = prompt('Enter a new title for this search session:', currentTitle);
  if (!newTitle || !newTitle.trim() || newTitle.trim() === currentTitle) return;

  const uid = getCurrentUserId();
  try {
    const res = await fetch(`${API_BASE}/api/v1/chat/conversations/${conversationId}?user_id=${encodeURIComponent(uid)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: newTitle.trim() })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    await loadUserConversations();
  } catch (err) {
    alert(`Rename failed: ${err.message}`);
  }
};

window.deleteConversation = async function (e, conversationId) {
  if (e) e.stopPropagation();
  if (!confirm('Are you sure you want to delete this search session? This will remove all retrieved tiles and queries permanently.')) return;

  const uid = getCurrentUserId();
  try {
    const res = await fetch(`${API_BASE}/api/v1/chat/conversations/${conversationId}?user_id=${encodeURIComponent(uid)}`, {
      method: 'DELETE'
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    if (currentConversationId === conversationId) {
      startNewChat();
    }
    await loadUserConversations();
  } catch (err) {
    alert(`Delete failed: ${err.message}`);
  }
};

function initChatSystem() {
  const uid = getCurrentUserId();
  const label = document.getElementById('analystIdLabel');
  if (label) label.textContent = `Analyst: ${uid}`;

  const feed = document.getElementById('retrievalChatFeed');
  if (feed) {
    defaultWelcomeFeedHtml = feed.innerHTML;
  }

  // Restore sidebar collapse state
  if (localStorage.getItem('canopus_sidebar_collapsed') === 'true') {
    const sidebar = document.getElementById('chatSidebar');
    const wrapper = document.querySelector('.retrieval-layout-wrapper');
    const floatBtn = document.getElementById('floatingSidebarBtn');
    if (sidebar) sidebar.classList.add('collapsed');
    if (wrapper) wrapper.classList.add('sidebar-collapsed');
    if (floatBtn) floatBtn.style.display = 'inline-flex';
  }

  loadUserConversations();
}

window.submitSemanticSearch = async function () {
  const textInput = document.getElementById('searchPromptInput');
  const promptText = textInput ? textInput.value.trim() : '';

  if (!promptText && !attachedSearchFile) {
    alert("Please enter a search prompt or upload a reference image.");
    return;
  }

  const chatFeed = document.getElementById('retrievalChatFeed');
  if (!chatFeed) return;

  const submitBtn = document.getElementById('submitSearchBtn');
  if (submitBtn) submitBtn.classList.add('searching');
  if (window.tacticalAudio) window.tacticalAudio.playLockOnChirp();

  // 1. Gather Filters
  let rawTopK = parseInt(document.getElementById('filterTopK')?.value || '5', 10);
  const topK = isNaN(rawTopK) ? 5 : Math.min(100, Math.max(1, rawTopK));

  const directSensor = document.getElementById('directSensorSelect')?.value;
  const popoverSensor = document.getElementById('filterSensor')?.value?.trim();
  const rawSensor = activeSensorFilter !== undefined ? activeSensorFilter : ((directSensor && directSensor !== "") ? directSensor : popoverSensor);
  const sensor = (rawSensor && rawSensor !== "" && rawSensor !== "Any") ? rawSensor : undefined;
  const startDate = document.getElementById('filterStartDate')?.value || undefined;
  const endDate = document.getElementById('filterEndDate')?.value || undefined;

  // Parse and normalize minQuality to strictly satisfy backend Pydantic constraint (0.0 to 1.0)
  let rawQuality = parseFloat(document.getElementById('filterMinQuality')?.value || '0.0');
  let minQuality = 0.0;
  if (!isNaN(rawQuality) && rawQuality > 0) {
    if (rawQuality > 1.0) {
      rawQuality = rawQuality / 100.0;
    }
    minQuality = Math.min(1.0, Math.max(0.0, rawQuality));
  }

  // Parse and clamp maxCloud to [0.0, 100.0]
  let rawCloud = parseFloat(document.getElementById('filterMaxCloud')?.value || '100.0');
  let maxCloud = 100.0;
  if (!isNaN(rawCloud)) {
    maxCloud = Math.min(100.0, Math.max(0.0, rawCloud));
  }

  // Query Turn Context for Visual Pipeline Rendering
  const queryTurnContext = {
    promptText: promptText,
    attachedFile: attachedSearchFile,
    attachedFileName: attachedSearchFile ? attachedSearchFile.name : null,
    filePreviewUrl: attachedSearchFile ? URL.createObjectURL(attachedSearchFile) : null,
    sensor: sensor || 'All Sensors',
    topK: topK,
    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  };

  // 2. Render User Message in Chat
  renderUserChatMessage(promptText, attachedSearchFile);

  // Clear input fields
  if (textInput) textInput.value = '';
  const searchFileToUpload = attachedSearchFile;
  clearAttachedSearchImage();

  // 3. Render Assistant Loading Message
  const loadingMsgId = `loading_${Date.now()}`;
  renderAssistantLoadingBubble(loadingMsgId);
  chatFeed.scrollTop = chatFeed.scrollHeight;

  try {
    let responseData = null;

    if (searchFileToUpload) {
      // Image-to-Image Query (Multipart)
      const formData = new FormData();
      formData.append('file', searchFileToUpload);
      formData.append('top_k', topK.toString());
      if (sensor) formData.append('sensor', sensor);
      if (startDate) formData.append('start_date', startDate);
      if (endDate) formData.append('end_date', endDate);
      formData.append('min_quality', minQuality.toFixed(2));
      formData.append('max_cloud_pct', maxCloud.toString());
      formData.append('min_similarity', '0.65');

      const res = await fetch(`${API_BASE}/api/v1/search/image`, {
        method: 'POST',
        body: formData
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || `Server returned error ${res.status}`);
      }
      responseData = await res.json();

    } else {
      // Text Query (JSON)
      const payload = {
        query_text: promptText,
        top_k: topK,
        filters: {
          sensor: sensor || undefined,
          start_date: startDate ? new Date(startDate).toISOString() : undefined,
          end_date: endDate ? new Date(endDate).toISOString() : undefined,
          min_quality: minQuality > 0 ? parseFloat(minQuality.toFixed(2)) : undefined,
          max_cloud_pct: maxCloud < 100 ? parseFloat(maxCloud.toFixed(1)) : undefined
        }
      };

      const res = await fetch(`${API_BASE}/api/v1/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || `Server returned error ${res.status}`);
      }
      responseData = await res.json();
    }

    if (responseData && responseData.search_log_id) {
      window.currentSearchLogId = responseData.search_log_id;
    }
    if (responseData && responseData.query) {
      window.currentSearchQuery = responseData.query;
    } else if (promptText) {
      window.currentSearchQuery = promptText;
    } else if (searchFileToUpload) {
      window.currentSearchQuery = 'image:' + searchFileToUpload.name;
    }

    // Remove Loading Bubble
    const loadingElem = document.getElementById(loadingMsgId);
    if (loadingElem) loadingElem.remove();
    if (submitBtn) submitBtn.classList.remove('searching');

    // Render Search Results into Visual Pipeline
    renderAssistantResultsBubble(responseData, queryTurnContext);
    chatFeed.scrollTop = chatFeed.scrollHeight;

    // 4. Save turn to Persistent Database & Update Sidebar
    try {
      const uid = getCurrentUserId();
      // If currently in New Chat mode, initialize conversation thread
      if (!currentConversationId) {
        let titleCandidate = promptText || (searchFileToUpload ? `Image: ${searchFileToUpload.name}` : 'Tactical Search');
        if (titleCandidate.length > 38) titleCandidate = titleCandidate.substring(0, 38) + '...';
        const createRes = await fetch(`${API_BASE}/api/v1/chat/conversations`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_id: uid, title: titleCandidate })
        });
        if (createRes.ok) {
          const newConv = await createRes.json();
          currentConversationId = newConv.conversation_id;
        }
      }

      if (currentConversationId) {
        const turnMessages = [
          {
            role: 'user',
            content: promptText || '(Reference Image Query)',
            attached_image_name: searchFileToUpload ? searchFileToUpload.name : null,
            query_context: {
              sensor: sensor || 'All Sensors',
              topK: topK,
              startDate: startDate,
              endDate: endDate,
              minQuality: minQuality,
              maxCloud: maxCloud,
              timestamp: queryTurnContext.timestamp
            }
          },
          {
            role: 'assistant',
            content: `Retrieved ${responseData.total_found || (responseData.results ? responseData.results.length : 0)} candidate tiles`,
            query_context: {
              sensor: sensor || 'All Sensors',
              topK: topK,
              execution_time_ms: responseData.execution_time_ms,
              total_found: responseData.total_found,
              timestamp: queryTurnContext.timestamp
            },
            results: responseData.results || []
          }
        ];

        await fetch(`${API_BASE}/api/v1/chat/conversations/${currentConversationId}/messages?user_id=${encodeURIComponent(uid)}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(turnMessages)
        });

        // Refresh sidebar list
        await loadUserConversations();
      }
    } catch (saveErr) {
      console.warn('Could not persist chat turn to database:', saveErr);
    }

  } catch (err) {
    if (submitBtn) submitBtn.classList.remove('searching');
    const loadingElem = document.getElementById(loadingMsgId);
    if (loadingElem) loadingElem.remove();
    renderAssistantErrorBubble(err.message);
    chatFeed.scrollTop = chatFeed.scrollHeight;
  }
};

function renderUserChatMessage(text, file, attachedImageName) {
  const chatFeed = document.getElementById('retrievalChatFeed');
  if (!chatFeed) return;
  const msg = document.createElement('div');
  msg.className = 'chat-message user';

  let contentHtml = '';
  if (file) {
    const previewUrl = URL.createObjectURL(file);
    contentHtml += `<div style="margin-bottom: 8px;"><img src="${previewUrl}" style="max-height: 140px; border-radius: var(--radius-sm); border: 1px solid rgba(6,182,212,0.4);" alt="Uploaded Query"></div>`;
  } else if (attachedImageName) {
    contentHtml += `<div style="margin-bottom: 8px; display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: var(--radius-sm); background: rgba(6,182,212,0.15); border: 1px solid rgba(6,182,212,0.3); font-size: 11px; font-family: 'JetBrains Mono', monospace; color: #38bdf8;">
      <svg class="ui-icon icon-sm icon-cyan" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#06b6d4" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
      <span>${escapeHtml(attachedImageName)}</span>
    </div>`;
  }
  if (text) {
    contentHtml += `<div class="chat-text" style="font-weight: 500; font-size: 14px;">${escapeHtml(text)}</div>`;
  }

  msg.innerHTML = `
    <div class="chat-avatar">
      <svg class="ui-icon icon-md" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
    </div>
    <div class="chat-bubble">
      ${contentHtml}
    </div>
  `;
  chatFeed.appendChild(msg);
}

function renderAssistantLoadingBubble(id) {
  const chatFeed = document.getElementById('retrievalChatFeed');
  if (!chatFeed) return;
  const msg = document.createElement('div');
  msg.className = 'chat-message assistant';
  msg.id = id;
  msg.innerHTML = `
    <div class="chat-avatar">
      <svg class="ui-icon icon-md" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M5 3v4"/><path d="M19 17v4"/><path d="M3 5h4"/><path d="M17 19h4"/><path d="M15 9l5-5"/><path d="M9 15l-5 5"/></svg>
    </div>
    <div class="chat-bubble">
      <div style="display: flex; align-items: center; gap: 10px; color: var(--accent-cyan); font-size: 13px;">
        <span class="status-dot"></span>
        <span>Computing RemoteCLIP 512-dim embedding & searching Qdrant vector archive...</span>
      </div>
    </div>
  `;
  chatFeed.appendChild(msg);
}

function renderAssistantErrorBubble(errMsg) {
  const chatFeed = document.getElementById('retrievalChatFeed');
  if (!chatFeed) return;
  const msg = document.createElement('div');
  msg.className = 'chat-message assistant';
  msg.innerHTML = `
    <div class="chat-avatar" style="background: rgba(244, 63, 94, 0.2); border-color: rgba(244, 63, 94, 0.4); color: #f43f5e;">
      <svg class="ui-icon icon-md" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#f43f5e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
    </div>
    <div class="chat-bubble" style="border-color: rgba(244, 63, 94, 0.4); background: rgba(244, 63, 94, 0.1);">
      <div style="color: var(--accent-rose); font-weight: 700; font-size: 13px; margin-bottom: 4px;">Search Failed</div>
      <div style="font-size: 12px; color: var(--text-secondary); line-height: 1.4;">${escapeHtml(errMsg)}</div>
    </div>
  `;
  chatFeed.appendChild(msg);
}

function renderAssistantResultsBubble(response, queryInfo) {
  const chatFeed = document.getElementById('retrievalChatFeed');
  if (!chatFeed) return;

  if (window.playTacticalLockChirp) window.playTacticalLockChirp();
  currentSearchResults = response.results || [];
  const execTime = response.execution_time_ms || 0;
  const totalFound = response.total_found || 0;
  const requestedK = queryInfo?.topK || 5;
  const targetSensorName = queryInfo?.sensor || 'All Sensors';

  const searchLogId = (response && response.search_log_id) || (queryInfo && queryInfo.search_log_id) || (window.currentSearchLogId || '');

  // Build Results Grid or Empty State
  let resultsGridHtml = '';
  if (totalFound === 0) {
    resultsGridHtml = `
      <div style="padding: 36px 20px; text-align: center; background: rgba(0,0,0,0.25); border-radius: var(--radius-md); border: 1px dashed rgba(255,255,255,0.1); color: var(--text-secondary); font-size: 13px;">
        <svg class="ui-icon icon-lg icon-cyan" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#06b6d4" stroke-width="2" style="margin-bottom: 8px;"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
        <div style="font-weight: 600; color: #f1f5f9; margin-bottom: 4px;">No matching satellite tiles found in the archive</div>
        <div style="font-size: 12px; color: var(--text-muted);">Try broadening the sensor filter, adjusting the acquisition date range, or lowering the quality gate.</div>
      </div>
    `;
  } else {
    resultsGridHtml = `<div class="retrieval-results-grid">`;
    response.results.forEach((item, idx) => {
      const scorePct = (item.score * 100).toFixed(1);
      const thumbUrl = getTileThumbnailUrl(item);
      const dateStr = item.acquisition_date ? item.acquisition_date.split('T')[0] : 'Historical';
      const isMaxar = (item.sensor && item.sensor.toLowerCase().includes('maxar')) || item.tile_id.includes('maxar');
      const sensorLabel = isMaxar ? 'Maxar 0.5m' : 'Sentinel-2';

      let ndviVal = item.mean_ndvi !== null && item.mean_ndvi !== undefined ? item.mean_ndvi.toFixed(2) : null;
      if (ndviVal === null && item.spot_description) {
        const m = item.spot_description.match(/VARI visible vegetation index:\s*([\d\.\-]+)/i);
        if (m && m[1]) ndviVal = parseFloat(m[1]).toFixed(2);
      }
      const ndwiVal = item.mean_ndwi !== null && item.mean_ndwi !== undefined ? item.mean_ndwi.toFixed(2) : null;
      const ndbiVal = item.mean_ndbi !== null && item.mean_ndbi !== undefined ? item.mean_ndbi.toFixed(2) : null;

      let chipsHtml = '';
      if (isMaxar) {
        if (ndviVal !== null) {
          chipsHtml += `<span class="spec-chip ndvi">VARI ${ndviVal}</span>`;
        }
        chipsHtml += `<span class="spec-chip res">RGB 0.5m Optical</span>`;
      } else {
        if (ndviVal !== null) chipsHtml += `<span class="spec-chip ndvi">NDVI ${ndviVal}</span>`;
        if (ndwiVal !== null) chipsHtml += `<span class="spec-chip ndwi">NDWI ${ndwiVal}</span>`;
        if (ndbiVal !== null) chipsHtml += `<span class="spec-chip ndbi">NDBI ${ndbiVal}</span>`;
        if (!chipsHtml) chipsHtml = `<span class="spec-chip ndvi">10m Multi-Spectral</span>`;
      }

      const isHighConfidence = scorePct >= 80;
      resultsGridHtml += `
        <div class="result-card hud-bracketed ${isHighConfidence ? 'high-confidence-halo' : ''}">
          <div class="result-thumb-wrap laser-scanner-wrap">
            <img class="result-thumb-img" src="${thumbUrl}" alt="${item.tile_id}" onerror="handleTileThumbError(this, '${item.tile_id}')">
            <div class="result-sensor-tag">
              <svg class="ui-icon icon-sm" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M5 3v4"/><path d="M19 17v4"/><path d="M3 5h4"/><path d="M17 19h4"/><path d="M15 9l5-5"/><path d="M9 15l-5 5"/></svg>
              ${sensorLabel}
            </div>
            <div class="result-score-badge ${isHighConfidence ? 'high-confidence-halo' : ''}">${scorePct}% Match</div>
          </div>
          <div class="result-body">
            <div class="result-title-row">
              <span class="result-tile-id" title="${item.tile_id}">#${idx + 1} &bull; ${item.tile_id}</span>
              <span class="result-date">${dateStr}</span>
            </div>

            <!-- Spot Description -->
            <div class="result-spot-desc">
              ${escapeHtml(item.spot_description || (isMaxar ? 'High-resolution sub-meter optical reconnaissance imagery.' : 'Multi-spectral satellite signature matched.'))}
            </div>

            <!-- Spectral Chips -->
            <div class="result-spectral-chips">
              ${chipsHtml}
            </div>

            <!-- Action Buttons -->
            <div class="result-card-actions">
              <button class="result-action-btn primary" onclick="openTileInspect('${item.tile_id}')" title="Inspect imagery and spectral lineage">
                <svg class="ui-icon icon-sm icon-cyan" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#06b6d4" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
                Inspect
              </button>
              <button class="result-action-btn" onclick="openTileInMap('${item.tile_id}', ${item.centroid_lat || 'null'}, ${item.centroid_lon || 'null'}, ${JSON.stringify(item.geometry_geojson || null).replace(/"/g, '&quot;')})" title="Navigate to physical tile in Map Explorer">
                <svg class="ui-icon icon-sm" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21"/><line x1="9" y1="3" x2="9" y2="18"/><line x1="15" y1="6" x2="15" y2="21"/></svg>
                Map
              </button>
              <button class="result-action-btn card-accept-btn" onclick="quickCardFeedback('${item.tile_id}', true, this, '${searchLogId}')" title="Accept Target (Hit)" style="background: rgba(16,185,129,0.15); border-color: rgba(16,185,129,0.35); color: #34d399;">
                <svg class="ui-icon icon-emerald" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>
                Accept
              </button>
              <button class="result-action-btn card-reject-btn" onclick="quickCardFeedback('${item.tile_id}', false, this, '${searchLogId}')" title="Reject Noise (False Alarm)" style="background: rgba(244,63,94,0.15); border-color: rgba(244,63,94,0.35); color: #fb7185;">
                <svg class="ui-icon icon-rose" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                Reject
              </button>
            </div>
          </div>
        </div>
      `;
    });
    resultsGridHtml += `</div>`;
  }

  // Build clean message bubble
  const msg = document.createElement('div');
  msg.className = 'chat-message assistant';
  msg.innerHTML = `
    <div class="chat-avatar">
      <svg class="ui-icon icon-md" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M5 3v4"/><path d="M19 17v4"/><path d="M3 5h4"/><path d="M17 19h4"/><path d="M15 9l5-5"/><path d="M9 15l-5 5"/></svg>
    </div>
    <div class="chat-bubble" style="width: 100%; max-width: 100%;">
      <div class="chat-bubble-header" style="flex-wrap: wrap; gap: 8px; margin-bottom: 14px; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 10px;">
        <div style="display: flex; align-items: center; gap: 8px;">
          <span class="assistant-name" style="font-size: 14px; font-weight: 700;">Canopus Search Results</span>
          <span class="tag-badge" style="background: rgba(6,182,212,0.15); color: #38bdf8;">${targetSensorName}</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; font-size: 11px; font-family: 'JetBrains Mono', monospace;">
          <span class="pipeline-badge highlight" style="display: inline-flex; align-items: center; gap: 4px; padding: 3px 8px; border-radius: 4px; background: rgba(6,182,212,0.15); color: #38bdf8; border: 1px solid rgba(6,182,212,0.3);">
            <svg class="ui-icon icon-sm icon-cyan" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#06b6d4" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
            ${execTime} ms
          </span>
          <span class="pipeline-badge success" style="display: inline-flex; align-items: center; gap: 4px; padding: 3px 8px; border-radius: 4px; background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.3);">
            <svg class="ui-icon icon-sm icon-emerald" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>
            ${totalFound} Matches
          </span>
          <span style="color: var(--text-muted);">Top ${requestedK}</span>
        </div>
      </div>

      <!-- Clean Result Cards Grid -->
      ${resultsGridHtml}
    </div>
  `;

  chatFeed.appendChild(msg);
  chatFeed.scrollTop = chatFeed.scrollHeight;

  const asstNameEl = msg.querySelector('.assistant-name');
  if (asstNameEl && window.decodeScrambleText) {
    window.decodeScrambleText(asstNameEl, 'Canopus Search Results', 380);
  }
  if (window.tacticalAudio) {
    window.tacticalAudio.playSonarPing();
  }
}

// Fallback image generator for missing/mock tile previews
window.getTileThumbnailUrl = function (item) {
  if (!item) return '';
  if (item.thumbnail_url && !item.thumbnail_url.includes('/null/')) {
    let t = item.thumbnail_url.trim();
    if (t.startsWith('//')) t = t.replace(/^\/+/, '/');
    if (t.startsWith('/app/')) t = t.replace('/app/', '/');
    else if (t.startsWith('app/')) t = t.replace('app/', '/');
    return t.startsWith('http') ? t : `${API_BASE}${t.startsWith('/') ? '' : '/'}${t}`;
  }
  if (item.thumbnail_path) {
    let t = item.thumbnail_path.replace(/\\/g, '/');
    if (t.includes('/data/')) t = '/data/' + t.split('/data/')[1];
    else if (t.startsWith('/app/')) t = t.replace('/app/', '/');
    return `${API_BASE}${t.startsWith('/') ? '' : '/'}${t}`;
  }
  if (item.site_key && item.site_key !== 'null') {
    return `${API_BASE}/data/tiles/${item.site_key}/${item.tile_id}_thumb.jpg`;
  }
  return createTileSvgDataUri(item.tile_id);
};

window.handleTileThumbError = function (img, tileId) {
  if (!img) return;
  const currentSrc = img.src || '';
  if (currentSrc.includes('_thumb.jpg')) {
    img.src = currentSrc.replace('_thumb.jpg', '_preview.jpg');
    return;
  }
  if (currentSrc.includes('_preview.jpg')) {
    img.src = currentSrc.replace('_preview.jpg', '.jpg');
    return;
  }
  img.onerror = null;
  img.src = createTileSvgDataUri(tileId);
};

function createTileSvgDataUri(tileId) {
  const cleanId = escapeHtml(tileId || 'Sentinel-2 Tile');
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
    <rect width="512" height="512" fill="#0d1424"/>
    <defs>
      <radialGradient id="grad" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stop-color="#1e293b"/>
        <stop offset="100%" stop-color="#070a12"/>
      </radialGradient>
      <pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse">
        <path d="M 32 0 L 0 0 0 32" fill="none" stroke="rgba(6,182,212,0.12)" stroke-width="1"/>
      </pattern>
    </defs>
    <rect width="512" height="512" fill="url(#grad)"/>
    <rect width="512" height="512" fill="url(#grid)"/>
    <circle cx="256" cy="256" r="140" fill="none" stroke="rgba(6,182,212,0.25)" stroke-width="2" stroke-dasharray="6,6"/>
    <line x1="256" y1="80" x2="256" y2="432" stroke="rgba(6,182,212,0.3)" stroke-width="1.5"/>
    <line x1="80" y1="256" x2="432" y2="256" stroke="rgba(6,182,212,0.3)" stroke-width="1.5"/>
    <text x="256" y="240" font-family="monospace" font-size="28" fill="#38bdf8" text-anchor="middle" font-weight="bold">SENTINEL-2</text>
    <text x="256" y="275" font-family="monospace" font-size="14" fill="#94a3b8" text-anchor="middle">512x512 MULTI-SPECTRAL</text>
    <text x="256" y="300" font-family="monospace" font-size="12" fill="#06b6d4" text-anchor="middle">${cleanId}</text>
  </svg>`;
  return 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);
}

// ============================================================
// 9. TILE INSPECTION MODAL & MAP HANDOFF
// ============================================================

window.openTileInspect = function (tileId) {
  if (window.playTacticalLockChirp) window.playTacticalLockChirp();
  // Comprehensive lookup across all tile stores
  let item = (window.__tileMap && window.__tileMap.get(tileId))
    || currentSearchResults.find(t => t.tile_id === tileId)
    || (window.__lastDiscoveredTiles && Array.isArray(window.__lastDiscoveredTiles) && window.__lastDiscoveredTiles.find(t => t.tile_id === tileId))
    || (typeof currentClusterTiles !== 'undefined' && Array.isArray(currentClusterTiles) && currentClusterTiles.find(t => t.tile_id === tileId))
    || (window.currentClusterTiles && Array.isArray(window.currentClusterTiles) && window.currentClusterTiles.find(t => t.tile_id === tileId));

  if (!item && typeof activeClusters !== 'undefined' && Array.isArray(activeClusters)) {
    const cl = activeClusters.find(c => c.representative_tile_id === tileId);
    if (cl) {
      item = {
        tile_id: tileId,
        sensor: cl.sensor || 'Multi-Sensor',
        spot_description: `Representative medoid tile of ${cl.label || 'cluster'}.`,
        score: 1.0,
        acquisition_date: cl.computed_at
      };
    }
  }

  // Graceful fallback so inspecting any tile ID never silently fails or blocks
  if (!item) {
    item = {
      tile_id: tileId,
      sensor: tileId.includes('maxar') ? 'Maxar WorldView' : 'Sentinel-2 L2A',
      spot_description: 'Satellite imagery details and multi-spectral analysis from archive.',
      score: 1.0,
      mean_ndvi: null,
      mean_ndwi: null,
      mean_ndbi: null,
      acquisition_date: 'Historical'
    };
  }

  currentInspectingTile = item;
  const modal = document.getElementById('tileInspectModal');
  if (!modal) return;

  // Move modal to end of body to guarantee it sits above all other modals (discovery, popovers, etc.)
  if (modal.parentElement !== document.body || modal !== document.body.lastElementChild) {
    document.body.appendChild(modal);
  }

  const isMaxar = (item.sensor && item.sensor.toLowerCase().includes('maxar')) || item.tile_id.includes('maxar');

  // Set Title & Subtitle
  const titleEl = document.getElementById('inspectTileTitle');
  if (titleEl) titleEl.innerText = `Tile: ${item.tile_id}`;

  const subTitleEl = document.getElementById('inspectTileSubtitle');
  if (subTitleEl) subTitleEl.innerText = `${item.sensor || (isMaxar ? 'Maxar WorldView' : 'Sentinel-2 L2A')} • Scene: ${item.scene_id || 'N/A'}`;

  // Image & Score
  const thumbUrl = getTileThumbnailUrl(item);
  const imgEl = document.getElementById('inspectTileImage');
  if (imgEl) imgEl.src = thumbUrl;

  const scoreBadgeEl = document.getElementById('inspectScoreBadge');
  if (scoreBadgeEl) scoreBadgeEl.innerText = `${item.score !== undefined && item.score !== null ? (item.score * 100).toFixed(1) : '100.0'}% Match`;

  // Description
  const descBoxEl = document.getElementById('inspectDescriptionBox');
  if (descBoxEl) {
    descBoxEl.innerText = item.spot_description || (isMaxar ? "High-resolution optical reconnaissance imagery from Maxar WorldView archive." : "Multi-spectral surface reflectance and tri-spectral indices evaluated.");
  }

  // Download GeoTIFF link
  const tifBtn = document.getElementById('inspectDownloadTifBtn');
  if (tifBtn) {
    tifBtn.href = `${API_BASE}/data/tiles/${item.site_key || 'default'}/${item.tile_id}.tif`;
  }

  // Spectral Meters
  let ndvi = item.mean_ndvi !== null && item.mean_ndvi !== undefined ? item.mean_ndvi : null;
  const ndwi = item.mean_ndwi !== null && item.mean_ndwi !== undefined ? item.mean_ndwi : null;
  const ndbi = item.mean_ndbi !== null && item.mean_ndbi !== undefined ? item.mean_ndbi : null;

  // Fallback check: if ndvi is null but spot_description mentions VARI
  if (ndvi === null && item.spot_description) {
    const m = item.spot_description.match(/VARI visible vegetation index:\s*([\d\.\-]+)/i);
    if (m && m[1]) {
      ndvi = parseFloat(m[1]);
    }
  }

  // Label
  const labelNdviEl = document.getElementById('inspectLabelNdvi');
  if (labelNdviEl) {
    labelNdviEl.innerText = isMaxar ? 'NDVI / VARI (Vegetation)' : 'NDVI (Vegetation)';
  }

  // Values and Progress Bars
  const valNdviEl = document.getElementById('inspectValNdvi');
  const barNdviEl = document.getElementById('inspectBarNdvi');
  if (valNdviEl && barNdviEl) {
    if (ndvi !== null && !isNaN(ndvi)) {
      valNdviEl.innerText = ndvi.toFixed(3);
      barNdviEl.style.width = `${Math.min(100, Math.max(0, ((ndvi + 1) / 2) * 100))}%`;
    } else {
      valNdviEl.innerText = isMaxar ? 'N/A (Optical RGB)' : 'N/A';
      barNdviEl.style.width = '0%';
    }
  }

  const valNdwiEl = document.getElementById('inspectValNdwi');
  const barNdwiEl = document.getElementById('inspectBarNdwi');
  if (valNdwiEl && barNdwiEl) {
    if (ndwi !== null && !isNaN(ndwi)) {
      valNdwiEl.innerText = ndwi.toFixed(3);
      barNdwiEl.style.width = `${Math.min(100, Math.max(0, ((ndwi + 1) / 2) * 100))}%`;
    } else {
      valNdwiEl.innerText = isMaxar ? 'N/A (Optical RGB)' : 'N/A';
      barNdwiEl.style.width = '0%';
    }
  }

  const valNdbiEl = document.getElementById('inspectValNdbi');
  const barNdbiEl = document.getElementById('inspectBarNdbi');
  if (valNdbiEl && barNdbiEl) {
    if (ndbi !== null && !isNaN(ndbi)) {
      valNdbiEl.innerText = ndbi.toFixed(3);
      barNdbiEl.style.width = `${Math.min(100, Math.max(0, ((ndbi + 1) / 2) * 100))}%`;
    } else {
      valNdbiEl.innerText = isMaxar ? 'N/A (Optical RGB)' : 'N/A';
      barNdbiEl.style.width = '0%';
    }
  }

  // Metadata Table
  const metaTileId = document.getElementById('inspectMetaTileId');
  if (metaTileId) metaTileId.innerText = item.tile_id;

  const metaSceneId = document.getElementById('inspectMetaSceneId');
  if (metaSceneId) metaSceneId.innerText = item.scene_id || 'N/A';

  const metaDate = document.getElementById('inspectMetaDate');
  if (metaDate) metaDate.innerText = item.acquisition_date ? item.acquisition_date.split('T')[0] : 'Historical';

  const metaCoords = document.getElementById('inspectMetaCoords');
  if (metaCoords) {
    metaCoords.innerText = (item.centroid_lat && item.centroid_lon)
      ? `${item.centroid_lat.toFixed(4)}° N, ${item.centroid_lon.toFixed(4)}° E`
      : 'N/A';
  }

  const metaCloud = document.getElementById('inspectMetaCloud');
  if (metaCloud) metaCloud.innerText = `${item.cloud_pct !== null && item.cloud_pct !== undefined ? item.cloud_pct.toFixed(1) : '0.0'}%`;

  const metaQuality = document.getElementById('inspectMetaQuality');
  if (metaQuality) metaQuality.innerText = `${item.quality_confidence !== null && item.quality_confidence !== undefined ? item.quality_confidence.toFixed(2) : '1.00'} (Gated)`;

  modal.style.zIndex = '100000';
  modal.style.setProperty('display', 'flex', 'important');

  // If tile metadata or thumbnail is missing/incomplete, asynchronously fetch from database API
  if (item.mean_ndvi === null || item.mean_ndvi === undefined || !item.thumbnail_url || !item.centroid_lat) {
    fetch(`${API_BASE}/api/v1/archive/tiles/${encodeURIComponent(tileId)}`)
      .then(res => res.ok ? res.json() : null)
      .then(data => {
        if (data && data.status === 'success' && data.tile) {
          const t = data.tile;
          currentInspectingTile = Object.assign(currentInspectingTile || {}, t);
          if (imgEl && t.thumbnail_url) imgEl.src = getTileThumbnailUrl(t);
          if (subTitleEl && t.scene_id) subTitleEl.innerText = `${t.sensor || (isMaxar ? 'Maxar WorldView' : 'Sentinel-2 L2A')} • Scene: ${t.scene_id}`;
          if (metaSceneId && t.scene_id) metaSceneId.innerText = t.scene_id;
          if (metaCoords && t.centroid_lat && t.centroid_lon) {
            metaCoords.innerText = `${t.centroid_lat.toFixed(4)}° N, ${t.centroid_lon.toFixed(4)}° E`;
          }
          if (metaDate && t.acquisition_date) metaDate.innerText = t.acquisition_date.split('T')[0];
          if (t.cloud_pct !== null && t.cloud_pct !== undefined && metaCloud) {
            metaCloud.innerText = `${t.cloud_pct.toFixed(1)}%`;
          }
          if (t.quality_confidence !== null && t.quality_confidence !== undefined && metaQuality) {
            metaQuality.innerText = `${t.quality_confidence.toFixed(2)} (Gated)`;
          }
          if (t.mean_ndvi !== null && t.mean_ndvi !== undefined && valNdviEl && barNdviEl) {
            valNdviEl.innerText = t.mean_ndvi.toFixed(3);
            barNdviEl.style.width = `${Math.min(100, Math.max(0, ((t.mean_ndvi + 1) / 2) * 100))}%`;
          }
          if (t.mean_ndwi !== null && t.mean_ndwi !== undefined && valNdwiEl && barNdwiEl) {
            valNdwiEl.innerText = t.mean_ndwi.toFixed(3);
            barNdwiEl.style.width = `${Math.min(100, Math.max(0, ((t.mean_ndwi + 1) / 2) * 100))}%`;
          }
          if (t.mean_ndbi !== null && t.mean_ndbi !== undefined && valNdbiEl && barNdbiEl) {
            valNdbiEl.innerText = t.mean_ndbi.toFixed(3);
            barNdbiEl.style.width = `${Math.min(100, Math.max(0, ((t.mean_ndbi + 1) / 2) * 100))}%`;
          }
          if (t.site_key && tifBtn) {
            tifBtn.href = `${API_BASE}/data/tiles/${t.site_key}/${t.tile_id}.tif`;
          }
        }
      })
      .catch(e => console.warn('Tile details async fetch error:', e));
  }
};

window.closeTileInspect = function () {
  if (window.playTacticalClick) window.playTacticalClick();
  const modal = document.getElementById('tileInspectModal');
  if (modal) modal.style.setProperty('display', 'none', 'important');
  currentInspectingTile = null;
};

function displayFloatingFeedbackToast(msg) {
  if (window.showTacticalToast) {
    window.showTacticalToast(msg);
  } else {
    const existing = document.getElementById('floatingFeedbackToast');
    if (existing) existing.remove();
    const toast = document.createElement('div');
    toast.id = 'floatingFeedbackToast';
    toast.style.cssText = 'position:fixed;bottom:24px;right:24px;background:#0d1424;border:1px solid #06b6d4;color:#38bdf8;padding:10px 18px;border-radius:6px;font-family:monospace;font-size:12px;z-index:99999;box-shadow:0 0 20px rgba(6,182,212,0.4);display:flex;align-items:center;gap:8px;';
    toast.innerHTML = `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#10b981;"></span> ${escapeHtml(msg)}`;
    document.body.appendChild(toast);
    setTimeout(() => {
      if (toast.parentNode) toast.remove();
    }, 2800);
  }
}

window.quickCardFeedback = async function (tileId, isRelevant, btnElem, searchLogId) {
  const card = btnElem ? btnElem.closest('.result-card') : null;
  const logId = searchLogId || window.currentSearchLogId || null;
  const inputEl = document.getElementById('searchPromptInput');
  let q = (inputEl && inputEl.value && inputEl.value.trim()) || '';
  if (!q && typeof currentSearchQuery !== 'undefined' && currentSearchQuery) {
    q = currentSearchQuery;
  }
  if (!q && window.currentSearchQuery) {
    q = window.currentSearchQuery;
  }
  if (!q && window.currentSearchImageName) {
    q = 'image:' + window.currentSearchImageName;
  }
  if (!q) q = 'General Satellite Reconnaissance';

  // Immediate visual feedback on card
  if (card) {
    card.style.transition = 'all 0.3s ease';
    if (isRelevant) {
      card.style.borderColor = 'rgba(16, 185, 129, 0.9)';
      card.style.boxShadow = '0 0 20px rgba(16, 185, 129, 0.35)';
      const hitBtn = card.querySelector('.card-accept-btn');
      if (hitBtn) {
        hitBtn.style.background = '#10b981';
        hitBtn.style.color = '#000';
      }
      const missBtn = card.querySelector('.card-reject-btn');
      if (missBtn) missBtn.style.opacity = '0.4';
    } else {
      card.style.borderColor = 'rgba(244, 63, 94, 0.9)';
      card.style.boxShadow = '0 0 20px rgba(244, 63, 94, 0.35)';
      const missBtn = card.querySelector('.card-reject-btn');
      if (missBtn) {
        missBtn.style.background = '#f43f5e';
        missBtn.style.color = '#fff';
      }
      const hitBtn = card.querySelector('.card-accept-btn');
      if (hitBtn) hitBtn.style.opacity = '0.4';
    }
  }

  if (window.playTacticalLockChirp && isRelevant) window.playTacticalLockChirp();
  else if (window.playTacticalClick) window.playTacticalClick();

  try {
    const res = await fetch(`${API_BASE}/api/v1/review/retrieval/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        search_log_id: logId ? parseInt(logId) : null,
        query_text: q,
        tile_id: tileId,
        relevant: isRelevant,
        feedback_type: isRelevant ? 'positive' : 'negative',
        analyst_id: 'CAPT. VERMA (INTEL-01)'
      })
    });
    const data = await res.json();
    if (data.status === 'success') {
      const msg = `TILE #${tileId} // ${isRelevant ? 'ACCEPTED (CONFIRMED TARGET)' : 'REJECTED (PENALIZED NOISE)'}`;
      displayFloatingFeedbackToast(msg);
    }
  } catch (err) {
    console.error('Feedback error:', err);
    displayFloatingFeedbackToast(`FEEDBACK ERROR // ${err.message}`);
  }
};

window.quickModalFeedback = async function (isRelevant) {
  if (!currentInspectingTile || !currentInspectingTile.tile_id) return;
  const tileId = currentInspectingTile.tile_id;
  const logId = window.currentSearchLogId || null;

  const inputEl = document.getElementById('searchPromptInput');
  let q = (inputEl && inputEl.value && inputEl.value.trim()) || '';
  if (!q && typeof currentSearchQuery !== 'undefined' && currentSearchQuery) {
    q = currentSearchQuery;
  }
  if (!q && window.currentSearchQuery) {
    q = window.currentSearchQuery;
  }
  if (!q) {
    q = (currentInspectingTile.spot_description)
      ? currentInspectingTile.spot_description.slice(0, 60)
      : 'General Satellite Reconnaissance';
  }

  try {
    const res = await fetch(`${API_BASE}/api/v1/review/retrieval/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        search_log_id: logId ? parseInt(logId) : null,
        query_text: q,
        tile_id: tileId,
        relevant: isRelevant,
        feedback_type: isRelevant ? 'positive' : 'negative',
        analyst_id: 'CAPT. VERMA (INTEL-01)'
      })
    });
    const data = await res.json();
    if (data.status === 'success') {
      const btnAccept = document.getElementById('inspectAcceptBtn');
      const btnReject = document.getElementById('inspectRejectBtn');
      if (btnAccept && btnReject) {
        if (isRelevant) {
          btnAccept.style.background = '#10b981';
          btnAccept.style.color = '#000';
          btnReject.style.opacity = '0.5';
          if (window.playTacticalLockChirp) window.playTacticalLockChirp();
        } else {
          btnReject.style.background = '#f43f5e';
          btnReject.style.color = '#fff';
          btnAccept.style.opacity = '0.5';
          if (window.playTacticalClick) window.playTacticalClick();
        }
      }
      displayFloatingFeedbackToast(`TILE #${tileId} // ${isRelevant ? 'ACCEPTED (TARGET VERIFIED)' : 'REJECTED (NOISE CLASSIFIED)'}`);
    }
  } catch (err) {
    displayFloatingFeedbackToast(`FEEDBACK ERROR // ${err.message}`);
  }
};

window.inspectOpenInMapClicked = function () {
  if (!currentInspectingTile) return;
  const item = currentInspectingTile;
  closeTileInspect();
  openTileInMap(item.tile_id, item.centroid_lat, item.centroid_lon, item.geometry_geojson);
};

window.inspectFindSimilarClicked = function () {
  if (!currentInspectingTile) return;
  const tileId = currentInspectingTile.tile_id;
  closeTileInspect();
  discoverSimilarFromTile(tileId);
};

// ============================================================
// 9.5 GENERIC SIMILARITY DISCOVERY (PS 2.2.4)
// ============================================================
window.__lastDiscoveredTiles = [];

window.discoverSimilarFromTile = async function (tileId) {
  if (!tileId) return;

  const chatFeed = document.getElementById('retrievalChatFeed') || document.getElementById('chatFeed');
  if (chatFeed) {
    // 1. Post user prompt in chat
    const userMsg = document.createElement('div');
    userMsg.className = 'chat-message user';
    userMsg.innerHTML = `
      <div class="chat-avatar">
        <svg class="ui-icon icon-md" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
      </div>
      <div class="chat-bubble">
        Find visually and semantically similar sites like <code>${tileId}</code> across the entire archive
      </div>
    `;
    chatFeed.appendChild(userMsg);

    // 2. Post loading placeholder
    const loadingMsg = document.createElement('div');
    loadingMsg.className = 'chat-message assistant';
    loadingMsg.id = 'discoveryLoadingIndicator';
    loadingMsg.innerHTML = `
      <div class="chat-avatar">
        <svg class="ui-icon icon-md" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M5 3v4"/><path d="M19 17v4"/><path d="M3 5h4"/><path d="M17 19h4"/><path d="M15 9l5-5"/><path d="M9 15l-5 5"/></svg>
      </div>
      <div class="chat-bubble" style="display: flex; align-items: center; gap: 10px;">
        <div class="status-dot"></div>
        <span>Searching Qdrant multi-region vector space for twins of <strong>${tileId}</strong> (zero re-encoding)...</span>
      </div>
    `;
    chatFeed.appendChild(loadingMsg);
    chatFeed.scrollTop = chatFeed.scrollHeight;
  }

  try {
    const res = await fetch(`${API_BASE}/api/v1/discover/${encodeURIComponent(tileId)}?top_k=10`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Discovery HTTP error ${res.status}`);
    }

    const rawResults = await res.json();

    // Filter results to match the seed's sensor domain so Sentinel doesn't pull in random Maxar tiles from another cluster
    const isSeedMaxar = tileId.toLowerCase().includes('maxar');
    let domainMatches = (rawResults || []).filter(item => {
      const isItemMaxar = (item.sensor && item.sensor.toLowerCase().includes('maxar')) || item.tile_id.toLowerCase().includes('maxar');
      return isSeedMaxar ? isItemMaxar : !isItemMaxar;
    });
    if (!domainMatches || domainMatches.length === 0) {
      domainMatches = rawResults || [];
    }

    // Keep exactly the top 5 domain-matching tiles so popup and chat message are 100% identical
    const results = domainMatches.slice(0, 5);
    window.__lastDiscoveredTiles = results;

    // Cache in universal tileMap and currentSearchResults
    window.__tileMap = window.__tileMap || new Map();
    results.forEach(r => {
      if (r && r.tile_id) window.__tileMap.set(r.tile_id, r);
      if (!currentSearchResults.some(t => t.tile_id === r.tile_id)) {
        currentSearchResults.push(r);
      }
    });

    // Remove loading indicator
    const loader = document.getElementById('discoveryLoadingIndicator');
    if (loader) loader.remove();

    // Open the rich interactive pop-up window showing the exact same 5 similar tiles
    showDiscoveryResultsModal(tileId, results);

    if (!chatFeed) {
      return;
    }

    // Render Discovery Response in Chat (as history) - exact same 5 tiles
    const msg = document.createElement('div');
    msg.className = 'chat-message assistant';

    const headerHtml = `
      <div class="retrieval-summary-header" style="background: linear-gradient(135deg, rgba(6,182,212,0.15), rgba(99,102,241,0.15)); border: 1px solid var(--border-active); border-radius: var(--radius-md); padding: 14px; margin-bottom: 14px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
          <div>
            <div style="display: flex; align-items: center; gap: 8px;">
              <svg class="ui-icon icon-cyan" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#06b6d4" stroke-width="2"><path d="M12 2a10 10 0 1 0 10 10"/><path d="M12 12 21.5 2.5"/><circle cx="12" cy="12" r="2"/></svg>
              <h4 style="font-size: 14px; font-weight: 700; color: #38bdf8; margin: 0;">Discovery: Similar Sites Found</h4>
              <span class="tag-badge" style="background: rgba(6,182,212,0.2); color: var(--accent-cyan);">PS 2.2.4</span>
            </div>
            <p style="font-size: 12px; color: var(--text-secondary); margin: 4px 0 0 0;">
              Seed: <code>${tileId}</code> &bull; <strong>${results.length} matches</strong> across multiple geographic sectors (Zero Re-Encoding)
            </p>
          </div>
          <button class="map-btn primary" onclick="openAllOnMap(window.__lastDiscoveredTiles)" style="padding: 8px 16px; font-weight: 700; background: #06b6d4; color: #000; display: inline-flex; align-items: center; gap: 6px;">
            <svg class="ui-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21"/><line x1="9" y1="3" x2="9" y2="18"/><line x1="15" y1="6" x2="15" y2="21"/></svg>
            See All Locations on Map
          </button>
        </div>
      </div>
    `;

    let gridHtml = `<div class="retrieval-results-grid">`;
    results.forEach((item, idx) => {
      const scorePct = (item.score * 100).toFixed(1);
      const thumbUrl = getTileThumbnailUrl(item);
      const dateStr = item.acquisition_date ? item.acquisition_date.split('T')[0] : 'Historical';

      const ndviVal = item.mean_ndvi !== null && item.mean_ndvi !== undefined ? item.mean_ndvi.toFixed(2) : '-';
      const ndwiVal = item.mean_ndwi !== null && item.mean_ndwi !== undefined ? item.mean_ndwi.toFixed(2) : '-';
      const ndbiVal = item.mean_ndbi !== null && item.mean_ndbi !== undefined ? item.mean_ndbi.toFixed(2) : '-';

      gridHtml += `
        <div class="result-card" style="border-color: rgba(6,182,212,0.3);">
          <div class="result-thumb-wrap">
            <img class="result-thumb-img" src="${thumbUrl}" alt="${item.tile_id}" onerror="handleTileThumbError(this, '${item.tile_id}')">
            <div class="result-score-badge" style="background: rgba(6,182,212,0.9);">${scorePct}% Match</div>
          </div>
          <div class="result-body">
            <div class="result-title-row">
              <span class="result-tile-id">#${idx + 1} &bull; ${item.tile_id}</span>
              <span class="result-date">${dateStr}</span>
            </div>
            <div class="result-spot-desc">
              ${escapeHtml(item.spot_description || 'Visually & semantically similar terrain signature.')}
            </div>
            <div class="result-spectral-chips">
              <span class="spec-chip ndvi">NDVI ${ndviVal}</span>
              <span class="spec-chip ndwi">NDWI ${ndwiVal}</span>
              <span class="spec-chip ndbi">NDBI ${ndbiVal}</span>
            </div>
            <div class="result-card-actions">
              <button class="result-action-btn" onclick="openTileInspect('${item.tile_id}')">
                <svg class="ui-icon icon-sm icon-cyan" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#06b6d4" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
                Inspect
              </button>
              <button class="result-action-btn" onclick="openTileInMap('${item.tile_id}', ${item.centroid_lat || 'null'}, ${item.centroid_lon || 'null'}, ${JSON.stringify(item.geometry_geojson || null).replace(/"/g, '&quot;')})">
                <svg class="ui-icon icon-sm" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21"/><line x1="9" y1="3" x2="9" y2="18"/><line x1="15" y1="6" x2="15" y2="21"/></svg>
                Map
              </button>
              <button class="result-action-btn" onclick="discoverSimilarFromTile('${item.tile_id}')" style="color: #38bdf8; font-weight: 600;" title="Recursive discovery from this match">
                <svg class="ui-icon icon-sm icon-cyan" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#06b6d4" stroke-width="2"><path d="M12 2a10 10 0 1 0 10 10"/><path d="M12 12 21.5 2.5"/><circle cx="12" cy="12" r="2"/></svg>
                Similar
              </button>
            </div>
          </div>
        </div>
      `;
    });
    gridHtml += `</div>`;

    msg.innerHTML = `
      <div class="chat-avatar">
        <svg class="ui-icon icon-md" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M5 3v4"/><path d="M19 17v4"/><path d="M3 5h4"/><path d="M17 19h4"/><path d="M15 9l5-5"/><path d="M9 15l-5 5"/></svg>
      </div>
      <div class="chat-bubble" style="width: 100%;">
        ${headerHtml}
        ${gridHtml}
      </div>
    `;

    chatFeed.appendChild(msg);
    chatFeed.scrollTop = chatFeed.scrollHeight;

  } catch (err) {
    const loader = document.getElementById('discoveryLoadingIndicator');
    if (loader) loader.remove();

    if (chatFeed) {
      const errMsg = document.createElement('div');
      errMsg.className = 'chat-message assistant';
      errMsg.innerHTML = `
        <div class="chat-avatar" style="background: rgba(244, 63, 94, 0.2); border-color: rgba(244, 63, 94, 0.4); color: #f43f5e;">
          <svg class="ui-icon icon-md" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#f43f5e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
        </div>
        <div class="chat-bubble" style="color: #f87171;">
          Discovery Error: ${escapeHtml(err.message)}
        </div>
      `;
      chatFeed.appendChild(errMsg);
    } else {
      alert(`Discovery Error: ${err.message}`);
    }
  }
};

window.showDiscoveryResultsModal = function (seedTileId, results) {
  let modal = document.getElementById('discoveryModalOverlay');
  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'discoveryModalOverlay';
    modal.className = 'discovery-modal-overlay';
    document.body.appendChild(modal);
  }

  // Register in tile map
  window.__tileMap = window.__tileMap || new Map();
  (results || []).forEach(t => {
    if (t && t.tile_id) window.__tileMap.set(t.tile_id, t);
  });
  window.__lastDiscoveredTiles = results || [];

  const displayTiles = (results || []).slice(0, 5);

  const resultCardsHtml = displayTiles.map((item, idx) => {
    const hasScore = item.score !== undefined && item.score !== null;
    const scorePct = hasScore ? (item.score * 100).toFixed(1) + '% Match' : 'Cluster Member';
    const thumbUrl = getTileThumbnailUrl(item);
    const dateStr = item.acquisition_date ? item.acquisition_date.split('T')[0] : 'Unknown Date';
    const ndviVal = item.mean_ndvi !== null && item.mean_ndvi !== undefined ? item.mean_ndvi.toFixed(2) : '-';
    const ndwiVal = item.mean_ndwi !== null && item.mean_ndwi !== undefined ? item.mean_ndwi.toFixed(2) : '-';
    const ndbiVal = item.mean_ndbi !== null && item.mean_ndbi !== undefined ? item.mean_ndbi.toFixed(2) : '-';
    const lat = item.centroid_lat ? item.centroid_lat.toFixed(4) : null;
    const lon = item.centroid_lon ? item.centroid_lon.toFixed(4) : null;

    return `
      <div class="result-card" style="border-color: rgba(6,182,212,0.35); background: rgba(13, 18, 31, 0.95); display: flex; flex-direction: column; min-width: 0;">
        <div class="result-thumb-wrap" style="height: 160px; position: relative;">
          <img class="result-thumb-img" src="${thumbUrl}" alt="${item.tile_id}" onerror="handleTileThumbError(this, '${item.tile_id}')">
          <div class="result-score-badge" style="background: rgba(6,182,212,0.95); color: #000; font-weight: 800;">${scorePct}</div>
          <div style="position: absolute; bottom: 8px; left: 8px; font-size: 10px; font-family: 'JetBrains Mono', monospace; background: rgba(0,0,0,0.75); padding: 2px 6px; border-radius: 4px; color: #38bdf8;">
            ${lat ? lat + '° N, ' + lon + '° E' : 'WGS84 Coordinates'}
          </div>
        </div>
        <div class="result-body" style="display: flex; flex-direction: column; flex: 1; padding: 14px;">
          <div class="result-title-row">
            <span class="result-tile-id" style="font-size: 12px; font-weight: 700;" title="${item.tile_id}">#${idx + 1} &bull; ${item.tile_id.length > 22 ? item.tile_id.slice(0, 20) + '...' : item.tile_id}</span>
            <span class="result-date" style="display: inline-flex; align-items: center; gap: 4px;">
              <svg class="ui-icon icon-sm" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>
              ${dateStr}
            </span>
          </div>
          <div class="result-spot-desc" style="font-size: 11px; margin: 6px 0; color: var(--text-secondary); line-height: 1.4;">
            ${escapeHtml(item.spot_description || 'High-dimensional visual & semantic twin.')}
          </div>
          <div class="result-spectral-chips" style="margin-bottom: 10px;">
            <span class="spec-chip ndvi">NDVI ${ndviVal}</span>
            <span class="spec-chip ndwi">NDWI ${ndwiVal}</span>
            <span class="spec-chip ndbi">NDBI ${ndbiVal}</span>
          </div>
          <div class="result-card-actions" style="margin-top: auto; display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
            <button class="map-btn primary" onclick="openTileInMap('${item.tile_id}', ${item.centroid_lat || 'null'}, ${item.centroid_lon || 'null'}, ${JSON.stringify(item.geometry_geojson || null).replace(/"/g, '&quot;')})" style="padding: 7px 8px; font-size: 11px; font-weight: 700; background: linear-gradient(135deg, #06b6d4, #6366f1); color: #fff; display: inline-flex; align-items: center; justify-content: center; gap: 4px;">
              <svg class="ui-icon icon-sm" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21"/><line x1="9" y1="3" x2="9" y2="18"/><line x1="15" y1="6" x2="15" y2="21"/></svg>
              See Map
            </button>
            <button class="map-btn" onclick="openTileInspect('${item.tile_id}')" style="padding: 7px 8px; font-size: 11px; font-weight: 600; display: inline-flex; align-items: center; justify-content: center; gap: 4px;">
              <svg class="ui-icon icon-sm icon-cyan" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#06b6d4" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
              Inspect
            </button>
          </div>
        </div>
      </div>
    `;
  }).join('');

  modal.innerHTML = `
    <div class="discovery-modal-content">
      <div class="discovery-modal-header">
        <div style="display: flex; align-items: center; gap: 10px;">
          <div style="width: 34px; height: 34px; border-radius: 8px; background: rgba(6,182,212,0.2); border: 1px solid var(--accent-cyan); display: flex; align-items: center; justify-content: center;">
            <svg class="ui-icon icon-md icon-cyan" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#06b6d4" stroke-width="2"><path d="M12 2a10 10 0 1 0 10 10"/><path d="M12 12 21.5 2.5"/><circle cx="12" cy="12" r="2"/></svg>
          </div>
          <div>
            <h3 style="font-size: 16px; font-weight: 800; color: #fff; margin: 0;">Top ${displayTiles.length} Discovered Similar Sites</h3>
            <span style="font-size: 12px; color: var(--text-muted);">Unsupervised High-Dimensional Similarity Search (PS 2.2.4 &bull; Zero GPU Re-Encoding)</span>
          </div>
        </div>
        <div style="display: flex; align-items: center; gap: 12px;">
          <button class="map-btn primary" onclick="openAllOnMap(window.__lastDiscoveredTiles)" style="padding: 8px 16px; font-weight: 700; background: #06b6d4; color: #000; display: inline-flex; align-items: center; gap: 6px;">
            <svg class="ui-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21"/><line x1="9" y1="3" x2="9" y2="18"/><line x1="15" y1="6" x2="15" y2="21"/></svg>
            See All ${displayTiles.length} on Map
          </button>
          <button class="modal-close-btn" onclick="closeDiscoveryModal()">&times;</button>
        </div>
      </div>
      <div class="discovery-modal-body">
        <div class="discovery-seed-box">
          <svg class="ui-icon icon-cyan" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#06b6d4" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>
          <div>
            <div style="font-size: 13px; font-weight: 700; color: #38bdf8;">Seed Tile: <code>${escapeHtml(seedTileId)}</code></div>
            <div style="font-size: 11px; color: var(--text-secondary); margin-top: 2px;">
              Directly queried 512-dim RemoteCLIP embedding in Qdrant &bull; Identified top ${displayTiles.length} visual and terrain signature twins across archive sectors.
            </div>
          </div>
        </div>
        <div class="discovery-results-grid">
          ${resultCardsHtml}
        </div>
      </div>
    </div>
  `;

  modal.style.zIndex = '3000';
  modal.style.display = 'flex';
};

window.closeDiscoveryModal = function () {
  const modal = document.getElementById('discoveryModalOverlay');
  if (modal) modal.style.display = 'none';
};

window.openAllOnMap = function (tiles) {
  if (!tiles || tiles.length === 0) {
    alert('No tiles available to plot on map.');
    return;
  }

  // Store discovery pins for map handoff
  localStorage.setItem('discovery_pins', JSON.stringify(tiles));

  // If on map page already:
  if (document.getElementById('leafletMap')) {
    renderDiscoveryPinsOnMap(tiles);
    return;
  }

  // Navigate to map view
  window.location.href = `/?discovery=true`;
};

window.renderDiscoveryPinsOnMap = function (tiles) {
  if (!map) {
    // Retry if map is still initializing
    setTimeout(() => window.renderDiscoveryPinsOnMap(tiles), 300);
    return;
  }
  if (!tiles || tiles.length === 0) return;

  if (discoveryMapLayerGroup) {
    try { map.removeLayer(discoveryMapLayerGroup); } catch (e) { }
  }
  discoveryMapLayerGroup = L.featureGroup().addTo(map);

  const bounds = L.latLngBounds();
  let validCount = 0;

  tiles.forEach((item, idx) => {
    const lat = parseFloat(item.centroid_lat);
    const lon = parseFloat(item.centroid_lon);
    if (isNaN(lat) || isNaN(lon)) return;

    validCount++;
    bounds.extend([lat, lon]);

    const hasScore = item.score !== undefined && item.score !== null;
    const scoreVal = hasScore ? (item.score * 100).toFixed(0) + '%' : 'Member';
    const scoreDisplay = hasScore ? (item.score * 100).toFixed(1) + '%' : 'Cluster Medoid / Member';

    const markerIcon = L.divIcon({
      className: 'discovery-map-pin',
      html: `
        <div style="position: relative; cursor: pointer; text-align: center;">
          <div style="background: linear-gradient(135deg, #06b6d4, #6366f1); color: #fff; font-weight: 700; font-size: 11px; padding: 3px 8px; border-radius: 12px; border: 1px solid #fff; box-shadow: 0 0 14px rgba(6,182,212,0.85); white-space: nowrap;">
            #${idx + 1} (${scoreVal})
          </div>
          <div style="width: 2px; height: 10px; background: #06b6d4; margin: 0 auto;"></div>
        </div>
      `,
      iconSize: [80, 32],
      iconAnchor: [40, 32]
    });

    const marker = L.marker([lat, lon], { icon: markerIcon }).addTo(discoveryMapLayerGroup);
    const thumbUrl = getTileThumbnailUrl(item);

    marker.bindPopup(`
      <div style="font-family: 'Inter', sans-serif; font-size: 12px; color: #fff; width: 240px; padding: 4px;">
        <div style="font-weight: 700; color: #38bdf8; margin-bottom: 4px;">Match #${idx + 1} &bull; ${scoreVal}</div>
        <img src="${thumbUrl}" style="width: 100%; height: 120px; object-fit: cover; border-radius: 4px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 6px;" onerror="handleTileThumbError(this, '${item.tile_id}')">
        <div style="font-size: 11px; color: #cbd5e1; margin-bottom: 8px;">
          <strong>Tile:</strong> <code>${item.tile_id}</code><br>
          <strong>Match:</strong> ${scoreDisplay}<br>
          <strong>Centroid:</strong> ${lat.toFixed(4)}° N, ${lon.toFixed(4)}° E
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px;">
          <button onclick="openTileInspect('${item.tile_id}')" style="background: rgba(255,255,255,0.1); color: #fff; border: 1px solid rgba(255,255,255,0.2); padding: 5px 8px; border-radius: 4px; font-weight: 600; cursor: pointer; font-size: 11px;">
            Inspect
          </button>
          <button onclick="discoverSimilarFromTile('${item.tile_id}')" style="background: #06b6d4; color: #000; border: none; padding: 5px 8px; border-radius: 4px; font-weight: 700; cursor: pointer; font-size: 11px;">
            Find Similar
          </button>
        </div>
      </div>
    `);

    // Draw footprint rectangle if available
    if (item.geometry_geojson) {
      try {
        const geom = typeof item.geometry_geojson === 'string' ? JSON.parse(item.geometry_geojson) : item.geometry_geojson;
        if (geom && (geom.coordinates || geom.type)) {
          L.geoJSON(geom, {
            style: {
              color: '#06b6d4',
              weight: 2,
              fillColor: '#06b6d4',
              fillOpacity: 0.15,
              dashArray: '4, 4'
            }
          }).addTo(discoveryMapLayerGroup);
        }
      } catch (e) { }
    }
  });

  if (validCount > 0 && bounds.isValid()) {
    map.fitBounds(bounds, { padding: [80, 80], maxZoom: 15 });
  }
};

window.openTileInMap = function (tileId, centroidLat, centroidLon, geometryGeoJson) {
  // If not on map page, navigate to map page with query params
  if (!document.getElementById('leafletMap')) {
    const params = new URLSearchParams();
    params.set('tile_id', tileId);
    if (centroidLat) params.set('lat', centroidLat);
    if (centroidLon) params.set('lon', centroidLon);
    window.location.href = `/?${params.toString()}`;
    return;
  }

  // If on map page with view switching:
  if (typeof switchPage === 'function') {
    switchPage('map');
  }

  if (!map) return;

  // Remove previous highlight
  if (mapTileHighlightLayer) {
    map.removeLayer(mapTileHighlightLayer);
    mapTileHighlightLayer = null;
  }

  // Draw highlighted tile polygon
  if (geometryGeoJson && geometryGeoJson.coordinates) {
    mapTileHighlightLayer = L.geoJSON(geometryGeoJson, {
      style: {
        color: '#f59e0b',
        weight: 3,
        opacity: 1,
        fillColor: '#f59e0b',
        fillOpacity: 0.35,
        dashArray: '6, 6'
      }
    }).addTo(map);

    mapTileHighlightLayer.bindPopup(`
      <div class="popup-title">Retrieved Tile: ${tileId}</div>
      <div class="popup-stat">Centroid: ${centroidLat ? centroidLat.toFixed(4) : ''}° N, ${centroidLon ? centroidLon.toFixed(4) : ''}° E</div>
    `).openPopup();

    map.fitBounds(mapTileHighlightLayer.getBounds(), { maxZoom: 15, padding: [50, 50] });
  } else if (centroidLat && centroidLon) {
    map.setView([centroidLat, centroidLon], 14);
    mapTileHighlightLayer = L.circleMarker([centroidLat, centroidLon], {
      radius: 10,
      color: '#f59e0b',
      fillColor: '#f59e0b',
      fillOpacity: 0.8
    }).addTo(map);
    mapTileHighlightLayer.bindPopup(`<b>Retrieved Tile:</b> ${tileId}`).openPopup();
  }
};

function checkUrlParamsForTileHighlight() {
  const params = new URLSearchParams(window.location.search);
  const isDiscovery = params.get('discovery');
  const tileId = params.get('tile_id');
  const lat = parseFloat(params.get('lat'));
  const lon = parseFloat(params.get('lon'));

  if (isDiscovery) {
    try {
      const cached = localStorage.getItem('discovery_pins');
      if (cached) {
        const tiles = JSON.parse(cached);
        setTimeout(() => {
          renderDiscoveryPinsOnMap(tiles);
        }, 500);
        return;
      }
    } catch (e) { }
  }

  if (tileId && lat && lon && map) {
    setTimeout(() => {
      openTileInMap(tileId, lat, lon, null);
    }, 500);
  }
}

function escapeHtml(str) {
  if (!str) return '';
  return str.toString().replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}



// ============================================================
// CANOPUS THEME MODE CONTROLLER (DARK / BRIGHT LIGHT)
// ============================================================
function initThemeSystem() {
  const saved = localStorage.getItem('canopus_theme') || 'light';
  document.documentElement.setAttribute('data-theme', saved);
  updateThemeButtonUI(saved);
}
window.initThemeSystem = initThemeSystem;

function toggleThemeMode() {
  const current = document.documentElement.getAttribute('data-theme') || 'light';
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('canopus_theme', next);
  updateThemeButtonUI(next);
}
window.toggleThemeMode = toggleThemeMode;

window.updateThemeButtonUI = function (theme) {
  const btn = document.getElementById('themeToggleBtn');
  const label = document.getElementById('themeToggleLabel');
  const sunIcons = document.querySelectorAll('.theme-icon-sun');
  const moonIcons = document.querySelectorAll('.theme-icon-moon');
  if (!btn) return;
  if (theme === 'light') {
    if (label) label.innerText = 'DARK';
    sunIcons.forEach(i => i.style.display = 'none');
    moonIcons.forEach(i => i.style.display = 'inline-block');
    btn.setAttribute('title', 'Switch to Dark Mode');
  } else {
    if (label) label.innerText = 'LIGHT';
    sunIcons.forEach(i => i.style.display = 'inline-block');
    moonIcons.forEach(i => i.style.display = 'none');
    btn.setAttribute('title', 'Switch to Light Mode');
  }
};

// Immediate application upon load
if (typeof document !== 'undefined') {
  const initialTheme = localStorage.getItem('canopus_theme') || 'light';
  document.documentElement.setAttribute('data-theme', initialTheme);
}

// Update header tile count on any page
window.updateGlobalArchiveTileCount = async function updateGlobalArchiveTileCount() {
  const el = document.getElementById('archiveTileCount');
  if (!el) return;
  try {
    const res = await fetch(`${API_BASE}/api/v1/coverage`);
    if (!res.ok) return;
    const data = await res.json();
    const count = data.total_tiles || 0;
    if (window.animateNumberCount) {
      window.animateNumberCount(el, 0, count, 600, ' TILES ONLINE');
    } else {
      el.innerText = `${count} TILES ONLINE`;
    }
  } catch (err) {
    console.debug('Tile counter fetch fallback:', err);
  }
};

document.addEventListener('DOMContentLoaded', () => {
  // Initialize Bright/Dark theme
  if (typeof initThemeSystem === 'function') {
    initThemeSystem();
  }

  // Update header tile count on every page
  if (typeof updateGlobalArchiveTileCount === 'function') {
    updateGlobalArchiveTileCount();
  }

  if (document.getElementById('leafletMap')) {
    initMap();
    checkUrlParamsForTileHighlight();
    if (new URLSearchParams(window.location.search).get('openIngest')) {
      setTimeout(() => { if (typeof openIngestModal === 'function') openIngestModal(); }, 400);
    }
  }
  if (document.getElementById('retrievalChatFeed')) {
    initChatSystem();
  }
});

