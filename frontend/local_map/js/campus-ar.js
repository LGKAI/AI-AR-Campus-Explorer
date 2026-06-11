/**
 * campus-ar.js — GPS-based outdoor AR for campus buildings
 *
 * How it works:
 *  1. GPS (navigator.geolocation) → user lat/lng position
 *  2. DeviceOrientation → compass heading (alpha) + device tilt (beta)
 *  3. For each building: compute bearing + distance via Haversine
 *  5. Draw floating AR labels on canvas
 */

// Prevent double-tap to zoom on iOS Safari
document.addEventListener('dblclick', (e) => {
  e.preventDefault();
}, { passive: false });

let lastTouchEnd = 0;
document.addEventListener('touchend', (e) => {
  const now = (new Date()).getTime();
  if (now - lastTouchEnd <= 300) {
    e.preventDefault();
  }
  lastTouchEnd = now;
}, { passive: false });

// ══════════════════════════════════════════════════════════════
//  STATE
// ══════════════════════════════════════════════════════════════
let buildings     = [];  // loaded from /api/buildings
let userPos       = null; // { lat, lng, accuracy }
let compassHeading = null; // degrees from north, CW
let deviceBeta    = 90;   // tilt: 0=flat, 90=upright
let selectedBldg  = null; // currently tapped building

const HFOV = 65;  // horizontal field of view (degrees) — typical phone camera
const VFOV = 50;  // vertical FOV
const MAX_DIST = 50000000;    // 50,000 km - Allow testing from anywhere
const MIN_OPACITY_DIST = 20000000;

let canvas, ctx;
let animFrame = 0;
let compassCalibrated = false;

// ══════════════════════════════════════════════════════════════
//  GEO MATH
// ══════════════════════════════════════════════════════════════

/** Haversine distance in metres */
function haversine(lat1, lng1, lat2, lng2) {
  const R = 6371000;
  const toRad = d => d * Math.PI / 180;
  const dLat = toRad(lat2 - lat1);
  const dLng = toRad(lng2 - lng1);
  const a = Math.sin(dLat/2)**2 +
            Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLng/2)**2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

/** Compass bearing in degrees (0=north, CW) from user to building */
function getBearing(lat1, lng1, lat2, lng2) {
  const toRad = d => d * Math.PI / 180;
  const dLng  = toRad(lng2 - lng1);
  const y = Math.sin(dLng) * Math.cos(toRad(lat2));
  const x = Math.cos(toRad(lat1)) * Math.sin(toRad(lat2))
          - Math.sin(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.cos(dLng);
  return (Math.atan2(y, x) * 180 / Math.PI + 360) % 360;
}

/** Format distance for display */
function fmtDist(m) {
  return m < 1000 ? `${Math.round(m)} m` : `${(m / 1000).toFixed(1)} km`;
}

// ══════════════════════════════════════════════════════════════
//  SCREEN PROJECTION
// ══════════════════════════════════════════════════════════════

/**
 * Project a building's GPS to a canvas position.
 * Returns { x, y, dist, opacity, scale } or null if outside FOV.
 */
function project(building) {
  if (!userPos || compassHeading === null) return null;

  const dist = haversine(userPos.lat, userPos.lng, building.lat, building.lng);
  if (dist > MAX_DIST || dist < 1) return null;

  const bearing = getBearing(userPos.lat, userPos.lng, building.lat, building.lng);

  // Horizontal angle relative to camera direction
  let hAngle = bearing - compassHeading;
  // Normalize to [-180, 180]
  while (hAngle >  180) hAngle -= 360;
  while (hAngle < -180) hAngle += 360;

  // Is it within horizontal FOV?
  if (Math.abs(hAngle) > HFOV * 0.7) return null;

  const W = canvas.width;
  const H = canvas.height;

  // Screen X: linear mapping from angle to pixels
  const x = W / 2 + (hAngle / (HFOV / 2)) * (W / 2);

  // Beta pitch: invert standard so tilting phone down moves labels UP
  const clampedBeta = Math.max(30, Math.min(150, deviceBeta));
  const horizonY = H / 2 + ((clampedBeta - 90) / (VFOV / 2)) * (H / 2);

  // Labels float slightly above the horizon
  const elevationOffset = -30 - Math.max(0, (150 - dist) / 5);
  const y = horizonY + elevationOffset;

  // Opacity fade with distance
  const opacity = dist > MIN_OPACITY_DIST
    ? 1 - (dist - MIN_OPACITY_DIST) / (MAX_DIST - MIN_OPACITY_DIST)
    : 1;

  // Scale inversely with distance (perspective)
  const scale = Math.max(0.4, Math.min(1.2, 80 / Math.sqrt(dist)));

  return { x, y, dist, opacity, scale, bearing, hAngle };
}

// ══════════════════════════════════════════════════════════════
//  CANVAS RENDERING
// ══════════════════════════════════════════════════════════════
function setupCanvas() {
  canvas = document.getElementById('canvas');
  ctx    = canvas.getContext('2d');
  resize();
  window.addEventListener('resize', resize);
}

function resize() {
  canvas.width  = window.innerWidth;
  canvas.height = window.innerHeight;
}

function renderFrame() {
  requestAnimationFrame(renderFrame);
  animFrame++;

  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // Vignette
  const vig = ctx.createRadialGradient(W/2, H/2, H*0.2, W/2, H/2, H*0.75);
  vig.addColorStop(0, 'rgba(0,0,0,0)');
  vig.addColorStop(1, 'rgba(0,0,0,0.42)');
  ctx.fillStyle = vig;
  ctx.fillRect(0, 0, W, H);

  // Draw horizon line (subtle)
  if (compassHeading !== null) {
    const clampedBeta = Math.max(30, Math.min(150, deviceBeta));
    const horizonY = H / 2 + ((clampedBeta - 90) / (VFOV / 2)) * (H / 2);
    ctx.beginPath();
    ctx.moveTo(0, horizonY);
    ctx.lineTo(W, horizonY);
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  if (buildings.length === 0) return;

  // Compute projections, sort by distance (far first → near on top)
  let projected = buildings
    .map(b => ({ b, p: project(b) }))
    .filter(({ p }) => p !== null)
    .sort((a, b) => b.p.dist - a.p.dist);

  // If a building is selected, ONLY show that building's label
  if (selectedBldg) {
    projected = projected.filter(({ b }) => b.id === selectedBldg.id);
  }

  // Draw connector lines first (below labels)
  for (const { b, p } of projected) {
    drawConnectorLine(b, p);
  }

  // Draw labels
  for (const { b, p } of projected) {
    drawBuildingLabel(b, p);
  }

  // Compass rose (bottom-right corner)
  drawCompassRose(W - 52, H - (document.getElementById('bottom-panel').offsetHeight || 0) - 72);
}

// ── Connector line from label to horizon ────────────────────
function drawConnectorLine(bldg, proj) {
  const color = bldg.color || '#00d4ff';
  const H = canvas.height;
  const clampedBeta = Math.max(30, Math.min(150, deviceBeta));
  const horizonY = H / 2 + ((clampedBeta - 90) / (VFOV / 2)) * (H / 2);

  const alpha = proj.opacity * 0.35;
  ctx.beginPath();
  ctx.moveTo(proj.x, proj.y + 8 * proj.scale);
  ctx.lineTo(proj.x, Math.min(horizonY, proj.y + 80));
  ctx.strokeStyle = color + Math.round(alpha * 255).toString(16).padStart(2, '0');
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 6]);
  ctx.stroke();
  ctx.setLineDash([]);

  // Dot at horizon
  ctx.beginPath();
  ctx.arc(proj.x, Math.min(horizonY, proj.y + 80), 3, 0, Math.PI * 2);
  ctx.fillStyle = color + Math.round(alpha * 2 * 255).toString(16).padStart(2, '0');
  ctx.fill();
}

// ── Building label card ──────────────────────────────────────
function drawBuildingLabel(bldg, proj) {
  const color   = bldg.color || '#00d4ff';
  const icon    = bldg.icon  || '🏢';
  const label   = bldg.name;
  const distStr = fmtDist(proj.dist);

  const scale   = proj.scale;
  const opacity = proj.opacity;
  const cx      = proj.x;
  const cy      = proj.y;

  // Font sizes
  const nameFontSize = Math.round(15 * scale);
  const distFontSize = Math.round(11 * scale);
  const iconSize     = Math.round(20 * scale);

  // Measure text
  ctx.font = `bold ${nameFontSize}px Inter, -apple-system, sans-serif`;
  const nameW = ctx.measureText(label).width;
  ctx.font = `${distFontSize}px Inter, sans-serif`;
  const distW = ctx.measureText(`📍 ${distStr}`).width;

  const padX = 14 * scale;
  const padY = 10 * scale;
  const boxW = Math.max(nameW, distW) + padX * 2 + iconSize + 8 * scale;
  const boxH = (nameFontSize + distFontSize + padY * 2 + 4 * scale);

  const bx = cx - boxW / 2;
  const by = cy - boxH / 2;

  const isSelected = selectedBldg?.id === bldg.id;

  // Glow
  if (isSelected) {
    ctx.shadowColor = color;
    ctx.shadowBlur  = 20;
  }

  // Card background
  const hexAlpha = Math.round(opacity * (isSelected ? 0.92 : 0.78) * 255).toString(16).padStart(2, '0');
  ctx.fillStyle = `rgba(7,11,23,${opacity * 0.88})`;
  roundRect(ctx, bx, by, boxW, boxH, 10 * scale);
  ctx.fill();

  // Border
  const borderAlpha = Math.round(opacity * (isSelected ? 0.9 : 0.4) * 255).toString(16).padStart(2, '0');
  ctx.strokeStyle = color + borderAlpha;
  ctx.lineWidth   = isSelected ? 2 : 1.5;
  roundRect(ctx, bx, by, boxW, boxH, 10 * scale);
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Icon
  const iconX = bx + padX;
  const iconY = by + padY + nameFontSize * 0.5;
  ctx.font = `${iconSize}px Arial`;
  ctx.textBaseline = 'middle';
  ctx.fillStyle = `rgba(255,255,255,${opacity})`;
  ctx.fillText(icon, iconX, iconY);

  const textX = iconX + iconSize + 8 * scale;

  // Building name
  ctx.font = `bold ${nameFontSize}px Inter, -apple-system, sans-serif`;
  ctx.fillStyle = `rgba(255,255,255,${opacity})`;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  ctx.fillText(label, textX, by + padY + nameFontSize * 0.55);

  // Distance
  ctx.font = `${distFontSize}px Inter, sans-serif`;
  ctx.fillStyle = color + Math.round(opacity * 220).toString(16).padStart(2, '0');
  ctx.fillText(`📍 ${distStr}`, textX, by + padY + nameFontSize + distFontSize * 0.6 + 4 * scale);

  // Pulse ring when selected
  if (isSelected) {
    const pulse = 0.5 + 0.5 * Math.sin(animFrame / 30 * Math.PI);
    ctx.beginPath();
    ctx.arc(cx, cy, (boxW / 2 + 10) * pulse, 0, Math.PI * 2);
    ctx.strokeStyle = color + Math.round(pulse * 0.3 * 255).toString(16).padStart(2, '0');
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

/** Helper: rounded rect path */
function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.arcTo(x + w, y, x + w, y + r, r);
  ctx.lineTo(x + w, y + h - r);
  ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
  ctx.lineTo(x + r, y + h);
  ctx.arcTo(x, y + h, x, y + h - r, r);
  ctx.lineTo(x, y + r);
  ctx.arcTo(x, y, x + r, y, r);
  ctx.closePath();
}

// ── Compass rose ─────────────────────────────────────────────
function drawCompassRose(cx, cy) {
  const r = 22;
  const heading = compassHeading ?? 0;

  ctx.save();
  ctx.globalAlpha = 0.7;

  // Background circle
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(7,11,23,0.7)';
  ctx.fill();
  ctx.strokeStyle = 'rgba(255,255,255,0.15)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // N/S needles
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(-heading * Math.PI / 180);

  // North (red)
  ctx.beginPath();
  ctx.moveTo(0, -r + 4);
  ctx.lineTo(-5, 4);
  ctx.lineTo(5, 4);
  ctx.closePath();
  ctx.fillStyle = '#ff4757';
  ctx.fill();

  // South (white)
  ctx.beginPath();
  ctx.moveTo(0, r - 4);
  ctx.lineTo(-5, -4);
  ctx.lineTo(5, -4);
  ctx.closePath();
  ctx.fillStyle = 'rgba(255,255,255,0.5)';
  ctx.fill();

  ctx.restore();

  // N label
  ctx.font = 'bold 9px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = '#ff4757';
  ctx.fillText('N', cx, cy - r - 8);

  ctx.globalAlpha = 1;
  ctx.restore();
}

// ══════════════════════════════════════════════════════════════
//  GPS
// ══════════════════════════════════════════════════════════════
function startGPS() {
  if (!navigator.geolocation) {
    setGPSPill('error', '❌ GPS không hỗ trợ');
    return;
  }

  setGPSPill('warn', '📍 Đang lấy GPS...');

  navigator.geolocation.watchPosition(
    (pos) => {
      userPos = {
        lat: pos.coords.latitude,
        lng: pos.coords.longitude,
        accuracy: pos.coords.accuracy,
      };
      const acc = Math.round(pos.coords.accuracy);
      if (acc <= 10) {
        setGPSPill('good', `📍 ±${acc}m`);
      } else if (acc <= 30) {
        setGPSPill('warn', `📍 ±${acc}m`);
      } else {
        setGPSPill('warn', `📍 ±${acc}m (kém)`);
      }
      updateNearbyList();
    },
    (err) => {
      setGPSPill('error', '❌ GPS thất bại');
      console.warn('GPS error:', err);
    },
    { enableHighAccuracy: true, timeout: 15000, maximumAge: 2000 }
  );
}

function setGPSPill(cls, text) {
  const el = document.getElementById('gps-pill');
  el.className = `hud-pill ${cls}`;
  el.textContent = text;
}

// ══════════════════════════════════════════════════════════════
//  COMPASS / ORIENTATION
// ══════════════════════════════════════════════════════════════
async function startCompass() {
  if (typeof DeviceOrientationEvent === 'undefined') return;

  if (typeof DeviceOrientationEvent.requestPermission === 'function') {
    try {
      const perm = await DeviceOrientationEvent.requestPermission();
      if (perm !== 'granted') return;
    } catch { return; }
  }

  // Try absolute orientation first (Android Chrome)
  const onAbsolute = (e) => {
    if (e.alpha === null) return;
    compassHeading = (360 - e.alpha + 360) % 360;
    updateCompassUI();
    compassCalibrated = true;
    document.getElementById('cal-hint').style.display = 'none';
    if (e.beta !== null) deviceBeta = e.beta;
  };

  const onRelative = (e) => {
    if (e.webkitCompassHeading !== undefined && e.webkitCompassHeading !== null) {
      compassHeading = e.webkitCompassHeading;
      compassCalibrated = true;
      document.getElementById('cal-hint').style.display = 'none';
    } else if (e.alpha !== null) {
      compassHeading = (360 - e.alpha + 360) % 360;
    }
    if (e.beta !== null) deviceBeta = e.beta;
    updateCompassUI();
  };

  window.addEventListener('deviceorientationabsolute', onAbsolute, true);
  window.addEventListener('deviceorientation', onRelative, true);

  // Show calibration hint after 3s if still not calibrated
  setTimeout(() => {
    if (!compassCalibrated) {
      document.getElementById('cal-hint').style.display = 'block';
    }
  }, 3000);
}

function updateCompassUI() {
  if (compassHeading === null) return;
  document.getElementById('compass-pill').textContent =
    `🧭 ${Math.round(compassHeading)}°`;
}

// ══════════════════════════════════════════════════════════════
//  CAMERA
// ══════════════════════════════════════════════════════════════
async function startCamera() {
  const video = document.getElementById('video');
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 } },
      audio: false,
    });
    video.srcObject = stream;
    await new Promise(r => video.onloadedmetadata = r);
    await video.play();
    return true;
  } catch (e) {
    console.error('Camera:', e);
    return false;
  }
}

// ══════════════════════════════════════════════════════════════
//  NEARBY LIST (bottom chips)
// ══════════════════════════════════════════════════════════════
function updateNearbyList() {
  if (!userPos || buildings.length === 0) return;
  const list = document.getElementById('nearby-list');

  const sorted = buildings
    .map(b => ({ b, dist: haversine(userPos.lat, userPos.lng, b.lat, b.lng) }))
    .sort((a, b) => a.dist - b.dist)
    .slice(0, 15);

  list.innerHTML = '';
  for (const { b, dist } of sorted) {
    const chip = document.createElement('div');
    chip.className = 'nearby-chip' + (selectedBldg?.id === b.id ? ' highlighted' : '');
    chip.dataset.id = b.id;
    chip.innerHTML = `<span>${b.icon || '🏢'}</span><span>${b.name}</span><span style="color:var(--text-muted);font-size:10px;">${fmtDist(dist)}</span>`;
    chip.addEventListener('click', () => selectBuilding(b));
    list.appendChild(chip);
  }
}

// ══════════════════════════════════════════════════════════════
//  BUILDING SELECTION / DETAIL CARD
// ══════════════════════════════════════════════════════════════
function selectBuilding(bldg) {
  selectedBldg = selectedBldg?.id === bldg.id ? null : bldg;

  const card  = document.getElementById('detail-card');
  const panel = document.getElementById('bottom-panel');

  if (!selectedBldg) {
    card.style.display  = 'none';
    panel.style.display = 'block';
    return;
  }

  const dist = userPos
    ? haversine(userPos.lat, userPos.lng, bldg.lat, bldg.lng)
    : null;

  document.getElementById('detail-icon').textContent = bldg.icon || '🏢';
  document.getElementById('detail-icon').style.background =
    (bldg.color || '#00d4ff') + '22';
  document.getElementById('detail-icon').style.border =
    `1px solid ${bldg.color || '#00d4ff'}44`;

  document.getElementById('detail-name').textContent = bldg.name;
  document.getElementById('detail-desc').textContent = bldg.description || '';
  document.getElementById('detail-dist').textContent = dist
    ? `📍 ${fmtDist(dist)}`
    : '';

  // ── POI chips ──────────────────────────────────────────────
  const poiSection = document.getElementById('detail-poi');
  const poiChips   = document.getElementById('poi-chips');
  const poi = bldg.poi || [];

  if (poi.length > 0) {
    poiChips.innerHTML = '';
    for (const p of poi) {
      const chip = document.createElement('button');
      chip.style.cssText = `
        display:inline-flex; align-items:center; gap:6px;
        padding:7px 12px;
        background:rgba(255,255,255,0.05);
        border:1px solid rgba(255,255,255,0.12);
        border-radius:99px;
        color:var(--text-secondary);
        font-size:12px; font-family:inherit;
        cursor:pointer; transition:all 0.2s;
        white-space:nowrap;
      `;
      chip.innerHTML = `${p.icon || '📍'} ${p.name}`;
      chip.addEventListener('mouseenter', () => {
        chip.style.background = 'rgba(0,212,255,0.1)';
        chip.style.borderColor = 'rgba(0,212,255,0.35)';
        chip.style.color = 'var(--text-primary)';
      });
      chip.addEventListener('mouseleave', () => {
        chip.style.background = 'rgba(255,255,255,0.05)';
        chip.style.borderColor = 'rgba(255,255,255,0.12)';
        chip.style.color = 'var(--text-secondary)';
      });
      poiChips.appendChild(chip);
    }
    poiSection.style.display = 'block';
  } else {
    poiSection.style.display = 'none';
  }

  panel.style.display = 'none';
  card.style.display  = 'block';
  updateNearbyList();
}


// ══════════════════════════════════════════════════════════════
//  CANVAS TAP → SELECT BUILDING
// ══════════════════════════════════════════════════════════════
function setupCanvasTap() {
  canvas.style.pointerEvents = 'auto';
  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const tapX = e.clientX - rect.left;
    const tapY = e.clientY - rect.top;

    // Check if tap is near any projected building label
    const HIT_RADIUS = 60;
    let closest = null;
    let closestDist = HIT_RADIUS;

    for (const bldg of buildings) {
      const proj = project(bldg);
      if (!proj) continue;
      const dx = tapX - proj.x;
      const dy = tapY - proj.y;
      const d  = Math.sqrt(dx * dx + dy * dy);
      if (d < closestDist) {
        closestDist = d;
        closest = bldg;
      }
    }

    if (closest) {
      selectBuilding(closest);
    } else if (selectedBldg) {
      // Tap empty space → deselect
      selectBuilding(selectedBldg); // toggles off
    }
  });
}

// ══════════════════════════════════════════════════════════════
//  INIT
// ══════════════════════════════════════════════════════════════
async function main() {
  // Load buildings
  try {
    const res = await fetch('/api/buildings');
    const data = await res.json();
    buildings = data.buildings || [];
  } catch (e) {
    buildings = [];
  }

  setupCanvas();

  // ── Launch button ─────────────────────────────────────
  document.getElementById('btn-launch').addEventListener('click', async () => {
    const btn = document.getElementById('btn-launch');
    btn.textContent = '⏳ Đang bật...';
    btn.disabled    = true;

    // Must request compass permission FIRST before any awaits delay the user gesture
    await startCompass();

    const camOk = await startCamera();
    if (!camOk) {
      btn.textContent = '❌ Không bật được camera';
      btn.disabled    = false;
      return;
    }

    startGPS();

    // Hide splash
    document.getElementById('splash').style.display = 'none';

    // Show UI
    document.getElementById('top-hud').style.display = 'flex';

    if (buildings.length === 0) {
      document.getElementById('no-buildings').style.display = 'flex';
    } else {
      document.getElementById('bottom-panel').style.display = 'block';
    }

    setupCanvasTap();

    // Start render loop
    renderFrame();
  });

  // ── Dismiss no-buildings ───────────────────────────────
  document.getElementById('btn-dismiss-notice').addEventListener('click', () => {
    document.getElementById('no-buildings').style.display = 'none';
    document.getElementById('bottom-panel').style.display = 'block';
    renderFrame();
  });

  // ── Detail card: close ─────────────────────────────────
  document.getElementById('btn-close-detail').addEventListener('click', () => {
    selectBuilding(selectedBldg); // toggle off
    document.getElementById('bottom-panel').style.display = 'block';
  });

  // ── Periodic nearby list update ───────────────────────
  setInterval(() => {
    updateNearbyList();
  }, 3000);
}

main();
