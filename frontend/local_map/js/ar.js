/**
 * VPS AR Navigation — ar.js
 *
 * Pipeline:
 *   Camera (getUserMedia) → video element
 *   → Canvas overlay (AR drawing)
 *   → jsQR scan every 300ms
 *   → DeviceOrientation (compass)
 *   → Dijkstra path → directional arrow
 */

// ══════════════════════════════════════════════════════════════
//  CONSTANTS & STATE
// ══════════════════════════════════════════════════════════════
const QR_SCAN_MS  = 300;  // scan interval
const ARRIVED_DIST = 0.5; // metres threshold (unused for now — we rely on QR)

let graphData     = null;
let navPath       = [];   // ordered array of node IDs
let stepIdx       = 0;    // index of the node the user is CURRENTLY at
let destId        = null;
let isScanning    = false; // actively looking for QR
let isNavigating  = false;

// Compass
let compassHeading   = null; // degrees, 0=north, CW
let compassSupported = false;

// Calibration: bearing = buildingAngle + calOffset → magnetic bearing
// Arrow rotation on screen = bearing − compassHeading
let calOffset = null;
const CAL_KEY = 'vps_cal_offset_v2';
const saved = parseFloat(localStorage.getItem(CAL_KEY));
if (!isNaN(saved)) calOffset = saved;

// Canvas
let canvas, ctx;
let scanAnim = 0; // frame counter for scan animation

// QR scan throttle
let lastScanTime = 0;
let qrScanCanvas, qrScanCtx; // reusable off-screen canvas

// ══════════════════════════════════════════════════════════════
//  GRAPH HELPERS
// ══════════════════════════════════════════════════════════════
function getCoords(nd) {
  return Array.isArray(nd) ? nd : nd.coords;
}

function nodeLabel(id) {
  const nd = graphData?.nodes?.[id];
  return nd?.label ?? id;
}

function edgeWeight(aId, bId) {
  const e = graphData.edges.find(([a, b]) =>
    (a === aId && b === bId) || (a === bId && b === aId)
  );
  return e?.[2] ?? null;
}

// Direction instruction between two nodes
function stepInstruction(fromId, toId) {
  const fC = getCoords(graphData.nodes[fromId]);
  const tC = getCoords(graphData.nodes[toId]);
  const dz = tC[2] - fC[2]; // vertical
  if (dz < -0.4) return '🔼 Đi lên cầu thang';
  if (dz >  0.4) return '🔽 Đi xuống cầu thang';
  const d2 = Math.hypot(tC[0] - fC[0], tC[1] - fC[1]);
  if (d2 < 3)    return '↕️ Di chuyển một đoạn ngắn';
  return '➡️ Đi theo hành lang';
}

// Building-coordinate angle (degrees) from node A toward node B
function buildingAngle(fromId, toId) {
  const fC = getCoords(graphData.nodes[fromId]);
  const tC = getCoords(graphData.nodes[toId]);
  const dx = tC[0] - fC[0];
  const dy = tC[1] - fC[1];
  // atan2 → degrees, where 0° = +X axis
  return Math.atan2(dy, dx) * (180 / Math.PI);
}

// Magnetic bearing to next waypoint (null if calibration missing)
function targetBearing() {
  if (calOffset === null || stepIdx >= navPath.length - 1) return null;
  const bldAngle = buildingAngle(navPath[stepIdx], navPath[stepIdx + 1]);
  return ((bldAngle + calOffset) + 360) % 360;
}

// On-screen arrow rotation (0 = pointing UP = north)
function arrowRotation() {
  const bearing = targetBearing();
  if (bearing === null || compassHeading === null) return null;
  return ((bearing - compassHeading) + 360) % 360;
}

// ══════════════════════════════════════════════════════════════
//  DIJKSTRA
// ══════════════════════════════════════════════════════════════
function dijkstra(startId, endId) {
  const ids = Object.keys(graphData.nodes);
  const dist = {}, prev = {};
  for (const id of ids) { dist[id] = Infinity; prev[id] = null; }
  dist[startId] = 0;

  const adj = {};
  for (const id of ids) adj[id] = [];
  for (const [a, b, w] of graphData.edges) {
    adj[a]?.push({ to: b, w });
    adj[b]?.push({ to: a, w });
  }

  const unvisited = new Set(ids);
  while (unvisited.size > 0) {
    let u = null;
    for (const id of unvisited) if (u === null || dist[id] < dist[u]) u = id;
    if (!u || dist[u] === Infinity || u === endId) break;
    unvisited.delete(u);
    for (const { to: v, w } of adj[u] ?? []) {
      if (!unvisited.has(v)) continue;
      const alt = dist[u] + w;
      if (alt < dist[v]) { dist[v] = alt; prev[v] = u; }
    }
  }

  if (dist[endId] === Infinity) return null;
  const path = [];
  let cur = endId;
  while (cur) { path.unshift(cur); cur = prev[cur]; }
  return { path, distance: dist[endId] };
}

// ══════════════════════════════════════════════════════════════
//  CAMERA
// ══════════════════════════════════════════════════════════════
async function startCamera() {
  const video = document.getElementById('ar-video');
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: 'environment' },
        width:  { ideal: 1920 },
        height: { ideal: 1080 },
      },
      audio: false,
    });
    video.srcObject = stream;
    await new Promise(r => (video.onloadedmetadata = r));
    await video.play();
    return true;
  } catch (e) {
    console.error('Camera error:', e);
    return false;
  }
}

// ══════════════════════════════════════════════════════════════
//  COMPASS
// ══════════════════════════════════════════════════════════════
async function startCompass() {
  if (typeof DeviceOrientationEvent === 'undefined') return false;

  // iOS 13+ needs explicit permission
  if (typeof DeviceOrientationEvent.requestPermission === 'function') {
    try {
      const res = await DeviceOrientationEvent.requestPermission();
      if (res !== 'granted') return false;
    } catch { return false; }
  }

  window.addEventListener('deviceorientation', onOrientation, true);
  return true;
}

function onOrientation(e) {
  // iOS: webkitCompassHeading is already in [0,360], 0=north, CW
  if (e.webkitCompassHeading !== undefined && e.webkitCompassHeading !== null) {
    compassHeading = e.webkitCompassHeading;
    compassSupported = true;
  } else if (e.alpha !== null && e.alpha !== undefined) {
    // Android: alpha = rotation around Z-axis (CCW), 0=north
    // On many devices: heading = 360 - alpha
    compassHeading = (360 - e.alpha + 360) % 360;
    compassSupported = true;
  }

  // Update HUD
  if (compassHeading !== null) {
    document.getElementById('compass-val').textContent = Math.round(compassHeading);
    const cl = document.getElementById('cal-compass-live');
    if (cl) cl.textContent = Math.round(compassHeading) + '°';
    document.getElementById('no-compass-badge').style.display = 'none';
  }
}

// ══════════════════════════════════════════════════════════════
//  QR SCANNING
// ══════════════════════════════════════════════════════════════
function scanQR() {
  const now = Date.now();
  if (now - lastScanTime < QR_SCAN_MS) return null;
  lastScanTime = now;

  const video = document.getElementById('ar-video');
  if (!video || video.readyState < 2 || video.videoWidth === 0) return null;

  const vw = video.videoWidth;
  const vh = video.videoHeight;

  if (!qrScanCanvas) {
    qrScanCanvas = document.createElement('canvas');
    qrScanCtx    = qrScanCanvas.getContext('2d', { willReadFrequently: true });
  }

  // Scan a center crop for performance (QR tends to be centered)
  const scanSize = Math.min(vw, vh);
  const sx = (vw - scanSize) / 2;
  const sy = (vh - scanSize) / 2;

  qrScanCanvas.width  = scanSize;
  qrScanCanvas.height = scanSize;
  qrScanCtx.drawImage(video, sx, sy, scanSize, scanSize, 0, 0, scanSize, scanSize);

  const imgData = qrScanCtx.getImageData(0, 0, scanSize, scanSize);
  const code = jsQR(imgData.data, scanSize, scanSize, { inversionAttempts: 'dontInvert' });

  return code?.data ?? null;
}

// ══════════════════════════════════════════════════════════════
//  AR CANVAS RENDERING
// ══════════════════════════════════════════════════════════════
function setupCanvas() {
  canvas = document.getElementById('ar-canvas');
  ctx    = canvas.getContext('2d');
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);
}

function resizeCanvas() {
  canvas.width  = window.innerWidth;
  canvas.height = window.innerHeight;
}

function renderFrame() {
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  scanAnim = (scanAnim + 1) % 120;

  // ── Vignette ──────────────────────────────────────────
  const vig = ctx.createRadialGradient(W / 2, H / 2, H * 0.25, W / 2, H / 2, H * 0.75);
  vig.addColorStop(0, 'rgba(0,0,0,0)');
  vig.addColorStop(1, 'rgba(0,0,0,0.5)');
  ctx.fillStyle = vig;
  ctx.fillRect(0, 0, W, H);

  const cx = W / 2;

  if (isNavigating && navPath.length > 0 && stepIdx < navPath.length - 1) {
    const fromId = navPath[stepIdx];
    const toId   = navPath[stepIdx + 1];
    const fC = getCoords(graphData.nodes[fromId]);
    const tC = getCoords(graphData.nodes[toId]);
    const dz = tC[2] - fC[2];

    // Arrow center: upper-center of screen
    const cy = H * 0.35;
    const rot = arrowRotation();

    if (Math.abs(dz) > 0.4) {
      // Vertical movement
      drawVertical(cx, cy, dz < 0);
    } else if (rot !== null) {
      // Compass-guided arrow
      drawCompassArrow(cx, cy, rot * Math.PI / 180);
    } else {
      // No compass
      drawNoCompass(cx, cy);
    }

    // QR scan corners (lower portion — user aims camera downward at QR)
    drawScanCorners(W, H);

  } else if (isScanning && !isNavigating) {
    // Setup scan mode — just show QR corners centered
    drawScanCornersCentered(W, H);
  }
}

// ── Vertical staircase indicator ────────────────────────────
function drawVertical(cx, cy, goingUp) {
  const color = goingUp ? '#00d4ff' : '#a855f7';
  const label = goingUp ? 'Lên cầu thang ↑' : 'Xuống cầu thang ↓';
  const emoji = goingUp ? '⬆' : '⬇';

  // Pulsing ring
  const pulse = 0.6 + 0.4 * Math.sin(scanAnim / 120 * Math.PI * 4);
  ctx.beginPath();
  ctx.arc(cx, cy, 72 + 8 * pulse, 0, Math.PI * 2);
  ctx.strokeStyle = color + Math.round(pulse * 0xcc).toString(16).padStart(2, '0');
  ctx.lineWidth = 2.5;
  ctx.shadowColor = color;
  ctx.shadowBlur = 20;
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Second ring (static)
  ctx.beginPath();
  ctx.arc(cx, cy, 68, 0, Math.PI * 2);
  ctx.strokeStyle = color + '33';
  ctx.lineWidth = 1;
  ctx.stroke();

  // Emoji
  ctx.font = '52px Arial';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = color;
  ctx.shadowColor = color;
  ctx.shadowBlur = 15;
  ctx.fillText(emoji, cx, cy);
  ctx.shadowBlur = 0;

  // Label below circle
  ctx.font = 'bold 14px Inter, -apple-system, sans-serif';
  ctx.fillStyle = '#ffffff';
  ctx.shadowColor = 'rgba(0,0,0,0.8)';
  ctx.shadowBlur = 8;
  ctx.fillText(label, cx, cy + 95);
  ctx.shadowBlur = 0;
}

// ── Directional arrow with compass ring ─────────────────────
function drawCompassArrow(cx, cy, rotRad) {
  // ── Outer ring ────────────────────────────────────────────
  const r = 100;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(255,255,255,0.1)';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // N / S / E / W markers (rotate with compass so they point correctly)
  const cardinals = [
    { label: 'N', angle: 0,           color: '#ff4757' },
    { label: 'S', angle: Math.PI,     color: 'rgba(255,255,255,0.5)' },
    { label: 'E', angle: Math.PI/2,   color: 'rgba(255,255,255,0.5)' },
    { label: 'W', angle: -Math.PI/2,  color: 'rgba(255,255,255,0.5)' },
  ];

  // compassHeading tells us where magnetic north is relative to device
  const northOffset = compassHeading !== null ? -compassHeading * Math.PI / 180 : 0;

  ctx.font = 'bold 11px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (const { label, angle, color } of cardinals) {
    const a = angle + northOffset - Math.PI / 2; // −π/2 because 0 is up
    const lx = cx + Math.cos(a) * (r + 14);
    const ly = cy + Math.sin(a) * (r + 14);
    ctx.fillStyle = color;
    ctx.fillText(label, lx, ly);
  }

  // ── Arrow ─────────────────────────────────────────────────
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(rotRad);

  const H = 78, W2 = 26, TAIL = 32;

  // Glow shadow
  ctx.shadowColor = '#00ff88';
  ctx.shadowBlur  = 28;

  // Gradient fill (tip = bright, tail = transparent)
  const grad = ctx.createLinearGradient(0, -H, 0, TAIL);
  grad.addColorStop(0,   '#00ff88');
  grad.addColorStop(0.55, '#00d4ff');
  grad.addColorStop(1,   'rgba(0,212,255,0)');

  ctx.beginPath();
  ctx.moveTo(0, -H);           // tip
  ctx.lineTo(-W2, -H * 0.08); // left wing
  ctx.lineTo(-W2 * 0.45, -H * 0.08);
  ctx.lineTo(-W2 * 0.45, TAIL); // left tail
  ctx.lineTo(W2 * 0.45, TAIL);  // right tail
  ctx.lineTo(W2 * 0.45, -H * 0.08);
  ctx.lineTo(W2, -H * 0.08);  // right wing
  ctx.closePath();

  ctx.fillStyle = grad;
  ctx.fill();
  ctx.shadowBlur = 0;

  // Thin outline
  ctx.strokeStyle = 'rgba(255,255,255,0.2)';
  ctx.lineWidth = 1;
  ctx.stroke();

  ctx.restore();

  // Center dot
  ctx.beginPath();
  ctx.arc(cx, cy, 5, 0, Math.PI * 2);
  ctx.fillStyle = '#ffffff';
  ctx.fill();
}

// ── No-compass fallback ──────────────────────────────────────
function drawNoCompass(cx, cy) {
  ctx.beginPath();
  ctx.arc(cx, cy, 68, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(0,212,255,0.25)';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  ctx.font = '44px Arial';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('🧭', cx, cy);

  ctx.font = 'bold 13px Inter, sans-serif';
  ctx.fillStyle = 'rgba(255,255,255,0.65)';
  ctx.shadowColor = 'rgba(0,0,0,0.8)';
  ctx.shadowBlur = 8;
  ctx.fillText('Không có la bàn', cx, cy + 88);
  ctx.font = '11px Inter, sans-serif';
  ctx.fillStyle = 'rgba(255,255,255,0.35)';
  ctx.fillText('Theo chỉ dẫn văn bản bên dưới', cx, cy + 106);
  ctx.shadowBlur = 0;
}

// ── QR scan corners (lower center) ──────────────────────────
function drawScanCorners(W, H) {
  const size   = Math.min(W, H) * 0.48;
  const scanCx = W / 2;
  const scanCy = H * 0.72;
  drawCornerBox(scanCx - size / 2, scanCy - size / 2, size);

  // Scan line
  const progress = (scanAnim % 60) / 60;
  const lineY = scanCy - size / 2 + size * progress;
  const lineGrad = ctx.createLinearGradient(scanCx - size / 2, 0, scanCx + size / 2, 0);
  lineGrad.addColorStop(0,   'rgba(0,212,255,0)');
  lineGrad.addColorStop(0.5, `rgba(0,212,255,${0.5 + 0.3 * Math.sin(scanAnim / 20)})`);
  lineGrad.addColorStop(1,   'rgba(0,212,255,0)');
  ctx.beginPath();
  ctx.moveTo(scanCx - size / 2 + 10, lineY);
  ctx.lineTo(scanCx + size / 2 - 10, lineY);
  ctx.strokeStyle = lineGrad;
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

function drawScanCornersCentered(W, H) {
  const size = Math.min(W, H) * 0.56;
  const x0   = (W - size) / 2;
  const y0   = (H - size) / 2;
  drawCornerBox(x0, y0, size);

  // Scan line
  const progress = (scanAnim % 60) / 60;
  const lineY = y0 + size * progress;
  const grad = ctx.createLinearGradient(x0, 0, x0 + size, 0);
  grad.addColorStop(0,   'rgba(0,212,255,0)');
  grad.addColorStop(0.5, `rgba(0,212,255,${0.5 + 0.3 * Math.sin(scanAnim / 20)})`);
  grad.addColorStop(1,   'rgba(0,212,255,0)');
  ctx.beginPath();
  ctx.moveTo(x0 + 10, lineY);
  ctx.lineTo(x0 + size - 10, lineY);
  ctx.strokeStyle = grad;
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

function drawCornerBox(x0, y0, size) {
  const pulse    = 0.7 + 0.3 * Math.sin(scanAnim / 120 * Math.PI * 4);
  const cornerL  = 26;
  const cr       = 7;

  ctx.strokeStyle = `rgba(0,212,255,${pulse})`;
  ctx.lineWidth   = 2.5;
  ctx.shadowColor = '#00d4ff';
  ctx.shadowBlur  = 10;
  ctx.lineCap     = 'round';

  const corners = [
    [x0,        y0,        1,  1],
    [x0 + size, y0,       -1,  1],
    [x0,        y0 + size, 1, -1],
    [x0 + size, y0 + size,-1, -1],
  ];

  for (const [px, py, dx, dy] of corners) {
    ctx.beginPath();
    ctx.moveTo(px + dx * cornerL, py);
    ctx.lineTo(px + dx * cr, py);
    ctx.arcTo(px, py, px, py + dy * cr, cr);
    ctx.lineTo(px, py + dy * cornerL);
    ctx.stroke();
  }
  ctx.shadowBlur = 0;
}

// ══════════════════════════════════════════════════════════════
//  NAVIGATION STATE
// ══════════════════════════════════════════════════════════════
function startNav(fromNodeId, toNodeId) {
  const result = dijkstra(fromNodeId, toNodeId);
  if (!result) { alert('Không tìm thấy đường đi!'); return false; }

  navPath     = result.path;
  stepIdx     = 0;
  destId      = toNodeId;
  isNavigating = true;
  isScanning   = true;

  showNavPanel();
  updateNavUI();
  return true;
}

function advanceStep() {
  if (stepIdx >= navPath.length - 1) {
    showArrived();
    return;
  }
  stepIdx++;
  if (stepIdx >= navPath.length - 1) {
    showArrived();
  } else {
    updateNavUI();
    flashQRSuccess();
  }
}

function showArrived() {
  isNavigating = false;
  isScanning   = false;
  document.getElementById('nav-panel').style.display = 'none';
  document.getElementById('step-pill').style.display = 'none';
  document.getElementById('arrived-desc').textContent =
    `Bạn đã đến "${nodeLabel(destId)}" thành công! 🎉`;
  setScreen('arrived');
}

function resetNav() {
  navPath      = [];
  stepIdx      = 0;
  destId       = null;
  isNavigating = false;
  isScanning   = false;
  document.getElementById('dest-select').value = '';
  document.getElementById('btn-confirm-dest').disabled = true;
  document.getElementById('btn-confirm-dest').textContent = '📷 Bắt đầu — Quét QR xác định vị trí';
  document.getElementById('scan-waiting').style.display = 'none';
  document.getElementById('step-pill').style.display    = 'none';
  document.getElementById('nav-panel').style.display    = 'none';
  showSetup();
}

// ══════════════════════════════════════════════════════════════
//  UI HELPERS
// ══════════════════════════════════════════════════════════════
function setScreen(name) {
  for (const s of ['perm', 'arrived']) {
    const el = document.getElementById('screen-' + s);
    if (el) el.style.display = s === name ? 'flex' : 'none';
  }
}

function showSetup() {
  document.getElementById('setup-panel').style.display = 'block';
  document.getElementById('top-hud').style.display      = 'flex';
}

function showNavPanel() {
  document.getElementById('setup-panel').style.display = 'none';
  document.getElementById('nav-panel').style.display   = 'block';
  document.getElementById('step-pill').style.display   = 'flex';
}

function updateNavUI() {
  const total   = navPath.length;
  const current = stepIdx;
  const nextIdx = current + 1;

  // Step pill
  document.getElementById('step-pill-text').textContent =
    `Bước ${current + 1} / ${total - 1}`;

  // Dots
  const dotsEl = document.getElementById('step-dots');
  dotsEl.innerHTML = '';
  for (let i = 0; i < total - 1; i++) {
    const d = document.createElement('div');
    d.className = 'step-dot-item ' +
      (i < current ? 'done' : i === current ? 'current' : 'ahead');
    dotsEl.appendChild(d);
  }

  // Next node info
  if (nextIdx < total) {
    const nextId   = navPath[nextIdx];
    const label    = nodeLabel(nextId);
    const instr    = stepInstruction(navPath[current], nextId);
    const dist     = edgeWeight(navPath[current], nextId);

    document.getElementById('nav-next-name').textContent =
      `📍 ${label}`;
    document.getElementById('nav-instruction').textContent = instr;
    document.getElementById('nav-distance').textContent =
      dist ? `Khoảng cách: ~${dist.toFixed(1)} m` : '';

    document.getElementById('nav-scan-text').textContent =
      `Đến ${label} — quét QR ở đó để tiếp tục`;
    document.getElementById('btn-manual-advance').textContent =
      `✅ Tôi đã đến "${label}"`;
  }
}

function flashQRSuccess() {
  const el = document.getElementById('qr-flash');
  el.style.opacity = '1';
  setTimeout(() => { el.style.opacity = '0'; }, 350);
}

// ══════════════════════════════════════════════════════════════
//  POPULATE DROPDOWNS
// ══════════════════════════════════════════════════════════════
function populateNodeDropdowns() {
  const targets = ['dest-select', 'manual-pos-select', 'cal-from', 'cal-to'];

  // Group by floor
  const floors = {};
  for (const [id, nd] of Object.entries(graphData.nodes)) {
    const fl = nd.floor ?? '?';
    if (!floors[fl]) floors[fl] = [];
    floors[fl].push({ id, label: nd.label || id });
  }
  const sortedFloors = Object.keys(floors).sort((a, b) => Number(b) - Number(a));

  for (const selId of targets) {
    const sel = document.getElementById(selId);
    if (!sel) continue;
    const placeholder = sel.options[0]?.value === '' ? sel.options[0].text : null;
    sel.innerHTML = placeholder ? `<option value="">${placeholder}</option>` : '';

    for (const fl of sortedFloors) {
      const name = fl == 1.5 ? 'Chiếu nghỉ cầu thang' : `Tầng ${fl}`;
      const og   = document.createElement('optgroup');
      og.label   = name;
      for (const { id, label } of floors[fl]) {
        og.appendChild(new Option(label, id));
      }
      sel.appendChild(og);
    }
  }
}

// ══════════════════════════════════════════════════════════════
//  CALIBRATION
// ══════════════════════════════════════════════════════════════
function updateCalUI() {
  const saved = document.getElementById('cal-saved');
  if (saved) {
    saved.textContent = calOffset !== null ? `${calOffset.toFixed(1)}°` : 'Chưa có';
  }
}

function doCalibrate() {
  const fromId = document.getElementById('cal-from').value;
  const toId   = document.getElementById('cal-to').value;
  if (!fromId || !toId || fromId === toId) {
    alert('Chọn hai node khác nhau!');
    return;
  }
  if (compassHeading === null) {
    alert('Chưa nhận được dữ liệu la bàn từ thiết bị. Hãy thử xoay điện thoại vài vòng.');
    return;
  }

  const bldAngle = buildingAngle(fromId, toId);
  // calOffset = compassHeading - bldAngle
  // So: bearing (magnetic) = bldAngle + calOffset = compassHeading ✓
  calOffset = ((compassHeading - bldAngle) + 360) % 360;
  localStorage.setItem(CAL_KEY, calOffset);
  updateCalUI();

  document.getElementById('calibration-modal').style.display = 'none';
  const toLabel = nodeLabel(toId);
  alert(`✅ Căn chỉnh thành công!\nOffset: ${calOffset.toFixed(1)}°\n\nMũi tên AR sẽ hiển thị chính xác theo hướng "${toLabel}".`);
}

// ══════════════════════════════════════════════════════════════
//  MAIN ANIMATION LOOP
// ══════════════════════════════════════════════════════════════
function mainLoop() {
  requestAnimationFrame(mainLoop);

  // Render AR frame
  renderFrame();

  // QR scan (throttled)
  if (isScanning) {
    const data = scanQR();
    if (data) onQRDetected(data);
  }
}

function onQRDetected(nodeId) {
  // Validate: is this a known node?
  if (!graphData?.nodes?.[nodeId]) return;

  if (!isNavigating) {
    // SETUP mode: start navigation from this node
    const dest = document.getElementById('dest-select').value;
    if (!dest) {
      // Highlight the select
      const sel = document.getElementById('dest-select');
      sel.style.outline = '2px solid #ff4757';
      setTimeout(() => sel.style.outline = '', 1500);
      return;
    }
    if (nodeId === dest) { alert('Bạn đang ở điểm đến rồi!'); return; }
    flashQRSuccess();
    startNav(nodeId, dest);
    return;
  }

  // NAVIGATING mode: check if this QR matches expected next node
  const expected = navPath[stepIdx + 1];
  if (nodeId === expected) {
    flashQRSuccess();
    advanceStep();
  } else if (nodeId === navPath[stepIdx]) {
    // Scanned current node again — ignore
  } else if (graphData.nodes[nodeId]) {
    // Off-path node → re-route
    const dest = navPath[navPath.length - 1];
    if (nodeId === dest) { showArrived(); return; }
    flashQRSuccess();
    startNav(nodeId, dest);
  }
}

// ══════════════════════════════════════════════════════════════
//  INIT
// ══════════════════════════════════════════════════════════════
async function main() {
  // Load graph
  try {
    const res = await fetch('/api/graph');
    graphData  = await res.json();
  } catch (e) {
    graphData = { nodes: {}, edges: [] };
  }

  setupCanvas();

  // ── Btn: Start AR ─────────────────────────────────────
  document.getElementById('btn-start-ar').addEventListener('click', async () => {
    const btn = document.getElementById('btn-start-ar');
    btn.textContent = '⏳ Đang bật camera...';
    btn.disabled    = true;

    const camOk = await startCamera();
    if (!camOk) {
      btn.textContent = '❌ Không thể bật camera';
      btn.disabled    = false;
      return;
    }

    // Try compass (optional, ignore failure)
    const compassOk = await startCompass();
    if (!compassOk) {
      compassSupported = false;
      document.getElementById('no-compass-badge').style.display = 'block';
      document.getElementById('compass-pill').style.display     = 'none';
    }

    populateNodeDropdowns();
    updateCalUI();

    document.getElementById('screen-perm').style.display = 'none';
    document.getElementById('top-hud').style.display     = 'flex';
    document.getElementById('setup-panel').style.display = 'block';

    mainLoop();
  });

  // ── Btn: Confirm destination ───────────────────────────
  document.getElementById('dest-select').addEventListener('change', (e) => {
    const ok = !!e.target.value;
    document.getElementById('btn-confirm-dest').disabled = !ok;
  });

  document.getElementById('btn-confirm-dest').addEventListener('click', () => {
    const dest = document.getElementById('dest-select').value;
    if (!dest) return;
    isScanning = true;
    document.getElementById('scan-waiting').style.display = 'block';
    document.getElementById('btn-confirm-dest').textContent = '📷 Đang quét QR...';
    document.getElementById('btn-confirm-dest').disabled    = true;
  });

  // ── Btn: Back/Reset ────────────────────────────────────
  document.getElementById('btn-back-hud').addEventListener('click', (e) => {
    e.preventDefault();
    if (isNavigating) {
      if (!confirm('Dừng điều hướng và quay lại?')) return;
    }
    resetNav();
  });

  // ── Btn: Reset from nav panel ──────────────────────────
  // (done via back-hud above)

  // ── Btn: Manual position ───────────────────────────────
  document.getElementById('btn-manual-pos').addEventListener('click', () => {
    document.getElementById('manual-modal').style.display = 'flex';
  });

  document.getElementById('btn-close-manual').addEventListener('click', () => {
    document.getElementById('manual-modal').style.display = 'none';
  });

  document.getElementById('btn-confirm-manual').addEventListener('click', () => {
    const nodeId = document.getElementById('manual-pos-select').value;
    const dest   = document.getElementById('dest-select').value;
    if (!nodeId) { alert('Vui lòng chọn vị trí của bạn!'); return; }
    if (!dest)   { alert('Vui lòng chọn điểm đến trước!'); return; }
    if (nodeId === dest) { alert('Bạn đang ở điểm đến rồi!'); return; }
    document.getElementById('manual-modal').style.display = 'none';
    startNav(nodeId, dest);
  });

  // ── Btn: Manual advance ────────────────────────────────
  document.getElementById('btn-manual-advance').addEventListener('click', () => {
    if (confirm('Xác nhận bạn đã đến điểm tiếp theo?')) {
      advanceStep();
    }
  });

  // ── Btn: Nav again (arrived screen) ────────────────────
  document.getElementById('btn-nav-again').addEventListener('click', () => {
    setScreen(null);
    resetNav();
  });

  // ── Calibration modal ──────────────────────────────────
  document.getElementById('btn-open-calibration').addEventListener('click', () => {
    document.getElementById('calibration-modal').style.display = 'flex';
  });
  document.getElementById('btn-close-calibration').addEventListener('click', () => {
    document.getElementById('calibration-modal').style.display = 'none';
  });
  document.getElementById('btn-do-calibrate').addEventListener('click', doCalibrate);
}

main();
