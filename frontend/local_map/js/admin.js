/**
 * VPS Admin Panel — CRUD nodes & edges + 3D preview
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ══════════════════════════════════════════════════════════════
//  STATE
// ══════════════════════════════════════════════════════════════
let graphData  = null;  // live copy being edited
let hasUnsaved = false;
let editingNodeId = null;  // null = add mode, string = edit mode
let editingEdgeIdx = null; // null = add mode, number = edit mode

// 3D preview
let scene3, camera3, renderer3, controls3;
const previewMeshes = {};
const previewLines  = [];

// ══════════════════════════════════════════════════════════════
//  UTILS
// ══════════════════════════════════════════════════════════════
function getCoords(nd) {
  return Array.isArray(nd) ? nd : nd.coords;
}

function euclidean(aCoords, bCoords) {
  const dx = aCoords[0] - bCoords[0];
  const dy = aCoords[1] - bCoords[1];
  const dz = aCoords[2] - bCoords[2];
  return Math.sqrt(dx*dx + dy*dy + dz*dz);
}

function markUnsaved() {
  hasUnsaved = true;
  document.getElementById('save-status').innerHTML =
    '<span class="unsaved-dot"></span>Có thay đổi chưa lưu';
}

function toast(msg, type = 'success') {
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 2900);
}

// ══════════════════════════════════════════════════════════════
//  3D PREVIEW (lightweight — no PLY)
// ══════════════════════════════════════════════════════════════
function setup3DPreview() {
  const canvas    = document.getElementById('canvas-admin');
  const container = canvas.parentElement;

  renderer3 = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer3.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer3.setSize(container.clientWidth, container.clientHeight);

  scene3 = new THREE.Scene();
  scene3.background = new THREE.Color(0x070b17);

  const aspect = container.clientWidth / container.clientHeight;
  camera3 = new THREE.PerspectiveCamera(55, aspect, 0.01, 1000);
  camera3.position.set(0, 15, 30);

  scene3.add(new THREE.AmbientLight(0x334466, 4));
  const dl = new THREE.DirectionalLight(0xffffff, 3);
  dl.position.set(10, 20, 10);
  scene3.add(dl);

  const grid = new THREE.GridHelper(80, 40, 0x112233, 0x0d1a2a);
  grid.position.y = -5.5;
  scene3.add(grid);

  controls3 = new OrbitControls(camera3, renderer3.domElement);
  controls3.enableDamping = true;
  controls3.dampingFactor = 0.07;

  window.addEventListener('resize', () => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera3.aspect = w / h;
    camera3.updateProjectionMatrix();
    renderer3.setSize(w, h);
  });

  (function loop() {
    requestAnimationFrame(loop);
    controls3.update();
    renderer3.render(scene3, camera3);
  })();
}

function toThree3(coords) {
  // same mapping as main app: nav(x,y,z) → Three(x, z, y)
  return new THREE.Vector3(coords[0], coords[2], coords[1]);
}

function rebuildPreview() {
  // Remove old meshes
  for (const m of Object.values(previewMeshes)) scene3.remove(m);
  for (const l of previewLines) scene3.remove(l);
  Object.keys(previewMeshes).forEach(k => delete previewMeshes[k]);
  previewLines.length = 0;

  if (!graphData) return;

  // Edges
  for (const [aId, bId] of graphData.edges.map(e => [e[0], e[1]])) {
    const aN = graphData.nodes[aId];
    const bN = graphData.nodes[bId];
    if (!aN || !bN) continue;
    const pts = [toThree3(getCoords(aN)), toThree3(getCoords(bN))];
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    const mat = new THREE.LineBasicMaterial({ color: 0x334466, transparent: true, opacity: 0.7 });
    const line = new THREE.Line(geo, mat);
    scene3.add(line);
    previewLines.push(line);
  }

  // Nodes
  const sGeo = new THREE.SphereGeometry(0.3, 14, 14);
  for (const [id, nd] of Object.entries(graphData.nodes)) {
    const mat  = new THREE.MeshPhongMaterial({ color: 0x00d4ff, emissive: 0x002233 });
    const mesh = new THREE.Mesh(sGeo, mat.clone());
    mesh.position.copy(toThree3(getCoords(nd)));
    scene3.add(mesh);
    previewMeshes[id] = mesh;
  }
}

// ══════════════════════════════════════════════════════════════
//  NODE TABLE
// ══════════════════════════════════════════════════════════════
function renderNodeTable() {
  const tbody = document.getElementById('node-tbody');
  tbody.innerHTML = '';

  const entries = Object.entries(graphData.nodes);
  document.getElementById('node-count').textContent  = entries.length;
  document.getElementById('node-count2').textContent = entries.length;

  for (const [id, nd] of entries) {
    const coords = getCoords(nd);
    const tr = document.createElement('tr');
    if (editingNodeId === id) tr.classList.add('editing-row');

    tr.innerHTML = `
      <td class="table-id">${id}</td>
      <td>${nd.label || '—'}</td>
      <td>${nd.floor ?? '?'}</td>
      <td style="font-family:monospace;font-size:11px;color:var(--text-muted)">
        ${coords[0].toFixed(2)}, ${coords[1].toFixed(2)}, ${coords[2].toFixed(2)}
      </td>
      <td style="white-space:nowrap;">
        <button class="btn-edit"   data-action="edit-node"   data-id="${id}">✏️ Sửa</button>
        <button class="btn-delete" data-action="delete-node" data-id="${id}">🗑️</button>
      </td>`;
    tbody.appendChild(tr);
  }
}

function fillNodeForm(id) {
  const nd = graphData.nodes[id];
  if (!nd) return;
  const coords = getCoords(nd);
  document.getElementById('node-id').value    = id;
  document.getElementById('node-id').disabled = true; // can't change key while editing
  document.getElementById('node-label').value = nd.label || '';
  document.getElementById('node-floor').value = nd.floor ?? '';
  document.getElementById('node-x').value     = coords[0];
  document.getElementById('node-y').value     = coords[1];
  document.getElementById('node-z').value     = coords[2];
  document.getElementById('btn-add-node').textContent = '💾 Cập nhật node';
  document.getElementById('btn-cancel-node').style.display = 'block';
  document.getElementById('node-form-title').textContent   = '✏️  Chỉnh Sửa Node';
}

function clearNodeForm() {
  ['node-id','node-label','node-floor','node-x','node-y','node-z']
    .forEach(id => { document.getElementById(id).value = ''; });
  document.getElementById('node-id').disabled = false;
  document.getElementById('btn-add-node').textContent = '➕ Thêm node';
  document.getElementById('btn-cancel-node').style.display = 'none';
  document.getElementById('node-form-title').textContent   = '➕  Thêm Node Mới';
  editingNodeId = null;
}

function saveNode() {
  const id    = document.getElementById('node-id').value.trim().replace(/\s+/g,'_');
  const label = document.getElementById('node-label').value.trim();
  const floor = parseFloat(document.getElementById('node-floor').value) || 1;
  const x     = parseFloat(document.getElementById('node-x').value);
  const y     = parseFloat(document.getElementById('node-y').value);
  const z     = parseFloat(document.getElementById('node-z').value);

  if (!id)    { toast('Vui lòng nhập ID!',    'error'); return; }
  if (!label) { toast('Vui lòng nhập tên!',   'error'); return; }
  if (isNaN(x) || isNaN(y) || isNaN(z)) { toast('Toạ độ không hợp lệ!', 'error'); return; }
  if (!editingNodeId && graphData.nodes[id]) {
    toast(`ID "${id}" đã tồn tại!`, 'error'); return;
  }

  graphData.nodes[id] = { coords: [x, y, z], label, floor };
  clearNodeForm();
  markUnsaved();
  renderNodeTable();
  populateEdgeDropdowns();
  rebuildPreview();
  toast(editingNodeId ? 'Đã cập nhật node!' : 'Đã thêm node mới!');
}

// ══════════════════════════════════════════════════════════════
//  EDGE TABLE
// ══════════════════════════════════════════════════════════════
function renderEdgeTable() {
  const tbody = document.getElementById('edge-tbody');
  tbody.innerHTML = '';

  document.getElementById('edge-count').textContent  = graphData.edges.length;
  document.getElementById('edge-count2').textContent = graphData.edges.length;

  graphData.edges.forEach(([aId, bId, w], idx) => {
    const aLabel = graphData.nodes[aId]?.label || aId;
    const bLabel = graphData.nodes[bId]?.label || bId;
    const tr = document.createElement('tr');
    if (editingEdgeIdx === idx) tr.classList.add('editing-row');

    tr.innerHTML = `
      <td>${aLabel}</td>
      <td>${bLabel}</td>
      <td style="font-family:monospace;font-size:11px;">${w.toFixed(2)} m</td>
      <td style="white-space:nowrap;">
        <button class="btn-edit"   data-action="edit-edge"   data-idx="${idx}">✏️ Sửa</button>
        <button class="btn-delete" data-action="delete-edge" data-idx="${idx}">🗑️</button>
      </td>`;
    tbody.appendChild(tr);
  });
}

function fillEdgeForm(idx) {
  const [aId, bId, w] = graphData.edges[idx];
  document.getElementById('edge-from').value   = aId;
  document.getElementById('edge-to').value     = bId;
  document.getElementById('edge-weight').value = w;
  document.getElementById('btn-add-edge').textContent = '💾 Cập nhật edge';
  document.getElementById('btn-cancel-edge').style.display = 'block';
  document.getElementById('edge-form-title').textContent   = '✏️  Chỉnh Sửa Edge';
}

function clearEdgeForm() {
  document.getElementById('edge-from').value   = '';
  document.getElementById('edge-to').value     = '';
  document.getElementById('edge-weight').value = '';
  document.getElementById('btn-add-edge').textContent = '➕ Thêm edge';
  document.getElementById('btn-cancel-edge').style.display = 'none';
  document.getElementById('edge-form-title').textContent   = '➕  Thêm Edge Mới';
  editingEdgeIdx = null;
}

function saveEdge() {
  const fromId = document.getElementById('edge-from').value;
  const toId   = document.getElementById('edge-to').value;
  let   wRaw   = document.getElementById('edge-weight').value;

  if (!fromId || !toId)  { toast('Vui lòng chọn hai node!', 'error'); return; }
  if (fromId === toId)   { toast('Hai node không được trùng nhau!', 'error'); return; }

  let w;
  if (wRaw === '' || wRaw == null) {
    // Auto-calculate
    const aC = getCoords(graphData.nodes[fromId]);
    const bC = getCoords(graphData.nodes[toId]);
    w = parseFloat(euclidean(aC, bC).toFixed(3));
  } else {
    w = parseFloat(wRaw);
    if (isNaN(w) || w <= 0) { toast('Khoảng cách không hợp lệ!', 'error'); return; }
  }

  const newEdge = [fromId, toId, w];

  if (editingEdgeIdx !== null) {
    graphData.edges[editingEdgeIdx] = newEdge;
  } else {
    // Check duplicate
    const dup = graphData.edges.some(([a, b]) =>
      (a === fromId && b === toId) || (a === toId && b === fromId)
    );
    if (dup) { toast('Edge này đã tồn tại!', 'error'); return; }
    graphData.edges.push(newEdge);
  }

  clearEdgeForm();
  markUnsaved();
  renderEdgeTable();
  rebuildPreview();
  toast(editingEdgeIdx !== null ? 'Đã cập nhật edge!' : `Đã thêm edge (${w.toFixed(2)} m)!`);
}

// ══════════════════════════════════════════════════════════════
//  POPULATE EDGE DROPDOWNS
// ══════════════════════════════════════════════════════════════
function populateEdgeDropdowns() {
  const fromSel = document.getElementById('edge-from');
  const toSel   = document.getElementById('edge-to');
  const fVal = fromSel.value;
  const tVal = toSel.value;

  fromSel.innerHTML = '<option value="">— Chọn —</option>';
  toSel.innerHTML   = '<option value="">— Chọn —</option>';

  for (const [id, nd] of Object.entries(graphData.nodes)) {
    fromSel.appendChild(new Option(nd.label || id, id));
    toSel.appendChild(new Option(nd.label || id, id));
  }

  fromSel.value = fVal;
  toSel.value   = tVal;
}

// ══════════════════════════════════════════════════════════════
//  SAVE / LOAD
// ══════════════════════════════════════════════════════════════
async function loadGraph() {
  try {
    const res = await fetch('/api/graph');
    graphData = await res.json();
    hasUnsaved = false;
    document.getElementById('save-status').textContent = 'Dữ liệu đã đồng bộ';
    renderNodeTable();
    renderEdgeTable();
    populateEdgeDropdowns();
    rebuildPreview();
    fitPreviewCamera();
  } catch (e) {
    toast('Không thể tải dữ liệu từ server!', 'error');
  }
}

async function saveGraph() {
  if (!graphData) return;
  try {
    const res = await fetch('/api/graph', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(graphData),
    });
    const result = await res.json();
    if (result.success) {
      hasUnsaved = false;
      document.getElementById('save-status').textContent =
        `Đã lưu lúc ${new Date().toLocaleTimeString('vi-VN')}`;
      toast('Lưu thành công! ✅');
    } else {
      toast('Lỗi khi lưu: ' + result.error, 'error');
    }
  } catch (e) {
    toast('Lỗi kết nối server!', 'error');
  }
}

// ══════════════════════════════════════════════════════════════
//  FIT CAMERA TO GRAPH
// ══════════════════════════════════════════════════════════════
function fitPreviewCamera() {
  if (!graphData) return;
  const positions = Object.values(graphData.nodes).map(nd => toThree3(getCoords(nd)));
  if (positions.length === 0) return;

  const box = new THREE.Box3();
  positions.forEach(p => box.expandByPoint(p));
  const center = new THREE.Vector3(); box.getCenter(center);
  const size   = new THREE.Vector3(); box.getSize(size);
  const maxDim = Math.max(size.x, size.y, size.z);

  controls3.target.copy(center);
  camera3.position.set(
    center.x + maxDim,
    center.y + maxDim * 0.8,
    center.z + maxDim,
  );
  camera3.lookAt(center);
  controls3.update();
}

// ══════════════════════════════════════════════════════════════
//  TAB SWITCHER
// ══════════════════════════════════════════════════════════════
function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
    });
  });
}

// ══════════════════════════════════════════════════════════════
//  EVENT DELEGATION
// ══════════════════════════════════════════════════════════════
function initTableEvents() {
  // Node table actions
  document.getElementById('node-tbody').addEventListener('click', (e) => {
    const btn = e.target.closest('[data-action]');
    if (!btn) return;
    const { action, id } = btn.dataset;

    if (action === 'edit-node') {
      editingNodeId = id;
      fillNodeForm(id);
      renderNodeTable(); // highlight row
    }
    if (action === 'delete-node') {
      if (!confirm(`Xoá node "${graphData.nodes[id]?.label || id}"?\nCác edge liên quan cũng sẽ bị xoá.`)) return;
      delete graphData.nodes[id];
      // Remove related edges
      graphData.edges = graphData.edges.filter(([a, b]) => a !== id && b !== id);
      markUnsaved();
      renderNodeTable();
      renderEdgeTable();
      populateEdgeDropdowns();
      rebuildPreview();
      toast('Đã xoá node!', 'info');
    }
  });

  // Edge table actions
  document.getElementById('edge-tbody').addEventListener('click', (e) => {
    const btn = e.target.closest('[data-action]');
    if (!btn) return;
    const { action, idx } = btn.dataset;
    const i = parseInt(idx);

    if (action === 'edit-edge') {
      editingEdgeIdx = i;
      fillEdgeForm(i);
      renderEdgeTable(); // highlight row
    }
    if (action === 'delete-edge') {
      if (!confirm('Xoá edge này?')) return;
      graphData.edges.splice(i, 1);
      markUnsaved();
      renderEdgeTable();
      rebuildPreview();
      toast('Đã xoá edge!', 'info');
    }
  });
}

// ══════════════════════════════════════════════════════════════
//  MAIN
// ══════════════════════════════════════════════════════════════
async function main() {
  setup3DPreview();
  initTabs();
  initTableEvents();

  // Node form buttons
  document.getElementById('btn-add-node').addEventListener('click', saveNode);
  document.getElementById('btn-cancel-node').addEventListener('click', () => clearNodeForm());

  // Edge form buttons
  document.getElementById('btn-add-edge').addEventListener('click', saveEdge);
  document.getElementById('btn-cancel-edge').addEventListener('click', () => clearEdgeForm());

  // Save / reload
  document.getElementById('btn-save').addEventListener('click', saveGraph);
  document.getElementById('btn-reload').addEventListener('click', async () => {
    if (hasUnsaved && !confirm('Bạn có thay đổi chưa lưu. Tải lại sẽ mất tất cả. Tiếp tục?')) return;
    await loadGraph();
  });

  // Warn before leaving with unsaved changes
  window.addEventListener('beforeunload', (e) => {
    if (hasUnsaved) {
      e.preventDefault();
      e.returnValue = '';
    }
  });

  // Keyboard shortcut: Ctrl+S to save
  window.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 's') {
      e.preventDefault();
      saveGraph();
    }
  });

  await loadGraph();
}

main();
