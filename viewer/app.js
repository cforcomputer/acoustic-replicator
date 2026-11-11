// Replicator Mk2 simulation viewer.  Everything shown is read from the project's own output files:
// hardware/machine.json (+ meshes.bin) for the machine, hardware/pcb/TB-25.json for the boards,
// out/print/* for the simulated print, out/viewer/replay_*.json for the physics test flights.
import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const V3 = THREE.Vector3;
const MM = 1000;                       // metres -> scene units (mm)
const G = 9.81;

// ------------------------------------------------------------------ renderer
const canvas = $('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
const scene = new THREE.Scene();
const BG = new THREE.Color(0x0d1117);
scene.background = BG;
const camera = new THREE.PerspectiveCamera(36, 1, 0.5, 8000);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

// soft studio environment so metal and plastic read as what they are
function makeEnvironment() {
  const s = new THREE.Scene();
  s.background = new THREE.Color(0x30343a);
  const panel = (w, h, c, pos, look) => {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color: c, side: THREE.DoubleSide }));
    m.position.set(...pos); m.lookAt(...look); s.add(m);
  };
  panel(8, 8, 0xffffff, [0, 6, 0], [0, 0, 0]);
  panel(6, 4, 0xbfd4ff, [-7, 1, 3], [0, 0, 0]);
  panel(6, 4, 0xfff0d8, [7, 2, -3], [0, 0, 0]);
  panel(10, 10, 0x15171a, [0, -6, 0], [0, 0, 0]);
  const pm = new THREE.PMREMGenerator(renderer);
  const tex = pm.fromScene(s, 0.03).texture;
  pm.dispose();
  return tex;
}
const ENV = makeEnvironment();
scene.environment = ENV;
const hemi = new THREE.HemisphereLight(0xdfe9ff, 0x20242c, 0.75); scene.add(hemi);
const sun = new THREE.DirectionalLight(0xffffff, 1.5); sun.position.set(180, -260, 320); scene.add(sun);
const fill = new THREE.DirectionalLight(0x9db4ff, 0.45); fill.position.set(-200, 150, -80); scene.add(fill);

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.floor(w * renderer.getPixelRatio()) || canvas.height !== Math.floor(h * renderer.getPixelRatio())) {
    renderer.setSize(w, h, false);
    camera.aspect = w / Math.max(h, 1);
    camera.updateProjectionMatrix();
  }
}
function setCam(pos, target) { camera.position.set(...pos); controls.target.set(...target); controls.update(); }
const CAMS = {
  all: [[430, -690, 300], [0, 0, -45]],
  build: [[150, -285, 105], [0, 0, -8]],
  part: [[30, -46, 8], [2, 0, 1]],
  drawer: [[150, -600, 60], [0, 0, -190]],
};

// ------------------------------------------------------------------ materials
const matCache = new Map();
function material(mat, color) {
  const key = mat + color;
  if (matCache.has(key)) return matCache.get(key);
  const o = { color: new THREE.Color(color), roughness: 0.6, metalness: 0.0 };
  if (mat === 'metal') Object.assign(o, { metalness: 0.9, roughness: 0.38 });
  else if (mat === 'plastic') Object.assign(o, { roughness: 0.55 });
  else if (mat === 'pcb') Object.assign(o, { roughness: 0.45, metalness: 0.05 });
  else if (mat === 'foam') Object.assign(o, { roughness: 1.0 });
  else if (mat === 'wood') Object.assign(o, { roughness: 0.82 });
  else if (mat === 'acrylic') Object.assign(o, { roughness: 0.08, transparent: true, opacity: 0.38, depthWrite: false });
  else if (mat === 'glass') Object.assign(o, { roughness: 0.1, transparent: true, opacity: 0.72 });
  const m = new THREE.MeshStandardMaterial(o);
  matCache.set(key, m);
  return m;
}
function canTopTexture() {            // front of a 40 kHz transducer: rim, dark recess, fine mesh
  const c = document.createElement('canvas'); c.width = c.height = 128;
  const g = c.getContext('2d');
  g.fillStyle = '#c3c7cc'; g.fillRect(0, 0, 128, 128);
  g.fillStyle = '#30343a'; g.beginPath(); g.arc(64, 64, 50, 0, 7); g.fill();
  g.strokeStyle = '#8a9099'; g.lineWidth = 1.6;
  for (let i = -50; i <= 50; i += 7) { g.beginPath(); g.moveTo(64 + i, 14); g.lineTo(64 + i, 114); g.stroke(); g.beginPath(); g.moveTo(14, 64 + i); g.lineTo(114, 64 + i); g.stroke(); }
  g.globalCompositeOperation = 'destination-in'; g.beginPath(); g.arc(64, 64, 63, 0, 7); g.fill();
  g.globalCompositeOperation = 'destination-over'; g.fillStyle = '#c3c7cc'; g.fillRect(0, 0, 128, 128);
  g.globalCompositeOperation = 'source-over'; g.strokeStyle = '#d9dde3'; g.lineWidth = 6; g.beginPath(); g.arc(64, 64, 53, 0, 7); g.stroke();
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4;
  return t;
}

function stripeTexture(kind) {       // heat-sink fins / perforated supply cover, drawn rather than modelled
  const c = document.createElement('canvas'); c.width = c.height = 256;
  const g = c.getContext('2d');
  g.fillStyle = kind === 'fins' ? '#c8ccd2' : '#b9bdc3'; g.fillRect(0, 0, 256, 256);
  if (kind === 'fins') { g.fillStyle = '#5d636b'; for (let x = 6; x < 256; x += 16) g.fillRect(x, 0, 7, 256); }
  else { g.fillStyle = '#3d4249'; for (let y = 10; y < 256; y += 20) for (let x = 10 + (y / 20 % 2) * 10; x < 256; x += 20) { g.beginPath(); g.arc(x, y, 5, 0, 7); g.fill(); } }
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4;
  return t;
}
const finMat = new THREE.MeshStandardMaterial({ map: stripeTexture('fins'), metalness: 0.7, roughness: 0.45 });
const perfMat = new THREE.MeshStandardMaterial({ map: stripeTexture('perf'), metalness: 0.7, roughness: 0.5 });
function fanRotor(c) {                // hub and blades inside the fan frame (the frame itself comes from the CAD model)
  const g = new THREE.Group();
  const k = c.size.indexOf(Math.min(...c.size)), R = Math.max(...c.size) * MM / 2 - 5, th = c.size[k] * MM * 0.7;
  const dark = material('plastic', '#101114');
  const hub = new THREE.Mesh(new THREE.CylinderGeometry(R * 0.33, R * 0.33, th, 28), dark); g.add(hub);
  for (let i = 0; i < 7; i++) {
    const b = new THREE.Mesh(new THREE.BoxGeometry(R * 0.66, 1.4, R * 0.5), dark);
    const a = i * 2 * Math.PI / 7;
    b.position.set(Math.cos(a) * R * 0.63, 0, Math.sin(a) * R * 0.63); b.rotation.set(0.5, -a, 0); g.add(b);
  }
  const ax = new V3(0, 0, 0); ax.setComponent(k, 1);
  g.quaternion.setFromUnitVectors(YAX, ax);
  g.position.set(c.center[0] * MM, c.center[1] * MM, c.center[2] * MM);
  return g;
}

// ------------------------------------------------------------------ hardware
const hw = { groups: {}, front: [], pick: [], uvLens: [], diffusers: [], cams: [], lights: [] };
const YAX = new V3(0, 1, 0);
const CAGE = new Set(['frame', 'boards', 'transducers', 'fasteners', 'uv', 'camera', 'strobe', 'pods', 'scanner', 'boardparts', 'top', 'pole']);
const CAT = {                           // which checkbox controls which group
  enclosure_shell: 'enc', enclosure_foam: 'enc', thermal: 'enc', hardware: 'enc', door: 'door', lid: 'door',
  boards: 'boards', transducers: 'boards', boardparts: 'boards', wires: 'wires',
};
function basisQuat(x, z) {
  const X = new V3(...x).normalize(), Z = new V3(...z).normalize();
  const Y = new V3().crossVectors(Z, X).normalize();
  return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(X, Y, Z));
}
function group(name) {
  if (!hw.groups[name]) { hw.groups[name] = new THREE.Group(); hw.groups[name].name = name; scene.add(hw.groups[name]); }
  return hw.groups[name];
}
const info = (c) => ({ label: c.label, mount: c.mount ? `fastened to ${c.mount.to.replace(/_/g, ' ')}: ${c.mount.by}` : (c.body ? `part of the printed ${c.body.replace(/_/g, ' ')}` : '') });

async function loadHardware() {
  const t = '?' + Date.now();
  const [m, buf, pcb] = await Promise.all([
    fetch('/hardware/machine.json' + t).then((r) => r.json()),
    fetch('/hardware/meshes.bin' + t).then((r) => r.arrayBuffer()),
    fetch('/hardware/pcb/TB-25.json' + t).then((r) => r.json()),
  ]);
  hw.machine = m;
  const tr = { back: [], front: [] };
  const isFrontPt = (y) => y * MM < -35;
  for (const c of m.components) {
    if (c.hidden) continue;
    let g = c.group;
    if (g === 'transducers') { (isFrontPt(c.top[1]) ? tr.front : tr.back).push(c); continue; }
    if (g === 'fasteners' && c.id.startsWith('sq')) g = 'boards';
    let mesh, cy;
    const mat = material(c.mat || 'plastic', c.color);
    if (c.mesh) {
      const pos = new Float32Array(buf.slice(c.mesh.voff, c.mesh.voff + c.mesh.nv * 12));
      for (let i = 0; i < pos.length; i++) pos[i] *= MM;
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      geo.setIndex(new THREE.BufferAttribute(new Uint32Array(buf.slice(c.mesh.ioff, c.mesh.ioff + c.mesh.ni * 4)), 1));
      geo.computeVertexNormals(); geo.computeBoundingBox();
      mesh = new THREE.Mesh(geo, mat);
      cy = (geo.boundingBox.min.y + geo.boundingBox.max.y) / 2 / MM;
    } else if (c.kind === 'box') {
      mesh = new THREE.Mesh(new THREE.BoxGeometry(c.size[0] * MM, c.size[1] * MM, c.size[2] * MM), mat);
      mesh.position.set(c.center[0] * MM, c.center[1] * MM, c.center[2] * MM);
      mesh.quaternion.copy(basisQuat(c.x, c.z));
      cy = c.center[1];
    } else {
      const ax = new V3(...c.axis).normalize();
      mesh = new THREE.Mesh(new THREE.CylinderGeometry(c.r * MM, c.r * MM, c.h * MM, c.r * MM > 6 ? 40 : 20), mat);
      mesh.position.set(c.top[0] * MM, c.top[1] * MM, c.top[2] * MM).addScaledVector(ax, -c.h * MM / 2);
      mesh.quaternion.setFromUnitVectors(YAX, ax);
      cy = c.top[1];
    }
    mesh.userData.info = info(c);
    if (c.id.includes('.sink_')) mesh.material = finMat;
    if (c.id.startsWith('psu')) mesh.material = perfMat;
    if (c.id.endsWith('.fan')) group(g).add(fanRotor(c));
    if (c.mat === 'acrylic' || c.mat === 'glass') mesh.renderOrder = 4;
    if (c.id.endsWith('.lens') && c.group === 'uv') { mesh.material = mesh.material.clone(); hw.uvLens.push(mesh); }
    if (c.id.endsWith('.diffuser')) { mesh.material = new THREE.MeshStandardMaterial({ color: 0xf4f1ea, roughness: 0.6, emissive: 0xff2010, emissiveIntensity: 0 }); hw.diffusers.push(mesh); }
    if (CAGE.has(g) && isFrontPt(cy)) hw.front.push(mesh);
    group(g).add(mesh); hw.pick.push(mesh);
  }
  // transducers: instanced, with the mesh front drawn on the radiating face
  const canGeo = new THREE.CylinderGeometry(4.95, 4.95, 7.1, 20);
  const canMats = [material('metal', '#c9ccd1'), new THREE.MeshStandardMaterial({ map: canTopTexture(), metalness: 0.55, roughness: 0.5 }), material('metal', '#c9ccd1')];
  const dummy = new THREE.Object3D();
  for (const key of ['back', 'front']) {
    const list = tr[key];
    const inst = new THREE.InstancedMesh(canGeo, canMats, list.length);
    list.forEach((c, i) => {
      const ax = new V3(...c.axis).normalize();
      dummy.position.set(c.top[0] * MM, c.top[1] * MM, c.top[2] * MM).addScaledVector(ax, -3.55);
      dummy.quaternion.setFromUnitVectors(YAX, ax); dummy.scale.set(1, 1, 1);
      dummy.updateMatrix(); inst.setMatrixAt(i, dummy.matrix);
    });
    inst.instanceMatrix.needsUpdate = true;
    inst.userData.infoFn = () => ({ label: '40 kHz transducer (10 mm can)', mount: 'soldered to its TB-25 board (2 leads)' });
    group('transducers').add(inst); hw.pick.push(inst);
    if (key === 'front') hw.front.push(inst);
  }
  // board electronics: every placed part of the TB-25, on all 18 boards
  const byCol = {};
  for (const b of pcb.bodies) (byCol[b.col + (b.round ? 'r' : '')] ||= []).push(b);
  const unitBox = new THREE.BoxGeometry(1, 1, 1), unitCyl = new THREE.CylinderGeometry(0.5, 0.5, 1, 16).rotateX(Math.PI / 2);
  for (const key of ['back', 'front']) {
    const boards = m.boards.filter((b) => isFrontPt(b.c[1]) === (key === 'front'));
    for (const [ck, list] of Object.entries(byCol)) {
      const round = ck.endsWith('r'), col = round ? ck.slice(0, -1) : ck;
      const metal = col === '#c9ccd1' || col === '#d4af37';
      const inst = new THREE.InstancedMesh(round ? unitCyl : unitBox, material(metal ? 'metal' : 'plastic', col), boards.length * list.length);
      let k = 0;
      for (const b of boards) {
        const c = new V3(...b.c).multiplyScalar(MM), u = new V3(...b.u), v = new V3(...b.v), n = new V3(...b.n);
        const q = basisQuat(b.u, b.n);
        for (const p of list) {
          dummy.position.copy(c).addScaledVector(u, p.x).addScaledVector(v, p.y).addScaledVector(n, p.z / 2);
          dummy.quaternion.copy(q);
          // box axes: x = u, y = n x u = -v, z = n
          dummy.scale.set(p.w, p.h, p.z);
          dummy.updateMatrix(); inst.setMatrixAt(k++, dummy.matrix);
        }
      }
      inst.instanceMatrix.needsUpdate = true;
      const refs = list.map((p) => p.ref + ' (' + p.part + ')');
      inst.userData.infoFn = (id) => ({ label: 'TB-25 board, ' + refs[id % list.length], mount: 'soldered' });
      group('boardparts').add(inst); hw.pick.push(inst);
      if (key === 'front') hw.front.push(inst);
    }
  }
  // wiring
  const wg = group('wires');
  for (const w of m.wires) {
    const pts = [];
    for (let i = 0; i < w.pts.length; i++) {
      const p = new V3(w.pts[i][0] * MM, w.pts[i][1] * MM, w.pts[i][2] * MM);
      if (i > 0) { const a = pts[pts.length - 1], n = Math.floor(a.distanceTo(p) / 7); for (let k = 1; k <= n; k++) pts.push(a.clone().lerp(p, k / (n + 1))); }
      pts.push(p);
    }
    const curve = new THREE.CatmullRomCurve3(pts, false, 'centripetal');
    const len = curve.getLength();
    const tube = new THREE.Mesh(new THREE.TubeGeometry(curve, Math.max(8, Math.round(len / 2.5)), w.r * MM, w.fn === 'usb' ? 8 : 5, false),
      material(w.fn === 'resin' ? 'glass' : 'plastic', w.color));
    const lg = m.wire_legend[w.fn];
    tube.userData.info = { label: w.label + ' — ' + (lg ? lg.meaning : w.fn), mount: `${w.a.replace(/_/g, ' ')} → ${w.b.replace(/_/g, ' ')} · ${(len).toFixed(0)} mm` };
    wg.add(tube); hw.pick.push(tube);
    if (pts.reduce((a, p) => a + p.y, 0) / pts.length < -30 && pts.reduce((a, p) => a + p.z, 0) / pts.length > -100) hw.front.push(tube);   // hidden by the cutaway
  }
  const used = new Set(m.wires.map((w) => w.fn));
  $('wireLegend').innerHTML = Object.entries(m.wire_legend).filter(([k]) => used.has(k))
    .map(([, v]) => `<i style="background:${v.color}"></i><span>${v.meaning}</span>`).join('');
  // lights inside the cage: red strobes (lower ports) and the 405 nm cure emitters, both aimed at the build centre
  for (const l of m.lights) {
    const p = new V3(l.pos[0] * MM, l.pos[1] * MM, l.pos[2] * MM);
    if (l.kind === 'strobe') {
      const pl = new THREE.PointLight(0xff3018, 0, 0, 2); pl.position.copy(p).addScaledVector(new V3(...l.dir), 3);
      scene.add(pl); hw.lights.push({ kind: 'strobe', light: pl });
    } else if (l.kind === 'uv') {
      const sl = new THREE.SpotLight(0x7b2cff, 0, 0, 0.21, 0.7, 2); sl.position.copy(p);
      sl.target.position.set(0, 0, 0); scene.add(sl); scene.add(sl.target); hw.lights.push({ kind: 'uv', light: sl });
    }
  }
  hw.cams = m.cameras;
  NOZ.set(m.lights.find((l) => l.kind === 'jet').pos[0] * MM, 0, m.params.nozzle_tip_z * MM);
  PARK.set(0, 0, (m.params.park_z ?? -0.02) * MM);
  if (m.name) document.querySelector('#hud h1').textContent = m.name + ' — acoustic-levitation resin printer';
  applyToggles();
}

function applyToggles() {
  const cam = mode === 'cam';
  const vis = { enc: cam || $('tEnc').checked, door: cam || ($('tEnc').checked && $('tDoor').checked), boards: cam || $('tBoards').checked, wires: cam || $('tWires').checked };
  for (const [g, grp] of Object.entries(hw.groups)) grp.visible = CAT[g] ? vis[CAT[g]] : true;
  const cut = !cam && $('tCut').checked;
  for (const o of hw.front) o.visible = !cut;
  if (ghost) ghost.visible = $('tGhost').checked && mode === 'print';
  if (pathLine) pathLine.visible = $('tPath').checked && mode === 'print';
}
for (const id of ['tEnc', 'tDoor', 'tCut', 'tBoards', 'tWires', 'tGhost', 'tPath']) $(id).addEventListener('change', applyToggles);

// hover: what is this part, and what is it fastened to
const ray = new THREE.Raycaster(); const mouse = new THREE.Vector2(); let lastPick = 0;
canvas.addEventListener('pointermove', (e) => {
  const now = performance.now(); if (now - lastPick < 70 || mode === 'cam' || e.buttons) { if (e.buttons) $('tip').style.display = 'none'; return; }
  lastPick = now;
  const r = canvas.getBoundingClientRect();
  mouse.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(mouse, camera);
  const hit = ray.intersectObjects(hw.pick.filter((o) => o.visible && o.parent.visible), false)[0];
  const tip = $('tip');
  if (!hit) { tip.style.display = 'none'; return; }
  const inf = hit.object.userData.infoFn ? hit.object.userData.infoFn(hit.instanceId) : hit.object.userData.info;
  if (!inf) { tip.style.display = 'none'; return; }
  tip.innerHTML = `<b></b><span></span>`; tip.firstChild.textContent = inf.label; tip.lastChild.textContent = inf.mount || '';
  tip.style.display = 'block';
  tip.style.left = Math.min(e.clientX - r.left + 14, r.width - 350) + 'px'; tip.style.top = (e.clientY - r.top + 14) + 'px';
});
canvas.addEventListener('pointerleave', () => { $('tip').style.display = 'none'; });

// ------------------------------------------------------------------ print playback
let mode = 'print';
let meta = null, dep = null, nDep = 0, cum = new Float64Array(1), blobs = null, ghost = null;
let simTime = 0, playing = true, lastShown = -1;
const printGroup = new THREE.Group(); scene.add(printGroup);
const dummy = new THREE.Object3D();
const colLo = new THREE.Color('#c2410c'), colHi = new THREE.Color('#fbbf24'), colTmp = new THREE.Color();

const dropMat = new THREE.MeshStandardMaterial({ color: 0xffe6a8, roughness: 0.08, metalness: 0.0, emissive: 0x7a4a00, emissiveIntensity: 0.5 });
const droplet = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16), dropMat);
droplet.visible = false; scene.add(droplet);
const echoes = [0, 1, 2].map(() => { const m = new THREE.Mesh(droplet.geometry, dropMat); m.visible = false; scene.add(m); return m; });   // multi-flash strobe images
let pathLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x58a6ff, transparent: true, opacity: 0.75 }));
pathLine.frustumCulled = false; printGroup.add(pathLine);
let uvOn = false, uvSite = null;

const PARK = new V3(0, 0, -20), NOZ = new V3(0, 0, -82);     // overwritten from machine.json
const T_RISE = 0.068, T_HOLD = 0.02, T_PRESS = 0.022, T_CURE = 0.020, A_MAX = 0.5 * G, GAP0 = 1.2, STRAIGHT = 1.5;   // as in the landing tests
const T_DIRECT = 0.113;        // direct ascent: jet -> stand-off under the site, no parking (lands at ~0.14 s, as simulated)

function siteOf(i) {
  const o = i * 10;
  return { c: new V3(dep[o] * MM, dep[o + 1] * MM, dep[o + 2] * MM), r: dep[o + 3] * MM,
           p: new V3(dep[o + 4] * MM, dep[o + 5] * MM, dep[o + 6] * MM), n: new V3(dep[o + 7], dep[o + 8], dep[o + 9]).normalize() };
}

// Same construction as replicator/delivery.py (direct ascent, or spiral round a bounding sphere and turn onto the normal).
function buildPath(i) {
  const s = siteOf(i), a = meta.drop_d * MM / 2;
  const ctr = new V3(meta.bound_center[0] * MM, meta.bound_center[1] * MM, meta.bound_center[2] * MM);
  const Rs = meta.bound_radius * MM + 5;
  const land = s.p.clone().addScaledVector(s.n, a);
  const xpre = s.p.clone().addScaledVector(s.n, a + GAP0);
  const ps = xpre.clone().addScaledVector(s.n, STRAIGHT);
  const oc = s.p.clone().sub(ctr), b = oc.dot(s.n);
  let tb = -b + Math.sqrt(Math.max(b * b - (oc.lengthSq() - Rs * Rs), 0));
  tb = Math.max(tb, a + GAP0 + STRAIGHT + 2);
  const pb = s.p.clone().addScaledVector(s.n, tb);
  const va = PARK.clone().sub(ctr), vb = pb.clone().sub(ctr);
  const ra = va.length(), rb = vb.length();
  const ua = va.clone().normalize(), ub = vb.clone().normalize();
  const ang = Math.acos(THREE.MathUtils.clamp(ua.dot(ub), -1, 1));
  const lat0 = Math.hypot(xpre.x - NOZ.x, xpre.y - NOZ.y);
  if (s.n.z < -0.6 && lat0 < 16) {                   // direct ascent
    const pts = [];
    for (let k = 0; k <= 14; k++) {
      const u = k / 14, sl = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5, zf = 2 * u - u * u;
      pts.push(new V3(NOZ.x + (xpre.x - NOZ.x) * sl, NOZ.y + (xpre.y - NOZ.y) * sl, NOZ.z + (xpre.z - NOZ.z) * zf));
    }
    const curve = new THREE.CatmullRomCurve3(pts, false, 'centripetal');
    return { curve, L: curve.getLength(), xpre, land, site: s, direct: true };
  }
  const pts = [];
  const N = 22;
  for (let k = 0; k <= N; k++) {
    const f = k / N;
    let u;
    if (ang < 1e-4) u = ua.clone();
    else u = ua.clone().multiplyScalar(Math.sin((1 - f) * ang) / Math.sin(ang)).addScaledVector(ub, Math.sin(f * ang) / Math.sin(ang));
    const r = Math.max(ra + (rb - ra) * (3 * f * f - 2 * f * f * f), Math.min(ra, rb));
    pts.push(ctr.clone().addScaledVector(u, r));
  }
  pts.push(ps, xpre);
  const curve = new THREE.CatmullRomCurve3(pts, false, 'centripetal');
  return { curve, L: curve.getLength(), xpre, land, site: s };
}
function phases(path) {
  const tRise = path.direct ? 0 : T_RISE, tHold = path.direct ? 0.004 : T_HOLD;
  const tTransit = path.direct ? T_DIRECT : Math.sqrt(5.7735 * (path.L / MM) / A_MAX);
  return { tRise, tTransit, tHold, tLand: tRise + tTransit + tHold + T_PRESS };
}
function durationOf(path) { return phases(path).tLand + T_CURE; }
function flightPos(path, t) {            // where the droplet is t seconds after it left the nozzle
  const ph = phases(path);
  if (t < 0) return null;
  if (t < ph.tRise) { const u = t / ph.tRise; return NOZ.clone().lerp(PARK, 2 * u - u * u); }
  if (t < ph.tRise + ph.tTransit) {
    const u = (t - ph.tRise) / ph.tTransit;
    return path.curve.getPointAt(path.direct ? u : 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5);
  }
  if (t < ph.tRise + ph.tTransit + ph.tHold) return path.xpre.clone();
  const u = Math.min((t - ph.tRise - ph.tTransit - ph.tHold) / T_PRESS, 1);
  return path.xpre.clone().lerp(path.land, u * u);
}

function onDeposits(buf) {
  const n = Math.floor(buf.byteLength / 40);
  dep = new Float32Array(buf, 0, n * 10);
  const cap = Math.max(meta.n_sites, n) + 2000;
  if (!blobs || blobs.userData.cap < n) {
    if (blobs) printGroup.remove(blobs);
    const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.3, metalness: 0.0 });
    blobs = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 1), mat, cap);
    blobs.userData.cap = cap; blobs.count = 0; blobs.frustumCulled = false;
    printGroup.add(blobs);
    nDep = 0;
  }
  const c2 = new Float64Array(n + 1); c2.set(cum.subarray(0, Math.min(cum.length, nDep + 1)));
  const zTop = meta.pole_tip[2] * MM, zSpan = 20;
  for (let i = nDep; i < n; i++) {
    const s = siteOf(i);
    dummy.position.copy(s.c); dummy.scale.setScalar(s.r); dummy.quaternion.identity(); dummy.updateMatrix();
    blobs.setMatrixAt(i, dummy.matrix);
    colTmp.copy(colLo).lerp(colHi, THREE.MathUtils.clamp((zTop - s.c.z) / zSpan, 0, 1));
    blobs.setColorAt(i, colTmp);
    c2[i + 1] = c2[i] + durationOf(buildPath(i));
  }
  cum = c2; nDep = n;
  blobs.instanceMatrix.needsUpdate = true;
  if (blobs.instanceColor) blobs.instanceColor.needsUpdate = true;
  $('scrub').max = String(n);
  $('empty').style.display = n ? 'none' : 'flex';
}

let PRINT = new URLSearchParams(location.search).get('print') || 'print';     // which out/<folder> to play back
async function pollPrint() {
  try {
    const m = await (await fetch(`/out/${PRINT}/meta.json?` + Date.now())).json();
    if (!meta || m.n_written !== meta.n_written || m.done !== meta.done || m.n_sites !== meta.n_sites) {
      const restart = meta && m.n_written < meta.n_written;
      meta = m;
      const buf = await (await fetch(`/out/${PRINT}/deposits.bin?` + Date.now())).arrayBuffer();
      if (restart) { nDep = 0; simTime = 0; cum = new Float64Array(1); if (blobs) { printGroup.remove(blobs); blobs = null; }
        if (surf) { printGroup.remove(surf); surf = null; surfKey = ''; } }
      onDeposits(buf);
    }
  } catch (e) { if (!meta) $('empty').style.display = 'flex'; }
  if (!ghost) loadGhost();
  loadSurface();
}

async function loadGhost() {
  try {
    const r = await fetch('/out/viewer/target.bin?' + Date.now());
    if (!r.ok) return;
    const f = new Float32Array(await r.arrayBuffer());
    const g = new THREE.BufferGeometry();
    const pos = new Float32Array(f.length); for (let i = 0; i < f.length; i++) pos[i] = f[i] * MM;
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.computeVertexNormals();
    ghost = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: 0x7dd3fc, transparent: true, opacity: 0.07, depthWrite: false, side: THREE.DoubleSide }));
    ghost.renderOrder = 3; printGroup.add(ghost); applyToggles();
  } catch (e) { /* optional */ }
}

function indexAt(t) {            // number of droplets fully cured at machine time t
  let lo = 0, hi = nDep;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (cum[mid] <= t) lo = mid; else hi = mid - 1; }
  return lo;
}
function fmtTime(s) {
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), ss = Math.floor(s % 60);
  return (h ? h + ' h ' : '') + String(m).padStart(h ? 2 : 1, '0') + ' min ' + String(ss).padStart(2, '0') + ' s';
}

// Cured surface from the simulation's own signed-distance field.  Each vertex carries the index of the
// droplet that completed it, so the shader can reveal the surface as the print advances.
let surf = null, surfKey = '', surfMats = null;
const surfUniforms = { uNow: { value: 0 } };
function revealMaterial(opts) {
  const mat = new THREE.MeshStandardMaterial(opts);
  mat.onBeforeCompile = (sh) => {
    sh.uniforms.uNow = surfUniforms.uNow;
    sh.vertexShader = 'attribute float aTime;\nvarying float vTime;\n' + sh.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\nvTime = aTime;');
    sh.fragmentShader = 'uniform float uNow;\nvarying float vTime;\n' + sh.fragmentShader.replace('#include <clipping_planes_fragment>', '#include <clipping_planes_fragment>\nif (vTime > uNow) discard;');
  };
  return mat;
}
async function loadSurface() {
  if (!meta || !meta.surface || !meta.done) { if (surf) { printGroup.remove(surf); surf = null; surfKey = ''; } return; }
  const key = meta.surface.n_verts + ':' + meta.surface.n_faces + ':' + meta.n_written;
  if (key === surfKey) return;
  try {
    const t = '?' + Date.now();
    const [pb, ib, tb] = await Promise.all(['surface_pos.bin', 'surface_idx.bin', 'surface_time.bin']
      .map((f) => fetch(`/out/${PRINT}/` + f + t).then((r) => r.arrayBuffer())));
    const pos = new Float32Array(pb), tim = new Float32Array(tb), col = new Float32Array(pos.length);
    const zTop = meta.pole_tip[2], zSpan = 0.020;
    for (let i = 0; i < pos.length; i += 3) {
      colTmp.copy(colLo).lerp(colHi, THREE.MathUtils.clamp((zTop - pos[i + 2]) / zSpan, 0, 1));
      col[i] = colTmp.r; col[i + 1] = colTmp.g; col[i + 2] = colTmp.b;
      pos[i] *= MM; pos[i + 1] *= MM; pos[i + 2] *= MM;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    g.setAttribute('aTime', new THREE.BufferAttribute(tim, 1));
    g.setIndex(new THREE.BufferAttribute(new Uint32Array(ib), 1));
    g.computeVertexNormals();
    surfMats = { view: revealMaterial({ vertexColors: true, roughness: 0.38, metalness: 0.0, side: THREE.DoubleSide }),
                 real: revealMaterial({ color: 0xe6dcb4, roughness: 0.22, metalness: 0.0, side: THREE.DoubleSide }) };   // cured clear resin
    if (surf) printGroup.remove(surf);
    surf = new THREE.Mesh(g, mode === 'cam' ? surfMats.real : surfMats.view); surf.frustumCulled = false; printGroup.add(surf); surfKey = key;
  } catch (e) { /* keep spheres */ }
}

const FRESH = 900;            // most recent droplets shown as individual blobs on top of the cured surface
let curPath = null, curIdx = -1, curPos = null, flightT = 0;
function stepPrint(dtWall) {
  if (!dep || !nDep) return;
  const speed = parseFloat($('speed').value);
  if (playing) simTime = Math.min(simTime + dtWall * speed, cum[nDep]);
  const k = indexAt(simTime);
  if (surf) {
    surfUniforms.uNow.value = k;
    if (blobs && blobs.userData.shown !== k) {       // only the freshest droplets as blobs
      const i0 = Math.max(0, k - FRESH);
      for (let i = i0; i < k; i++) {
        const sdp = siteOf(i);
        dummy.position.copy(sdp.c); dummy.scale.setScalar(Math.min(sdp.r, meta.drop_d * MM * 0.75)); dummy.quaternion.identity(); dummy.updateMatrix();
        blobs.setMatrixAt(i - i0, dummy.matrix);
        colTmp.set(0xffe2a8); blobs.setColorAt(i - i0, colTmp);
      }
      blobs.count = k - i0; blobs.userData.shown = k;
      blobs.instanceMatrix.needsUpdate = true; if (blobs.instanceColor) blobs.instanceColor.needsUpdate = true;
    }
  } else if (blobs) blobs.count = k;
  // current droplet in flight, and the cure flash that follows its landing
  const showFlight = speed <= 25 && k < nDep;
  uvOn = false; curPos = null;
  for (const e of echoes) e.visible = false;
  if (showFlight) {
    if (curIdx !== k) { curPath = buildPath(k); curIdx = k;
      pathLine.geometry.dispose(); pathLine.geometry = new THREE.BufferGeometry().setFromPoints(curPath.curve.getPoints(60).concat([curPath.land])); }
    const t = simTime - cum[k], a = meta.drop_d * MM / 2, ph = phases(curPath);
    flightT = t;
    const p = flightPos(curPath, t);
    droplet.position.copy(p); droplet.scale.setScalar(a); droplet.visible = true; curPos = p;
    if (mode === 'cam') echoes.forEach((e, j) => {            // the strobe fires four times per frame, 4 ms apart
      const q = flightPos(curPath, t - 0.004 * (j + 1));
      if (q && t < ph.tLand) { e.position.copy(q); e.scale.setScalar(a); e.visible = true; }
    });
    pathLine.visible = $('tPath').checked && mode === 'print';
    if (t >= ph.tLand) { uvOn = true; uvSite = curPath.site; }
  } else {
    droplet.visible = false; pathLine.visible = false;
    if (playing && k < nDep && k > 0) { uvOn = Math.random() < 0.5; uvSite = siteOf(k - 1); }   // too fast to resolve: flicker
  }
  if (k !== lastShown) {
    lastShown = k;
    if (document.activeElement !== $('scrub')) $('scrub').value = String(k);
  }
  const vol = k * (4 / 3) * Math.PI * (meta.drop_d * MM / 2) ** 3 * 0.95;
  const zNow = k > 0 ? siteOf(Math.max(k - 1, 0)).p.z : meta.pole_tip[2] * MM;
  $('stats').innerHTML =
    `<div><span>droplets cured</span><span>${k.toLocaleString()} / ${Math.max(meta.n_sites, nDep).toLocaleString()}</span></div>` +
    `<div><span>machine time</span><span>${fmtTime(simTime)}</span></div>` +
    `<div><span>full print (est.)</span><span>${fmtTime(cum[nDep] * Math.max(meta.n_sites, nDep) / Math.max(nDep, 1))}</span></div>` +
    (meta.skin === 'fine' ? '' : `<div><span>resin deposited</span><span>${vol.toFixed(1)} mm³</span></div>`) +
    `<div><span>working height</span><span>z = ${zNow.toFixed(2)} mm</span></div>` +
    `<div><span>droplet / cycle</span><span>${(meta.drop_d * MM).toFixed(2)} mm${meta.skin === 'fine' ? ' core, ' + (meta.skin_d * MM).toFixed(2) + ' mm shell' : ''} · ${(1000 * cum[nDep] / Math.max(nDep, 1)).toFixed(0)} ms</span></div>` +
    (meta.done ? '' : `<div><span>simulation</span><span style="color:#58a6ff">still computing…</span></div>`);
}

// ------------------------------------------------------------------ physics test flights
const physGroup = new THREE.Group(); physGroup.visible = false; scene.add(physGroup);
const phys = { files: {}, runs: [], t: 0, marker: null, R: 0 };
async function pollPhysics() {
  try {
    const names = await (await fetch('/api/list?' + Date.now())).json();
    let changed = false;
    for (const nm of names) {
      if (phys.files[nm] || !nm.includes('rco5')) continue;
      phys.files[nm] = await (await fetch('/out/viewer/' + nm + '?' + Date.now())).json();
      changed = true;
    }
    if (changed) rebuildPhysics();
  } catch (e) { /* none yet */ }
}
function rebuildPhysics() {
  while (physGroup.children.length) physGroup.remove(physGroup.children[0]);
  phys.runs = [];
  let R = 0;
  for (const f of Object.values(phys.files)) {
    R = Math.max(R, (f.R || 0) * MM);
    for (const run of f.runs) {
      const act = run.pts.map((q) => new V3(q[1] * MM, q[2] * MM, q[3] * MM));
      const col = run.status === 'landed' ? 0x3fb950 : 0xf85149;
      physGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(act), new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.55 })));
      phys.runs.push({ t: run.pts.map((q) => q[0]), act, label: `${f.scene} · ${f.site} · ${run.label}`, status: run.status, err: run.err_um });
    }
    if (f.plan) {
      const pl = f.plan.map((q) => new V3(q[0] * MM, q[1] * MM, q[2] * MM));
      physGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pl), new THREE.LineBasicMaterial({ color: 0x8b98a8, transparent: true, opacity: 0.5 })));
    }
  }
  if (R > 0) { physGroup.add(new THREE.Mesh(new THREE.SphereGeometry(R, 48, 32), new THREE.MeshStandardMaterial({ color: 0xd97706, roughness: 0.5 }))); phys.R = R; }
  phys.marker = new THREE.Mesh(new THREE.SphereGeometry(0.3, 16, 12), new THREE.MeshStandardMaterial({ color: 0xffd27a, emissive: 0xb86e00, emissiveIntensity: 0.7 }));
  physGroup.add(phys.marker);
}
let physIdx = 0;
function stepPhysics(dtWall) {
  if (!phys.runs.length) { $('stats').innerHTML = '<div><span>physics test flights</span><span>none saved yet</span></div>'; return; }
  const run = phys.runs[physIdx % phys.runs.length];
  if (playing) phys.t += dtWall * 0.02 * Math.max(parseFloat($('speed').value), 1) ** 0.5;   // 50x slow motion at 1x
  const T = run.t[run.t.length - 1];
  if (phys.t > T + 0.004) { phys.t = run.t[0]; physIdx++; return; }
  let j = 0; while (j < run.t.length - 2 && run.t[j + 1] < phys.t) j++;
  const f = THREE.MathUtils.clamp((phys.t - run.t[j]) / Math.max(run.t[j + 1] - run.t[j], 1e-9), 0, 1);
  phys.marker.position.copy(run.act[j]).lerp(run.act[j + 1], f);
  $('stats').innerHTML =
    `<div><span>test flight</span><span>${(physIdx % phys.runs.length) + 1} / ${phys.runs.length}</span></div>` +
    `<div><span>case</span><span>${run.label}</span></div>` +
    `<div><span>flight time</span><span>${(1000 * Math.max(phys.t, 0)).toFixed(0)} / ${(1000 * T).toFixed(0)} ms</span></div>` +
    `<div><span>outcome</span><span style="color:${run.status === 'landed' ? '#3fb950' : '#f85149'}">${run.status}${run.err != null ? ' · ' + run.err.toFixed(0) + ' µm off target' : ''}</span></div>`;
}

// ------------------------------------------------------------------ in-camera view
// The scene is rendered from a machine camera's real position with its real lens, lit only by what lights the
// real cage (red strobes, 405 nm flash), then passed through a sensor model: barrel distortion, the sensor's
// outline (the board is mounted rolled 45 degrees so its long diagonal follows the flight path), vignetting,
// a monochrome response and noise.
const K1 = -0.2;                                  // barrel distortion of a 2.8 mm M12 lens
const T_RT = 1.04;                                // half-width of the rendered field, in tangent units
const mcam = new THREE.PerspectiveCamera(2 * Math.atan(T_RT) * 180 / Math.PI, 1, 0.4, 900);
const rt = new THREE.WebGLRenderTarget(1536, 1536, { type: THREE.HalfFloatType, samples: 4 });
const post = {
  scene: new THREE.Scene(), cam: new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1),
  mat: new THREE.ShaderMaterial({
    uniforms: { tex: { value: rt.texture }, aspect: { value: 1 }, mono: { value: 1 }, time: { value: 0 }, roll: { value: Math.PI / 4 },
                sHalf: { value: new THREE.Vector2(0.848, 0.530) }, tHalf: { value: 0.8086 }, tRt: { value: T_RT }, k1: { value: K1 }, gain: { value: 1.0 } },
    vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
    fragmentShader: `
      varying vec2 vUv; uniform sampler2D tex; uniform float aspect, mono, time, roll, tHalf, tRt, k1, gain; uniform vec2 sHalf;
      float hash(vec2 p){ return fract(sin(dot(p, vec2(12.9898, 78.233)) + time) * 43758.5453); }
      void main(){
        vec2 q = (vUv - 0.5) * 2.0 * vec2(max(aspect, 1.0), max(1.0 / aspect, 1.0)) / 0.97;   // sensor plane, half-diagonal = 1
        float c = cos(roll), s = sin(roll);
        vec2 sq = vec2(c * q.x + s * q.y, -s * q.x + c * q.y);
        if (abs(sq.x) > sHalf.x || abs(sq.y) > sHalf.y) { gl_FragColor = vec4(0.012, 0.014, 0.018, 1.0); return; }
        float rd = length(q) * tHalf, t = rd;
        for (int i = 0; i < 5; i++) t = rd / (1.0 + k1 * t * t);                              // undo the lens distortion
        vec2 uv = (length(q) > 1e-5 ? q / length(q) : vec2(0.0)) * t / tRt * 0.5 + 0.5;
        vec3 col = texture2D(tex, uv).rgb * gain;
        col *= pow(1.0 / (1.0 + t * t), 1.5);                                                 // lens fall-off
        float n = hash(floor(sq * 640.0));
        if (mono > 0.5) {
          float g = dot(col, vec3(0.62, 0.25, 0.55));                                         // silicon sees red and violet well
          g = g / (1.0 + 0.55 * g);
          g += (n - 0.5) * 0.035 * (0.4 + sqrt(max(g, 0.0)));
          col = vec3(pow(clamp(g, 0.0, 1.0), 1.0 / 2.2));
        } else {
          col = col / (1.0 + 0.55 * col);
          col += (n - 0.5) * 0.02;
          col = pow(clamp(col, 0.0, 1.0), vec3(1.0 / 2.2));
        }
        gl_FragColor = vec4(col, 1.0);
      }`,
  }),
};
post.scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), post.mat));
let camIdx = 0, frameNo = 0;
function aimCamera() {
  const c = hw.cams[camIdx]; if (!c) return;
  mcam.position.set(c.pos[0] * MM, c.pos[1] * MM, c.pos[2] * MM);
  mcam.up.set(...c.up);
  mcam.lookAt(mcam.position.clone().add(new V3(...c.dir)));
  mcam.updateMatrixWorld();
}
function setLights(camMode) {
  const colour = $('camLook').value === 'colour';
  hemi.intensity = camMode ? 0.012 : 0.75; sun.intensity = camMode ? 0 : 1.5; fill.intensity = camMode ? 0 : 0.45;
  scene.environment = camMode ? null : ENV;
  scene.background = camMode ? new THREE.Color(0x000000) : BG;
  for (const l of hw.lights) {
    if (l.kind === 'strobe') l.light.intensity = camMode ? 26000 : 0;
    else l.light.intensity = uvOn ? (camMode ? 3.2e5 : 6e5) : 0;
  }
  for (const d of hw.diffusers) d.material.emissiveIntensity = camMode ? 3.0 : 0;
  for (const u of hw.uvLens) { u.material.emissive = new THREE.Color(uvOn ? 0x7b2cff : 0x000000); u.material.emissiveIntensity = uvOn ? 4 : 0; }
  dropMat.emissiveIntensity = camMode ? 0.0 : 0.5;
  post.mat.uniforms.mono.value = colour ? 0 : 1;
}
function renderCamera(now) {
  aimCamera();
  setLights(true);
  renderer.setRenderTarget(rt); renderer.render(scene, mcam); renderer.setRenderTarget(null);
  const w = canvas.clientWidth, h = canvas.clientHeight;
  post.mat.uniforms.aspect.value = w / Math.max(h, 1);
  post.mat.uniforms.time.value = (now % 1000) / 1000;
  renderer.render(post.scene, post.cam);
  // overlay: what the tracker reports for this frame
  frameNo++;
  const c = hw.cams[camIdx];
  $('camInfo').innerHTML = `CAM ${camIdx + 1} · OV9281 global shutter · 1280×800 · ${c.f_mm} mm lens · sensor rolled 45°<br>` +
    ($('camLook').value === 'mono' ? 'as recorded: monochrome, exposure locked to the red strobe (4 flashes / frame)' : 'what you would see by eye through the same port') ;
  $('uvFlag').style.visibility = uvOn ? 'visible' : 'hidden';
  $('camFrame').textContent = `frame ${String(frameNo).padStart(6, '0')} · t = ${(simTime).toFixed(3)} s`;
  const trk = $('trk');
  if (curPos && droplet.visible && mode === 'cam') {
    const v = curPos.clone().applyMatrix4(mcam.matrixWorldInverse);
    const tx = v.x / -v.z, ty = v.y / -v.z, tr = Math.hypot(tx, ty);
    const rd = tr * (1 + K1 * tr * tr) / 0.8086;                         // radius on the sensor, half-diagonal = 1
    const qx = tr > 1e-9 ? tx / tr * rd : 0, qy = tr > 1e-9 ? ty / tr * rd : 0;
    const S = 0.97 * Math.min(w, h) / 2;
    const sx = (qx + qy) * Math.SQRT1_2, sy = (-qx + qy) * Math.SQRT1_2;   // sensor axes
    if (v.z < 0 && Math.abs(sx) < 0.848 && Math.abs(sy) < 0.530) {
      trk.style.display = 'block'; trk.style.left = (w / 2 + qx * S) + 'px'; trk.style.top = (h / 2 - qy * S) + 'px';
      const px = 640 + sx * 2.264 / 0.003, py = 400 - sy * 2.264 / 0.003, dist = -v.z;
      $('trkText').textContent = `droplet · pixel (${px.toFixed(0)}, ${py.toFixed(0)}) · ${(meta.drop_d * MM / dist * c.f_mm / 0.003).toFixed(1)} px wide · ${dist.toFixed(0)} mm away`;
    } else trk.style.display = 'none';
  } else trk.style.display = 'none';
}

function setMode(m) {
  mode = m;
  $('mPrint').classList.toggle('on', m === 'print'); $('mPhys').classList.toggle('on', m === 'phys'); $('mCam').classList.toggle('on', m === 'cam');
  printGroup.visible = m !== 'phys'; physGroup.visible = m === 'phys';
  droplet.visible = false; for (const e of echoes) e.visible = false;
  $('scrub').disabled = m === 'phys';
  $('camhud').style.display = m === 'cam' ? 'block' : 'none';
  $('orbitRow').style.display = m === 'cam' ? 'none' : 'flex'; $('camRow').style.display = m === 'cam' ? 'flex' : 'none';
  $('toggles').style.display = m === 'cam' ? 'none' : 'block';
  $('modeLine').style.display = m === 'cam' ? 'none' : 'block';
  $('modeLine').textContent = m === 'print'
    ? 'Simulated print playback · drag to orbit, scroll to zoom, right-drag to pan · hover a part to see what it is and how it is fastened'
    : m === 'cam' ? 'What a machine camera sees during the print · the cage is dark; red strobes light each frame, the violet flash cures each droplet'
    : 'Full acoustic simulations of single droplet deliveries (imperfect hardware) · 50× slow motion, looping';
  controls.enabled = m !== 'cam';
  if (surf && surfMats) surf.material = m === 'cam' ? surfMats.real : surfMats.view;
  if (m === 'phys') setCam(...CAMS.build);
  applyToggles();
}
$('mPrint').onclick = () => setMode('print');
$('mPhys').onclick = () => setMode('phys');
$('mCam').onclick = () => setMode('cam');
[0, 1, 2].forEach((i) => { $('cam' + i).onclick = () => { camIdx = i; [0, 1, 2].forEach((j) => $('cam' + j).classList.toggle('on', j === i)); }; });

// ------------------------------------------------------------------ live status
async function pollStatus() {
  try {
    const s = await (await fetch('/out/status.json?' + Date.now())).json();
    $('actText').textContent = s.activity || '—';
    $('actTime').textContent = s.updated ? 'updated ' + s.updated : '';
    const v = $('verdict');
    if (s.verdict) { v.style.display = 'block'; v.textContent = s.verdict; } else v.style.display = 'none';
    const box = $('tests'); box.innerHTML = '';
    for (const key of s.order || []) {
      const t = s.tests[key];
      const row = document.createElement('div'); row.className = 'test';
      const dot = document.createElement('div'); dot.className = 'dot ' + t.state;
      const body = document.createElement('div');
      const ti = document.createElement('div'); ti.className = 'ti'; ti.textContent = t.title;
      const de = document.createElement('div'); de.className = 'de'; de.textContent = t.detail || '';
      body.append(ti, de); row.append(dot, body); box.append(row);
    }
  } catch (e) { /* no status yet */ }
}

// ------------------------------------------------------------------ controls + loop
$('play').onclick = () => { playing = !playing; $('play').textContent = playing ? 'Pause' : 'Play'; $('play').classList.toggle('on', playing); };
$('restart').onclick = () => { simTime = 0; phys.t = 0; lastShown = -1; };
$('scrub').addEventListener('input', (e) => { const k = Math.min(parseInt(e.target.value, 10) || 0, nDep); simTime = cum[k]; lastShown = -1; });
$('camAll').onclick = () => setCam(...CAMS.all);
$('camBuild').onclick = () => setCam(...CAMS.build);
$('camPart').onclick = () => setCam(...CAMS.part);
$('camDrawer').onclick = () => { $('tDoor').checked = false; applyToggles(); setCam(...CAMS.drawer); };
$('printSel').onchange = (e) => {            // switch between the simulated prints
  PRINT = e.target.value; meta = null; dep = null; nDep = 0; simTime = 0; cum = new Float64Array(1); lastShown = -1; curIdx = -1;
  if (blobs) { printGroup.remove(blobs); blobs = null; }
  if (surf) { printGroup.remove(surf); surf = null; surfKey = ''; }
  pollPrint();
};
async function listPrints() {              // offer the fine-shell print when it exists
  const sel = $('printSel');
  for (const [dir, name] of [['print', 'Benchy, 0.3 mm droplets (fast)'], ['print_fine', 'Benchy, 0.3 mm core + 0.2 mm shell (fine)'],
    ['print_mc', 'Benchy, 0.3 mm, landing errors taken from the flight tests']]) {
    try { const r = await fetch(`/out/${dir}/meta.json?` + Date.now()); if (!r.ok) continue; const mm = await r.json();
      const o = document.createElement('option'); o.value = dir; o.textContent = name + (mm.done ? '' : ' — computing'); sel.append(o); } catch (e) { /* not there */ }
  }
  if ([...sel.options].some((o) => o.value === 'print_fine' && !o.textContent.includes('computing')) && !new URLSearchParams(location.search).get('print')) PRINT = 'print_fine';
  sel.value = PRINT;
}

const q = new URLSearchParams(location.search);
if (q.get('speed')) $('speed').value = q.get('speed');
setCam(...(CAMS[q.get('cam')] || CAMS.all));
if (q.get('pos') && q.get('tgt')) setCam(q.get('pos').split(',').map(Number), q.get('tgt').split(',').map(Number));
for (const id of ['tEnc', 'tDoor', 'tCut', 'tBoards', 'tWires', 'tGhost', 'tPath']) { if (q.get(id) === '0') $(id).checked = false; if (q.get(id) === '1') $(id).checked = true; }
if (q.get('paused')) { playing = false; }
if (q.get('look')) $('camLook').value = q.get('look');
if (q.get('mcam')) { camIdx = parseInt(q.get('mcam'), 10) || 0; [0, 1, 2].forEach((j) => $('cam' + j).classList.toggle('on', j === camIdx)); }

let last = performance.now();
function frame(now) {
  const dt = Math.min((now - last) / 1000, 0.1); last = now;
  resize();
  if (mode === 'phys') { uvOn = false; stepPhysics(dt); } else stepPrint(dt);
  if (q.get('uv') === '1') uvOn = true;
  if (mode === 'cam') renderCamera(now);
  else { setLights(false); controls.update(); renderer.render(scene, camera); }
  requestAnimationFrame(frame);
}

(async function main() {
  await loadHardware();
  await listPrints();
  if (q.get('frac') || q.get('at')) {
    await pollPrint();
    const k = q.get('at') ? parseInt(q.get('at'), 10) : Math.floor(nDep * parseFloat(q.get('frac')));
    simTime = (cum[Math.min(k, nDep)] || 0) + (parseFloat(q.get('dt') || '0'));
  }
  if (q.get('mode') === 'phys') { await pollPhysics(); setMode('phys'); }
  if (q.get('mode') === 'cam') setMode('cam');
  pollPrint(); pollStatus(); pollPhysics();
  setInterval(pollPrint, 2500); setInterval(pollStatus, 2000); setInterval(pollPhysics, 6000);
  requestAnimationFrame(frame);
})();
