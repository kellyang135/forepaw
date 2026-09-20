import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';

// ---------------------------------------------------------------- telemetry
const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
async function listRuns() {
  try { const r = await fetch('/runs/index.json', { cache: 'no-store' }); return r.ok ? await r.json() : []; } catch { return []; }
}
async function fetchRecords(url) {
  const text = await (await fetch(url, { cache: 'no-store' })).text();
  return text.split('\n').filter((l) => l.trim()).map((l) => JSON.parse(l)).filter((r) => r.schema === 'go2wm.ui.v1');
}
const RUNS = await listRuns();
const LOG = params.get('log') || (RUNS.length ? '/' + RUNS[0] : null);
let RECORDS = [];
if (LOG) { try { RECORDS = await fetchRecords(LOG); } catch { RECORDS = []; } }
const RUN = RECORDS.find((r) => r.type === 'run_start');
if (!RUN) {
  $('loading').innerHTML = `<div class="empty"><b>No telemetry log found.</b><span>Record one, then reload:</span><code>python -m go2wm.ui record --model reference --scenario anomaly --out runs/ui-demo</code><code>python -m go2wm.ui serve</code></div>`;
  throw new Error('no telemetry log');
}
const A = RUN.scene.arena_width_m, H = A / 2, OFF = RUN.scene.frame === 'centered' ? H : 0;
const P = (x, y) => [x + OFF, y + OFF];
const HINTS = RUN.render_hints || {};
const RS = HINTS.robot_scale ?? 1, BOXSIZE = HINTS.box_size_m ?? 0.4;
const OVERLAY = 1;

// ---------------------------------------------------------------- palette
const COL = {
  pred: 0x0f7b53, ok: 0x0f7b53, alarm: 0xd12f35, trail: 0x18181b, future: 0x27272a, ink: 0x18181b,
};
const LIGHT_LOOK = new Set(HINTS.light_appearances || ['blue']);
const PARTS = [['goal', 'goal distance', '#3F3F46'], ['risk', 'failure risk', '#D12F35'], ['stall', 'stall', '#8A8A93'], ['effort', 'effort', '#B4B4BB'], ['change', 'command change', '#D4D4D8']];
const hex = (n) => '#' + n.toString(16).padStart(6, '0');

// sim (x, y) metres -> three (x, 0, z); sim +y points to -z, yaw maps to rotation.y
const W = (x, y, h = 0) => new THREE.Vector3(x - H, h, -(y - H));

// ---------------------------------------------------------------- renderer
const canvas = $('gl');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xecece9);
scene.fog = new THREE.FogExp2(0xecece9, 0.034);
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(renderer), 0.04).texture;


const camera = new THREE.PerspectiveCamera(38, 1, 0.05, 80);
camera.position.set(-5.4 * A / 6, 2.7 * Math.max(0.7, A / 6), 4.3 * A / 6);
camera.layers.enable(OVERLAY);
const controls = new OrbitControls(camera, canvas);
controls.target.set(-0.4 * A / 6, 0.2, -0.4 * A / 6);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.maxPolarAngle = Math.PI * 0.47; controls.minDistance = 1.2; controls.maxDistance = 16;

const rt = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4 });
const composer = new EffectComposer(renderer, rt);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.28, 0.5, 1.0);
bloom.enabled = false;
composer.addPass(new OutputPass());

// ---------------------------------------------------------------- textures
function canvasTex(size, draw, repeat) {
  const c = document.createElement('canvas'); c.width = c.height = size;
  draw(c.getContext('2d'), size);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8;
  if (repeat) { t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(repeat, repeat); }
  return t;
}
function rng(seed) { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); }

const concrete = canvasTex(1024, (g, n) => {
  const r = rng(3);
  g.fillStyle = '#b3b2ad'; g.fillRect(0, 0, n, n);
  for (let i = 0; i < 90; i++) {
    const x = r() * n, y = r() * n, rad = 40 + r() * 220, a = 0.03 + r() * 0.05;
    const grd = g.createRadialGradient(x, y, 0, x, y, rad);
    const tone = r() < 0.5 ? '255,255,255' : '60,58,52';
    grd.addColorStop(0, `rgba(${tone},${a})`); grd.addColorStop(1, `rgba(${tone},0)`);
    g.fillStyle = grd; g.fillRect(x - rad, y - rad, rad * 2, rad * 2);
  }
  const img = g.getImageData(0, 0, n, n), d = img.data;
  for (let i = 0; i < d.length; i += 4) { const k = (r() - 0.5) * 22; d[i] += k; d[i + 1] += k; d[i + 2] += k; }
  g.putImageData(img, 0, 0);
  g.strokeStyle = 'rgba(0,0,0,0.35)'; g.lineWidth = 2;
  g.beginPath(); g.moveTo(0, 1); g.lineTo(n, 1); g.moveTo(1, 0); g.lineTo(1, n); g.stroke();
}, 8);

const hazard = canvasTex(256, (g, n) => {
  g.fillStyle = '#e8b21c'; g.fillRect(0, 0, n, n);
  g.fillStyle = '#15171a';
  for (let i = -n; i < n * 2; i += 64) { g.beginPath(); g.moveTo(i, 0); g.lineTo(i + 32, 0); g.lineTo(i + 32 - n, n); g.lineTo(i - n, n); g.fill(); }
});
hazard.wrapS = THREE.RepeatWrapping;

function cardboardSide() {
  return canvasTex(512, (g, n) => {
    const r = rng(11);
    g.fillStyle = '#b98a52'; g.fillRect(0, 0, n, n);
    for (let i = 0; i < 5000; i++) { g.fillStyle = `rgba(${r() < 0.5 ? '90,60,30' : '230,200,150'},${0.05 + r() * 0.08})`; g.fillRect(r() * n, r() * n, 1 + r() * 3, 1); }
    for (let y = 0; y < n; y += 7) { g.fillStyle = 'rgba(80,55,25,0.05)'; g.fillRect(0, y, n, 2); }
    g.strokeStyle = 'rgba(70,45,20,0.55)'; g.lineWidth = 6; g.strokeRect(3, 3, n - 6, n - 6);
    g.fillStyle = 'rgba(40,28,15,0.8)'; g.font = '700 40px "Chakra Petch", Arial, sans-serif'; g.textAlign = 'center';
    g.fillText('GO2-WM LAB', n / 2, 200);
    g.font = '500 26px "IBM Plex Mono", monospace'; g.fillText('APPEARANCE · BLUE', n / 2, 250);
    g.lineWidth = 8; g.strokeStyle = 'rgba(40,28,15,0.8)';
    for (const x of [n / 2 - 60, n / 2 + 60]) { g.beginPath(); g.moveTo(x, 400); g.lineTo(x, 320); g.moveTo(x - 22, 345); g.lineTo(x, 320); g.lineTo(x + 22, 345); g.stroke(); }
    g.fillStyle = 'rgba(210,190,150,0.55)'; g.fillRect(0, 0, n, 34);
  });
}
function cardboardTop() {
  return canvasTex(512, (g, n) => {
    const r = rng(12);
    g.fillStyle = '#bc8d55'; g.fillRect(0, 0, n, n);
    for (let i = 0; i < 5000; i++) { g.fillStyle = `rgba(${r() < 0.5 ? '90,60,30' : '230,200,150'},${0.05 + r() * 0.08})`; g.fillRect(r() * n, r() * n, 1 + r() * 3, 1); }
    g.fillStyle = 'rgba(60,40,20,0.5)'; g.fillRect(n / 2 - 2, 0, 4, n);
    g.fillStyle = 'rgba(225,205,165,0.72)'; g.fillRect(n / 2 - 46, 0, 92, n);
    g.fillStyle = 'rgba(255,255,255,0.12)'; g.fillRect(n / 2 - 46, 0, 12, n);
  });
}
function crateSide() {
  return canvasTex(512, (g, n) => {
    const r = rng(21);
    g.fillStyle = '#39434d'; g.fillRect(0, 0, n, n);
    for (let i = 0; i < 2500; i++) { g.fillStyle = `rgba(${r() < 0.5 ? '0,0,0' : '255,255,255'},${0.03 + r() * 0.05})`; g.fillRect(r() * n, r() * n, 2 + r() * 6, 1); }
    g.strokeStyle = '#242b32'; g.lineWidth = 14; g.strokeRect(7, 7, n - 14, n - 14);
    g.strokeStyle = 'rgba(255,255,255,0.08)'; g.lineWidth = 2; g.strokeRect(16, 16, n - 32, n - 32);
    g.fillStyle = '#20262c';
    for (const [x, y] of [[26, 26], [n - 26, 26], [26, n - 26], [n - 26, n - 26], [n / 2, 26], [n / 2, n - 26], [26, n / 2], [n - 26, n / 2]]) { g.beginPath(); g.arc(x, y, 7, 0, 7); g.fill(); }
    const s = g.createPattern(hazard.image, 'repeat'); g.fillStyle = s; g.fillRect(16, n - 110, n - 32, 44);
    g.fillStyle = '#d9dde1'; g.font = '700 64px "Chakra Petch", Arial, sans-serif'; g.textAlign = 'center';
    g.fillText('BALLAST', n / 2, 210);
    g.font = '500 24px "IBM Plex Mono", monospace'; g.fillStyle = 'rgba(217,221,225,0.7)'; g.fillText('APPEARANCE · RED', n / 2, 255);
  });
}
function tagTex(seed) {
  return canvasTex(128, (g, n) => {
    const r = rng(seed); const c = n / 8;
    g.fillStyle = '#fff'; g.fillRect(0, 0, n, n); g.fillStyle = '#000'; g.fillRect(c * 0.5, c * 0.5, n - c, n - c);
    for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) if (r() < 0.5) { g.fillStyle = '#fff'; g.fillRect(c * (1 + i), c * (1 + j), c, c); }
  });
}

// ---------------------------------------------------------------- lights
const hemi = new THREE.HemisphereLight(0xffffff, 0xd9d7d1, 0.9); scene.add(hemi);
const key = new THREE.DirectionalLight(0xffffff, 2.1);
key.position.set(-4, 9, 5); key.castShadow = true;
key.shadow.mapSize.set(2048, 2048); key.shadow.camera.left = -6; key.shadow.camera.right = 6; key.shadow.camera.top = 6; key.shadow.camera.bottom = -6;
key.shadow.camera.near = 1; key.shadow.camera.far = 25; key.shadow.bias = -0.0004; key.shadow.normalBias = 0.02; key.shadow.radius = 4;
scene.add(key);
const rim = new THREE.DirectionalLight(0xffffff, 0.45); rim.position.set(6, 3, -6); scene.add(rim);
const alarmLight = new THREE.PointLight(0xff3030, 0, 5, 1.6); scene.add(alarmLight);

// ---------------------------------------------------------------- environment
const floor = new THREE.Mesh(new THREE.PlaneGeometry(60, 60), new THREE.MeshStandardMaterial({ map: concrete, roughness: 0.58, metalness: 0.0, color: 0xffffff, envMapIntensity: 0.25 }));
concrete.repeat.set(20, 20);
floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true; scene.add(floor);

const alu = new THREE.MeshStandardMaterial({ color: 0x9aa3ab, metalness: 0.8, roughness: 0.38, envMapIntensity: 0.45 });
const black = new THREE.MeshStandardMaterial({ color: 0x2a2c30, metalness: 0.35, roughness: 0.55 });
const tapeMat = new THREE.MeshStandardMaterial({ map: hazard, roughness: 0.55 });
function tape(len, x, z, rot) {
  const t = hazard.clone(); t.needsUpdate = true; t.wrapS = THREE.RepeatWrapping; t.repeat.set(len / 0.35, 1);
  const m = new THREE.Mesh(new THREE.PlaneGeometry(len, 0.06), new THREE.MeshStandardMaterial({ map: t, roughness: 0.5 }));
  m.rotation.x = -Math.PI / 2; m.rotation.z = rot; m.position.set(x, 0.002, z); m.receiveShadow = true; scene.add(m);
}
const E = H + 0.04;
tape(A + 0.14, 0, E, 0); tape(A + 0.14, 0, -E, 0); tape(A + 0.14, E, 0, Math.PI / 2); tape(A + 0.14, -E, 0, Math.PI / 2);
void tapeMat;
// perimeter rail on posts
for (const [x, z, rot] of [[0, E + 0.25, 0], [0, -E - 0.25, 0], [E + 0.25, 0, Math.PI / 2], [-E - 0.25, 0, Math.PI / 2]]) {
  const rail = new THREE.Mesh(new THREE.BoxGeometry(A + 0.6, 0.04, 0.04), alu);
  rail.position.set(x, 0.32, z); rail.rotation.y = rot; rail.castShadow = true; scene.add(rail);
  const low = rail.clone(); low.position.y = 0.12; scene.add(low);
  for (let k = -3; k <= 3; k++) {
    const p = new THREE.Mesh(new THREE.BoxGeometry(0.045, 0.36, 0.045), black);
    const off = (k * (A + 0.5)) / 6;
    p.position.set(x + (rot ? 0 : off), 0.18, z + (rot ? off : 0)); p.castShadow = true; scene.add(p);
  }
}
// fiducial tags at arena corners
[[-1, -1], [-1, 1], [1, -1], [1, 1]].forEach(([sx, sz], i) => {
  const m = new THREE.Mesh(new THREE.PlaneGeometry(0.22, 0.22), new THREE.MeshStandardMaterial({ map: tagTex(40 + i), roughness: 0.7 }));
  m.rotation.x = -Math.PI / 2; m.position.set(sx * (H - 0.2), 0.003, sz * (H - 0.2)); scene.add(m);
});
// overhead camera truss
const trussY = 7.2;
for (const z of [-0.35, 0.35]) { const b = new THREE.Mesh(new THREE.BoxGeometry(14, 0.12, 0.12), alu); b.position.set(0, trussY + 0.2, z); scene.add(b); }
const camBody = new THREE.Group();
const cb = new THREE.Mesh(new RoundedBoxGeometry(0.34, 0.2, 0.22, 3, 0.03), black); camBody.add(cb);
const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.07, 0.1, 32), new THREE.MeshStandardMaterial({ color: 0x080a0c, metalness: 0.2, roughness: 0.15 }));
lens.position.y = -0.14; camBody.add(lens);
const led = new THREE.Mesh(new THREE.SphereGeometry(0.012, 12, 12), new THREE.MeshBasicMaterial({ color: 0xff2a2a })); led.position.set(0.12, -0.02, 0.112); camBody.add(led);
camBody.position.set(0, trussY, 0); scene.add(camBody);
// ceiling light panels (bloom sources)
for (const [x, z] of [[-4, -3], [4, -3], [-4, 3], [4, 3], [0, -6], [0, 6]]) {
  const p = new THREE.Mesh(new THREE.PlaneGeometry(2.2, 0.5), new THREE.MeshBasicMaterial({ color: 0xdfe9f5 }));
  p.rotation.x = Math.PI / 2; p.position.set(x, 8.2, z); scene.add(p);
}
// background clutter: shelving silhouettes beyond the arena
for (let i = 0; i < 7; i++) {
  const sh = new THREE.Group();
  for (let lvl = 0; lvl < 4; lvl++) { const s = new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.04, 0.5), black); s.position.y = 0.3 + lvl * 0.55; sh.add(s); }
  for (const x of [-0.88, 0.88]) for (const z of [-0.23, 0.23]) { const u = new THREE.Mesh(new THREE.BoxGeometry(0.04, 2.0, 0.04), alu); u.position.set(x, 1.0, z); sh.add(u); }
  sh.position.set(-6 + i * 2.1, 0, -7.6); sh.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } }); scene.add(sh);
}

// ---------------------------------------------------------------- Go2 model
const L1 = 0.213, L2 = 0.213, STAND = 0.30, FOOT_R = 0.022;
function makeGo2(ghost) {
  const shell = ghost || new THREE.MeshPhysicalMaterial({ color: 0xb4bac0, metalness: 0.1, roughness: 0.42, clearcoat: 0.4, clearcoatRoughness: 0.3, envMapIntensity: 0.45 });
  const dark = ghost || new THREE.MeshStandardMaterial({ color: 0x1a1d21, metalness: 0.55, roughness: 0.45 });
  const rubber = ghost || new THREE.MeshStandardMaterial({ color: 0x0b0c0e, roughness: 0.92 });
  const glass = ghost || new THREE.MeshPhysicalMaterial({ color: 0x05080b, metalness: 0.1, roughness: 0.08, clearcoat: 1 });
  const eye = ghost || new THREE.MeshBasicMaterial({ color: 0x7fe8ff });
  const g = new THREE.Group(), body = new THREE.Group(); g.add(body);
  const add = (geo, mat, x, y, z, parent = body) => { const m = new THREE.Mesh(geo, mat); m.position.set(x, y, z); m.castShadow = !ghost; m.receiveShadow = !ghost; parent.add(m); return m; };
  add(new RoundedBoxGeometry(0.40, 0.11, 0.20, 4, 0.035), shell, 0, 0, 0);
  add(new RoundedBoxGeometry(0.22, 0.022, 0.13, 2, 0.008), dark, -0.03, 0.062, 0);
  add(new THREE.BoxGeometry(0.33, 0.03, 0.206), dark, 0, -0.022, 0);
  for (const z of [-0.104, 0.104]) for (let i = 0; i < 5; i++) add(new THREE.BoxGeometry(0.012, 0.018, 0.004), dark, -0.06 + i * 0.03, 0.02, z);
  add(new RoundedBoxGeometry(0.1, 0.095, 0.17, 3, 0.03), shell, 0.225, 0.008, 0);
  add(new RoundedBoxGeometry(0.014, 0.062, 0.13, 2, 0.006), glass, 0.272, 0.012, 0);
  for (const z of [-0.042, 0.042]) add(new THREE.BoxGeometry(0.004, 0.012, 0.03), eye, 0.28, 0.02, z);
  add(new THREE.CylinderGeometry(0.036, 0.036, 0.03, 32), dark, 0.245, -0.058, 0);
  add(new THREE.SphereGeometry(0.03, 24, 12, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2), glass, 0.245, -0.072, 0);
  add(new THREE.BoxGeometry(0.004, 0.012, 0.12), ghost || new THREE.MeshBasicMaterial({ color: 0xff5a3c }), -0.201, 0.01, 0);
  const legs = [];
  for (const [front, fx] of [[true, 0.1934], [false, -0.1934]]) for (const [left, sz] of [[true, -1], [false, 1]]) {
    const hz = sz * 0.0465;
    const motor = add(new THREE.CylinderGeometry(0.046, 0.046, 0.07, 32), dark, fx, 0, hz); motor.rotation.z = Math.PI / 2;
    const root = new THREE.Group(); root.position.set(fx, 0, hz + sz * 0.0955); body.add(root);
    const hipCap = add(new THREE.CylinderGeometry(0.044, 0.044, 0.05, 32), shell, 0, 0, 0, root); hipCap.rotation.x = Math.PI / 2;
    const thigh = new THREE.Group(); root.add(thigh);
    add(new RoundedBoxGeometry(0.052, L1 + 0.03, 0.036, 3, 0.016), shell, 0, -L1 / 2, 0, thigh);
    add(new THREE.BoxGeometry(0.02, L1 * 0.6, 0.038), dark, -0.018, -L1 / 2, 0, thigh);
    const knee = new THREE.Group(); knee.position.y = -L1; thigh.add(knee);
    const km = add(new THREE.CylinderGeometry(0.028, 0.028, 0.04, 24), dark, 0, 0, 0, knee); km.rotation.x = Math.PI / 2;
    const calf = add(new THREE.CylinderGeometry(0.011, 0.018, L2, 16), dark, 0, -L2 / 2, 0, knee); void calf;
    add(new THREE.SphereGeometry(FOOT_R, 20, 14), rubber, 0, -L2, 0, knee);
    legs.push({ front, left, thigh, knee, phase: (front === left) ? 0 : 0.5 });
  }
  body.position.y = STAND;
  return { g, body, legs, gait: 0 };
}
function solveLeg(leg, fx, fy) {
  const D = Math.min(L1 + L2 - 1e-4, Math.hypot(fx, fy));
  const alpha = Math.atan2(fx, -fy);
  const beta = Math.acos(Math.min(1, (L1 * L1 + D * D - L2 * L2) / (2 * L1 * D)));
  const t1 = alpha - beta; // knee sits behind the hip-foot line
  const kx = L1 * Math.sin(t1), ky = -L1 * Math.cos(t1);
  const t2 = Math.atan2(fx - kx, -(fy - ky));
  leg.thigh.rotation.z = t1; leg.knee.rotation.z = t2 - t1;
}
// Trot: diagonal pairs share phase; stride follows body speed, yaw adds differential stride.
function poseGo2(R, v, w, dtSim) {
  const T = 0.42, amp = Math.min(1, Math.abs(v) / 0.12 + Math.abs(w) / 0.5);
  R.amp = (R.amp || 0) + (amp - (R.amp || 0)) * Math.min(1, dtSim * 12);
  if (R.amp > 0.02) R.gait = (R.gait + dtSim / T) % 1;
  const a = R.amp, h = STAND - FOOT_R;
  for (const leg of R.legs) {
    const p = (R.gait + leg.phase) % 1;
    const stride = (v + (leg.left ? -1 : 1) * w * 0.14) * T * 0.5;
    let x, y;
    if (p < 0.5) { const s = p / 0.5; x = (-0.5 + s) * stride; y = -h + 0.07 * Math.sin(Math.PI * s) * a; }
    else { const s = (p - 0.5) / 0.5; x = (0.5 - s) * stride; y = -h; }
    solveLeg(leg, x * a + 0.012, y);
  }
  R.body.position.y = STAND + 0.006 * Math.sin(R.gait * 4 * Math.PI) * a;
}

const robot = makeGo2(null); scene.add(robot.g);
const holo = new THREE.MeshBasicMaterial({ color: COL.pred, transparent: true, opacity: 0.26, depthWrite: false });
const ghost = makeGo2(holo);
ghost.g.traverse((o) => o.layers.set(OVERLAY)); scene.add(ghost.g); ghost.g.visible = false;
poseGo2(robot, 0, 0, 0); poseGo2(ghost, 0, 0, 0);
robot.g.scale.setScalar(RS); ghost.g.scale.setScalar(RS);

// ---------------------------------------------------------------- boxes
const BS = BOXSIZE;
const matsLight = (() => { const s = cardboardSide(), t = cardboardTop(); const side = new THREE.MeshStandardMaterial({ map: s, roughness: 0.86 }); const top = new THREE.MeshStandardMaterial({ map: t, roughness: 0.8 }); return [side, side, top, side, side, side]; })();
const matsHeavy = (() => { const s = crateSide(); const side = new THREE.MeshStandardMaterial({ map: s, roughness: 0.48, metalness: 0.65 }); const top = new THREE.MeshStandardMaterial({ color: 0x323b44, roughness: 0.5, metalness: 0.7 }); return [side, side, top, side, side, side]; })();
const boxGeo = new RoundedBoxGeometry(BS, BS, BS, 2, 0.012);
// RoundedBoxGeometry has no face groups; fall back to BoxGeometry for per-face materials.
const boxGeoFaces = new THREE.BoxGeometry(BS, BS, BS);
void boxGeo;
const holoBoxMat = new THREE.MeshBasicMaterial({ color: COL.pred, transparent: true, opacity: 0.08, depthWrite: false });
const holoEdgeMat = new THREE.LineBasicMaterial({ color: COL.pred, transparent: true, opacity: 0.9 });
let boxMeshes = [], boxGhosts = [];
function buildBoxes(boxes) {
  boxMeshes.forEach((m) => scene.remove(m)); boxGhosts.forEach((m) => scene.remove(m));
  boxMeshes = boxes.map((b) => { const m = new THREE.Mesh(boxGeoFaces, LIGHT_LOOK.has(b.appearance) ? matsLight : matsHeavy); m.castShadow = m.receiveShadow = true; scene.add(m); return m; });
  boxGhosts = boxes.map(() => {
    const g = new THREE.Group();
    g.add(new THREE.Mesh(boxGeoFaces, holoBoxMat));
    g.add(new THREE.LineSegments(new THREE.EdgesGeometry(boxGeoFaces), holoEdgeMat));
    g.traverse((o) => o.layers.set(OVERLAY)); g.visible = false; scene.add(g); return g;
  });
}

// ---------------------------------------------------------------- goal
const goalG = new THREE.Group(); scene.add(goalG);
const goalRing = new THREE.Mesh(new THREE.RingGeometry(Math.max(0.01, RUN.goal.r - 0.03), RUN.goal.r, 96), new THREE.MeshBasicMaterial({ color: new THREE.Color(COL.ok), transparent: true, depthWrite: false }));
goalRing.rotation.x = -Math.PI / 2; goalRing.position.y = 0.004; goalG.add(goalRing);
const goalDisk = new THREE.Mesh(new THREE.CircleGeometry(RUN.goal.r, 96), new THREE.MeshBasicMaterial({ color: COL.ok, transparent: true, opacity: 0.08, depthWrite: false }));
goalDisk.rotation.x = -Math.PI / 2; goalDisk.position.y = 0.003; goalG.add(goalDisk);
const beamTex = canvasTex(64, (g, n) => { const grd = g.createLinearGradient(0, 0, 0, n); grd.addColorStop(0, 'rgba(255,255,255,0)'); grd.addColorStop(1, 'rgba(255,255,255,1)'); g.fillStyle = grd; g.fillRect(0, 0, n, n); });
const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.44, 0.44, 0.9, 64, 1, true), new THREE.MeshBasicMaterial({ color: COL.ok, alphaMap: beamTex, transparent: true, opacity: 0.05, side: THREE.DoubleSide, depthWrite: false }));
void beam;
goalG.traverse((o) => o.layers.enable(0));

// ---------------------------------------------------------------- overlay lines
const lineMats = [];
function lmat(color, width, opacity, dashed) {
  const m = new LineMaterial({ color, linewidth: width, transparent: true, opacity, depthWrite: false, dashed: !!dashed, dashSize: 0.12, gapSize: 0.08 });
  lineMats.push(m); return m;
}
function line(points, mat, h = 0.015) {
  const geo = new LineGeometry(); geo.setPositions(points.flatMap(([x, y]) => { const v = W(x, y, h); return [v.x, v.y, v.z]; }));
  const l = new Line2(geo, mat); l.computeLineDistances(); l.layers.set(OVERLAY); l.frustumCulled = false; scene.add(l); return l;
}
let futureLines = [], selLine = null, selDots = [], trailLine = null, whiskers = [];
const trailMat = lmat(COL.trail, 2, 0.85);
function clearFutures() {
  futureLines.forEach((l) => { scene.remove(l); l.geometry.dispose(); l.material.dispose(); }); futureLines = [];
  if (selLine) { scene.remove(selLine); selLine.geometry.dispose(); selLine = null; }
  selDots.forEach((d) => scene.remove(d)); selDots = [];
}
const dotGeo = new THREE.RingGeometry(0.035, 0.055, 32);
const dotMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(COL.pred), transparent: true, depthWrite: false });
const selMat = lmat(new THREE.Color(COL.pred), 3.2, 1);
function buildFutures(plan) {
  clearFutures();
  const origin = P(plan.current_estimate.x, plan.current_estimate.y);
  const path = (r) => [origin, ...r.path.map(([x, y]) => P(x, y))];
  plan.rankings.forEach((r) => {
    const top = r.rank <= 8;
    const m = new LineMaterial({ color: COL.future, linewidth: top ? 1.6 : 1, transparent: true, opacity: 0.4, depthWrite: false });
    m.resolution.copy(res);
    const l = line(path(r), m, 0.012 + (r.rank % 8) * 0.0008); l.userData = { top, segs: r.path.length }; futureLines.push(l);
  });
  const sel = plan.rankings.find((r) => r.id === plan.selected_id) || plan.rankings[0];
  selLine = line(path(sel), selMat, 0.03);
  plan.selected_states.forEach((s, i) => {
    const d = new THREE.Mesh(dotGeo, dotMat); d.rotation.x = -Math.PI / 2; d.position.copy(W(...P(s.x, s.y), 0.02));
    d.scale.setScalar(i === 0 ? 1.8 : 1); d.layers.set(OVERLAY); scene.add(d); selDots.push(d);
  });
}
function setTrail() {
  if (trailLine) { scene.remove(trailLine); trailLine.geometry.dispose(); }
  trailLine = S.trail.length > 1 ? line(S.trail, trailMat, 0.008) : null;
}
const whiskerMatOk = lmat(new THREE.Color(COL.pred), 1.6, 0.9), whiskerMatBad = lmat(new THREE.Color(COL.alarm), 2.4, 1);
const ringGeo = new THREE.RingGeometry(0.05, 0.07, 32);
function addWhisker(pred, act, bad) {
  const l = line([P(pred.x, pred.y), P(act.x, act.y)], bad ? whiskerMatBad : whiskerMatOk, 0.02);
  const r = new THREE.Mesh(ringGeo, new THREE.MeshBasicMaterial({ color: bad ? COL.alarm : COL.pred, transparent: true, depthWrite: false }));
  r.rotation.x = -Math.PI / 2; r.position.copy(W(...P(pred.x, pred.y), 0.021)); r.layers.set(OVERLAY); scene.add(r);
  whiskers.push(l, r);
}
function clearWhiskers() { whiskers.forEach((o) => scene.remove(o)); whiskers = []; }
const alarmRing = new THREE.Mesh(new THREE.RingGeometry(0.46, 0.5, 96), new THREE.MeshBasicMaterial({ color: new THREE.Color(COL.alarm), transparent: true, depthWrite: false }));
alarmRing.rotation.x = -Math.PI / 2; alarmRing.layers.set(OVERLAY); alarmRing.visible = false; scene.add(alarmRing);

// ---------------------------------------------------------------- instrumentation overlay (not seen by the model camera)
function segs(pairs, color, opacity, h = 0.004) {
  const pos = []; pairs.forEach(([a, b]) => { const p = W(a[0], a[1], h), q = W(b[0], b[1], h); pos.push(p.x, p.y, p.z, q.x, q.y, q.z); });
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  const m = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false }));
  m.layers.set(OVERLAY); scene.add(m); return m;
}
{
  const minor = [], major = [];
  for (let k = 0; k <= A * 4 + 1e-9; k++) { const m = k / 4; (k % 4 === 0 ? major : minor).push([[m, 0], [m, A]], [[0, m], [A, m]]); }
  segs(minor, COL.ink, 0.12); segs(major, COL.ink, 0.28);
  const cross = [];
  for (let i = 1; i < A; i++) for (let j = 1; j < A; j++) cross.push([[i - 0.06, j], [i + 0.06, j]], [[i, j - 0.06], [i, j + 0.06]]);
  segs(cross, COL.ink, 0.45, 0.005);
  const d = -0.62, t = 0.08;
  segs([[[0, d], [A, d]], [[0, d - t], [0, d + t]], [[A, d - t], [A, d + t]], [[0, -0.36], [0, d - t]], [[A, -0.36], [A, d - t]],
        [[d, 0], [d, A]], [[d - t, 0], [d + t, 0]], [[d - t, A], [d + t, A]], [[-0.36, 0], [d - t, 0]], [[-0.36, A], [d - t, A]]], COL.ink, 0.7, 0.006);
  for (let m = 0; m <= A; m++) segs([[[m, d], [m, d + 0.05]], [[d, m], [d + 0.05, m]]], COL.ink, 0.7, 0.006);
  const ax = new THREE.Group();
  const shaftMat = new THREE.MeshBasicMaterial({ color: COL.ink });
  for (const [dir, rot] of [[[1, 0], -Math.PI / 2], [[0, 1], 0]]) {
    const len = 0.5, shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, len, 8), shaftMat);
    const head = new THREE.Mesh(new THREE.ConeGeometry(0.03, 0.08, 16), shaftMat);
    const g = new THREE.Group(); shaft.position.y = len / 2; head.position.y = len + 0.04; g.add(shaft, head);
    g.rotation.x = -Math.PI / 2; if (dir[0]) g.rotation.z = rot; ax.add(g);
  }
  ax.position.copy(W(OFF, OFF, 0.02)); ax.traverse((o) => o.layers.set(OVERLAY)); scene.add(ax);
  const gc = new THREE.Group();
  gc.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-0.6, 0.005, 0), new THREE.Vector3(-0.3, 0.005, 0), new THREE.Vector3(0.3, 0.005, 0), new THREE.Vector3(0.6, 0.005, 0), new THREE.Vector3(0, 0.005, -0.6), new THREE.Vector3(0, 0.005, -0.3), new THREE.Vector3(0, 0.005, 0.3), new THREE.Vector3(0, 0.005, 0.6)]), new THREE.LineBasicMaterial({ color: COL.ok })));
  gc.add(new THREE.Mesh(new THREE.RingGeometry(0.02, 0.035, 24).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ color: COL.ok })));
  gc.children.forEach((c) => (c.position.y = 0.005)); gc.traverse((o) => o.layers.set(OVERLAY)); goalG.add(gc);
}

// ---------------------------------------------------------------- latents and frames
function paintZ(id, z, scale, diff) {
  const c = $(id), n = z.length;
  if (c.width !== n) c.width = n;
  const g = c.getContext('2d'), img = g.createImageData(n, 1);
  const mix = (col, t) => [228 + (col[0] - 228) * t, 228 + (col[1] - 228) * t, 231 + (col[2] - 231) * t];
  z.forEach((raw, i) => {
    const v = raw == null ? 0 : raw / (scale || 1);
    let rgb;
    if (diff) rgb = mix([209, 47, 53], Math.min(1, Math.abs(v)));
    else if (v >= 0) rgb = mix([15, 123, 83], Math.min(1, v));
    else rgb = mix([24, 24, 27], Math.min(1, -v));
    img.data.set([...rgb, 255], i * 4);
  });
  g.putImageData(img, 0, 0);
}
function drawFrame(i, png) {
  const c = $('f' + i);
  if (!png) { c.getContext('2d').clearRect(0, 0, c.width, c.height); return; }
  const img = new Image();
  img.onload = () => { c.width = img.width; c.height = img.height; c.getContext('2d').drawImage(img, 0, 0); };
  img.src = 'data:image/png;base64,' + png;
}
function showFrames() { S.frames.slice(-3).forEach((png, i) => drawFrame(i, png)); }

// ---------------------------------------------------------------- playback state
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
const DUR = reduce ? { imagine: 80, score: 80, lock: 80, execute: 400, compare: 80 } : { imagine: 900, score: 650, lock: 380, execute: 1100, compare: 520 };
const ORDER = ['imagine', 'score', 'lock', 'execute', 'compare'];
const BLOCKS = [];
let END = null, lastSeq = -1, S;
const DT = RUN.block_duration_s, TH = RUN.surprise.threshold;
function ingest(records) {
  for (const r of records) {
    if (r.seq <= lastSeq) continue;
    lastSeq = r.seq;
    if (r.type === 'plan_locked') BLOCKS[r.block] = { plan: r, exec: null };
    else if (r.type === 'block_executed' && BLOCKS[r.block]) BLOCKS[r.block].exec = r;
    else if (r.type === 'run_end') END = r;
  }
}
ingest(RECORDS);

const gt0 = RUN.ground_truth;
buildBoxes(gt0.objects);
const goal = RUN.goal;
goalG.position.copy(W(...P(goal.x, goal.y)));

function reset() {
  S = {
    k: 0, phase: 'score', phaseT: 1, status: 'paused', alarm: false, stepOnce: false,
    pose: { robot: { ...gt0.robot }, objects: gt0.objects.map((o) => ({ ...o })) },
    trail: [P(gt0.robot.x, gt0.robot.y)], surprise: [], frames: RUN.history.map((h) => h.png),
    hist: RUN.history_actions.slice(-2), t: RUN.history.length ? RUN.history[RUN.history.length - 1].sim_time_s : 0,
  };
  clearWhiskers(); clearFutures(); setTrail(); alarmRing.visible = false; ghost.g.visible = false; boxGhosts.forEach((g) => (g.visible = false));
  robot.amp = 0; poseGo2(robot, 0, 0, 0);
  showFrames();
  ['zPred', 'zObs', 'zDiff'].forEach((id) => paintZ(id, new Array(RUN.bundle.latent_dim).fill(0), 1, id === 'zDiff'));
  $('zRmse').textContent = 'rmse –';
  setBanner(null);
  if (BLOCKS[0]) buildFutures(BLOCKS[0].plan);
  renderPanels(); updateButtons();
}
const cur = () => BLOCKS[S.k];

function startBlock() {
  const b = cur();
  if (!b) {
    if (END && S.k >= END.blocks) finishRun();
    else { S.status = 'waiting'; updateButtons(); }
    return;
  }
  buildFutures(b.plan); S.phase = 'imagine'; S.phaseT = 0; renderPanels();
}
function enter(ph) {
  const b = cur();
  if (ph === 'execute' && !b.exec) { S.phaseT = 1; S.status = 'waiting'; updateButtons(); return; }
  S.phase = ph; S.phaseT = 0;
  if (ph === 'lock') {
    const pr = b.plan.selected_states[0];
    ghost.g.position.copy(W(...P(pr.x, pr.y))); ghost.g.rotation.y = pr.yaw; ghost.g.visible = true;
    const a = b.plan.first_action; ghost.gait = robot.gait; ghost.amp = robot.amp; poseGo2(ghost, a.forward_mps, a.yaw_rate_rps, 0.25);
    S.pose.objects.forEach((o, i) => {
      const p = pr.objects.find((q) => q.id === o.id);
      const moved = p && Math.hypot(p.x - o.x, p.y - o.y) > 0.01;
      boxGhosts[i].visible = !!moved;
      if (p) boxGhosts[i].position.copy(W(...P(p.x, p.y), BS / 2));
    });
  }
  if (ph === 'compare') finishBlock();
  renderPanels();
}
function finishBlock() {
  const { plan, exec } = cur();
  const end = exec.ground_truth.end, pred = plan.selected_states[0];
  S.pose = { robot: { ...end.robot }, objects: end.objects.map((o) => ({ ...o })) };
  S.trail.push(P(end.robot.x, end.robot.y)); setTrail();
  const d = exec.surprise.discrepancy;
  const bad = exec.surprise.stop_commanded || (d != null && d > TH);
  addWhisker(pred, end.robot, bad);
  S.surprise.push(d ?? 0);
  S.frames.push(exec.observation.png); showFrames();
  S.hist = [S.hist[S.hist.length - 1], exec.applied_action];
  S.t = exec.sim_end_s;
  const zp = plan.predicted_next_latent || [], zo = exec.observed_latent || [];
  const scale = Math.max(1e-9, ...zp.map(Math.abs), ...zo.map(Math.abs));
  const zd = zp.map((v, i) => Math.abs((v ?? 0) - (zo[i] ?? 0)));
  paintZ('zPred', zp, scale); paintZ('zObs', zo, scale); paintZ('zDiff', zd, Math.max(1e-9, ...zd), true);
  $('zRmse').textContent = `rmse ${Math.sqrt(zd.reduce((a, v) => a + v * v, 0) / Math.max(1, zd.length)).toPrecision(3)}`;
  if (exec.surprise.stop_commanded) S.alarm = true;
}
function finishRun() {
  S.status = 'ended';
  const r = END.reason;
  const map = {
    goal_reached: ['goal', 'Goal reached', `${END.blocks} blocks · ${END.sim_time_s.toFixed(1)} s simulated`],
    surprise_stop: ['alarm', 'Stop latched', `latent surprise exceeded the calibrated threshold ${TH.toPrecision(3)}`],
    stalled: ['stall', 'Planner stalled', 'zero first block chosen repeatedly near the goal (D-028)'],
    fall: ['alarm', 'Fall detected', `after ${END.blocks} blocks`],
    out_of_bounds: ['alarm', 'Out of bounds', `after ${END.blocks} blocks`],
    max_blocks: ['stall', 'Block budget spent', `${END.blocks} blocks`],
  };
  const [kind, title, sub] = map[r] || ['stall', r, ''];
  setBanner(kind, title, sub);
  if (S.alarm) alarmRing.visible = true;
  updateButtons();
}
function setBanner(kind, title, sub) {
  const b = $('banner'); b.hidden = !kind; b.className = 'banner ' + (kind || '');
  b.innerHTML = kind ? `<strong>${title}</strong><span>${sub}</span>` : '';
  $('vp').classList.toggle('alarm', kind === 'alarm');
}

// ---------------------------------------------------------------- HUD
const f2 = (v) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(2);
const fmt = (v) => (v == null ? '–' : Math.abs(v) >= 0.01 ? v.toFixed(3) : v.toExponential(2));
$('runId').textContent = RUN.run_id;
$('bundleId').textContent = RUN.bundle.bundle_id;
$('levelTag').textContent = RUN.fallback_level;
$('scope').textContent = RUN.evidence_scope;
$('modelLabel').textContent = RUN.bundle.model_label;
$('arenaTag').textContent = `${A.toFixed(3)} × ${RUN.scene.arena_height_m.toFixed(3)} m`;
$('camTag').textContent = `${RUN.camera.width_px} × ${RUN.camera.height_px} rgb`;
$('latTag').textContent = `${RUN.bundle.latent_dim}-d`;
function renderPanels() {
  const b = cur() || BLOCKS[BLOCKS.length - 1];
  if (b) {
    const P0 = b.plan, a = P0.first_action;
    $('pId').textContent = P0.selected_id;
    $('pAct').textContent = `v ${a.forward_mps.toFixed(2)} m/s · ω ${f2(a.yaw_rate_rps)} rad/s`;
    $('pReason').textContent = P0.selection_reason;
    $('pLock').className = 'lock';
    $('pLock').textContent = P0.locked_at_utc.slice(11, 23) + ' UTC';
    $('pLat').textContent = `${(P0.planning_latency_s * 1000).toFixed(1)} ms · ${P0.rankings.length} × ${RUN.bundle.horizon_blocks} predictions`;
    const rows = P0.rankings.slice(0, 6), max = Math.max(...rows.map((r) => r.total));
    $('cands').innerHTML = rows.map((r) => `<div class="cand${r.id === P0.selected_id ? ' sel' : ''}"><span class="rk">${String(r.rank).padStart(2, '0')}</span><div class="nm"><span title="${r.id}">${r.id}</span><div class="bar">${PARTS.map(([k, , c]) => `<span style="width:${(100 * Math.max(0, r.parts[k])) / max}%;background:${c}" title="${k} ${r.parts[k]}"></span>`).join('')}</div></div><span class="tot">${r.total.toFixed(3)}</span></div>`).join('');
  }
  $('legend').innerHTML = PARTS.map(([, n, c]) => `<span><i style="background:${c}"></i>${n}</span>`).join('');
  const h = S.hist;
  $('a0').innerHTML = `a₋₂ <b>${h[0] ? h[0].forward_mps.toFixed(2) + ' · ' + f2(h[0].yaw_rate_rps) : '–'}</b>`;
  $('a1').innerHTML = `a₋₁ <b>${h[1] ? h[1].forward_mps.toFixed(2) + ' · ' + f2(h[1].yaw_rate_rps) : '–'}</b>`;
  const d = S.surprise[S.surprise.length - 1];
  $('sNow').innerHTML = `${d === undefined ? '–' : fmt(d)}<small> / ${fmt(TH)} stop threshold · ${RUN.surprise.metric}</small>`;
  const pill = $('sPill');
  if (S.alarm) { pill.className = 'state alarm'; pill.textContent = 'alarm_latched'; }
  else if (d === undefined) { pill.className = 'state'; pill.textContent = 'unarmed'; }
  else { pill.className = 'state ok'; pill.textContent = 'normal'; }
  spark();
}
function spark() {
  const svg = $('spark'), w = svg.getBoundingClientRect().width || 300, h = 74, l = 40, r = 6, tp = 6, b = 14;
  const vals = S.surprise, n = Math.max(20, vals.length), ymax = Math.max(TH * 2, ...vals.map((v) => Math.min(v, TH * 6)));
  const x = (i) => l + (i / (n - 1)) * (w - l - r), y = (v) => tp + (1 - Math.min(v, ymax) / ymax) * (h - tp - b);
  let s = `<g font-family="Geist, ui-sans-serif, system-ui, sans-serif" font-size="9.5" fill="#8A8A93">`;
  for (const v of [0, TH, ymax]) s += `<text x="${l - 4}" y="${y(v) + 3}" text-anchor="end">${v === 0 ? '0' : v.toPrecision(2)}</text>`;
  s += `<text x="${l}" y="${h - 2}">block 1</text><text x="${w - r}" y="${h - 2}" text-anchor="end">${n}</text></g>`;
  s += `<line x1="${l}" x2="${w - r}" y1="${y(0)}" y2="${y(0)}" stroke="rgba(24,24,27,.12)"/>`;
  s += `<line x1="${l}" x2="${w - r}" y1="${y(TH)}" y2="${y(TH)}" stroke="#D12F35" stroke-opacity=".7" stroke-dasharray="3 3"/>`;
  if (vals.length) {
    const pts = vals.map((v, i) => `${x(i)},${y(v)}`).join(' ');
    s += `<polyline points="${pts}" fill="none" stroke="#18181B" stroke-width="1.4" stroke-linejoin="round"/>`;
    vals.forEach((v, i) => { const bad = v > TH; if (bad || i === vals.length - 1) s += `<circle cx="${x(i)}" cy="${y(v)}" r="${bad ? 3.8 : 2.8}" fill="${bad ? '#D12F35' : '#18181B'}"/>`; });
  }
  svg.setAttribute('viewBox', `0 0 ${w} ${h}`); svg.innerHTML = s;
}
function updateButtons() {
  const done = S.status === 'ended';
  $('runBtn').textContent = S.status === 'running' && !S.stepOnce ? 'Pause' : 'Play';
  $('runBtn').disabled = done; $('stepBtn').disabled = done || S.status === 'running';
  $('liveTag').textContent = END ? `recorded · complete · ${END.blocks} blocks` : `recorded · ${BLOCKS.length} blocks`;
}

// ---------------------------------------------------------------- labels
const labelsEl = $('labels');
const labels = {};
function label(key, cls) { if (!labels[key]) { const d = document.createElement('div'); d.className = 'lbl ' + (cls || ''); labelsEl.appendChild(d); labels[key] = d; } return labels[key]; }
const tmpV = new THREE.Vector3();
function placeLabel(el, pos, text, show = true) {
  if (!show) { el.hidden = true; return; }
  tmpV.copy(pos).project(camera);
  const r = canvas.getBoundingClientRect();
  el.hidden = tmpV.z > 1 || Math.abs(tmpV.x) > 1.05 || Math.abs(tmpV.y) > 1.05;
  el.style.left = ((tmpV.x + 1) / 2) * r.width + 'px'; el.style.top = ((1 - tmpV.y) / 2) * r.height + 'px';
  const key = Array.isArray(text) ? text.join('|') : text;
  if (el.dataset.k !== key) {
    el.dataset.k = key;
    if (Array.isArray(text)) el.innerHTML = `<div class="ct"><b>${text[0]}</b>${text.slice(1).map((t) => `<span>${t}</span>`).join('')}</div>`;
    else el.textContent = text;
  }
}

// ---------------------------------------------------------------- camera modes
let camMode = 'orbit';
const chasePos = new THREE.Vector3(), chaseLook = new THREE.Vector3();
const CAM0 = camera.position.clone(), TGT0 = controls.target.clone();
function applyCam(dt) {
  if (camMode === 'chase') {
    const r = robot.g.position, yaw = robot.g.rotation.y, k0 = Math.max(0.5, RS);
    const want = new THREE.Vector3(r.x - Math.cos(yaw) * 3.3 * k0 + Math.sin(yaw) * 1.4 * k0, 1.75 * k0, r.z + Math.sin(yaw) * 3.3 * k0 + Math.cos(yaw) * 1.4 * k0);
    const look = new THREE.Vector3(r.x + Math.cos(yaw) * 1.6 * k0, 0.25 * k0, r.z - Math.sin(yaw) * 1.6 * k0);
    const k = 1 - Math.exp(-dt * 3);
    chasePos.lerp(want, k); chaseLook.lerp(look, k);
    camera.position.copy(chasePos); camera.lookAt(chaseLook);
  } else if (camMode === 'overhead') {
    const k = 1 - Math.exp(-dt * 4);
    camera.position.lerp(new THREE.Vector3(0, A * 1.6, 0.01), k); camera.lookAt(0, 0, 0);
  } else controls.update();
}
document.querySelectorAll('.camsel button').forEach((b) => (b.onclick = () => {
  camMode = b.dataset.cam; document.querySelectorAll('.camsel button').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
  controls.enabled = camMode === 'orbit';
  if (camMode === 'chase') { chasePos.copy(camera.position); chaseLook.copy(controls.target); }
  if (camMode === 'orbit') { camera.position.copy(CAM0); controls.target.copy(TGT0); }
}));

// ---------------------------------------------------------------- frame loop
const res = new THREE.Vector2(1, 1);
function resize() {
  const r = canvas.getBoundingClientRect(), w = Math.max(1, r.width), h = Math.max(1, r.height);
  renderer.setSize(w, h, false); composer.setSize(w, h); bloom.setSize(w, h);
  camera.aspect = w / h; camera.fov = w / h < 1 ? 52 : 38; camera.updateProjectionMatrix();
  res.set(w * renderer.getPixelRatio(), h * renderer.getPixelRatio());
  lineMats.forEach((m) => m.resolution.copy(res)); futureLines.forEach((l) => l.material.resolution.copy(res));
  if (S) spark();
}
new ResizeObserver(resize).observe(canvas);

const lerpAngle = (a, b, t) => a + Math.atan2(Math.sin(b - a), Math.cos(b - a)) * t;
let last = performance.now(), clock = 0;
function frame(now) {
  const rdt = Math.min(0.1, (now - last) / 1000); last = now; clock += rdt;
  const speed = parseFloat($('speed').value);
  if (S.status === 'running') {
    S.phaseT += (rdt * 1000 * speed) / DUR[S.phase];
    if (S.phaseT >= 1) {
      const i = ORDER.indexOf(S.phase);
      if (i < ORDER.length - 1) enter(ORDER[i + 1]);
      else if (S.status === 'running') {
        S.k += 1;
        if (S.stepOnce) { S.status = 'paused'; S.stepOnce = false; S.phaseT = 1; if (!cur() && END && S.k >= END.blocks) finishRun(); else if (cur()) { buildFutures(cur().plan); S.phase = 'score'; } updateButtons(); renderPanels(); }
        else startBlock();
      }
    }
  } else if (S.status === 'waiting') {
    const b = cur();
    if (b && S.phase === 'lock' && b.exec) { S.status = 'running'; enter('execute'); updateButtons(); }
    else if (b && S.phase !== 'lock') { S.status = 'running'; startBlock(); updateButtons(); }
    else if (!b && END && S.k >= END.blocks) finishRun();
  }
  const ph = S.phase, t = Math.min(1, S.phaseT), b = cur();

  const showF = ['imagine', 'score', 'lock'].includes(ph) && !S.alarm && S.status !== 'ended' && !!b;
  futureLines.forEach((l, idx) => {
    l.visible = showF;
    if (!showF) return;
    const segs = l.userData.segs;
    if (ph === 'imagine' && S.status === 'running') { const k = Math.max(0, Math.min(1, t * 1.5 - (idx % 16) / 40)); l.geometry.instanceCount = Math.max(1, Math.ceil(segs * k)); l.material.opacity = 0.32; }
    else { l.geometry.instanceCount = segs; l.material.opacity = (l.userData.top ? 0.5 : 0.09) * (ph === 'lock' ? 1 - 0.6 * t : 1); }
  });
  const showSel = (ph === 'score' || ph === 'lock' || ph === 'execute') && !S.alarm && S.status !== 'ended' && !!b;
  if (selLine) { selLine.visible = showSel; selMat.opacity = ph === 'score' ? (S.status === 'running' ? Math.min(1, t * 2) : 1) : ph === 'execute' ? 0.45 : 0.95; }
  selDots.forEach((d) => (d.visible = showSel));
  if (ph === 'imagine' || ph === 'score' || S.status === 'ended') { ghost.g.visible = false; boxGhosts.forEach((g) => (g.visible = false)); }
  holo.opacity = 0.22 + 0.05 * Math.sin(clock * 5);

  let rob = S.pose.robot, objs = S.pose.objects, v = 0, w = 0, dtSim = 0;
  if (ph === 'execute' && b && b.exec && S.status !== 'ended') {
    const s0 = b.exec.ground_truth.start, s1 = b.exec.ground_truth.end;
    rob = { x: s0.robot.x + (s1.robot.x - s0.robot.x) * t, y: s0.robot.y + (s1.robot.y - s0.robot.y) * t, yaw: lerpAngle(s0.robot.yaw, s1.robot.yaw, t) };
    objs = s0.objects.map((o, j) => ({ ...o, x: o.x + (s1.objects[j].x - o.x) * t, y: o.y + (s1.objects[j].y - o.y) * t }));
    v = b.exec.applied_action.forward_mps; w = b.exec.applied_action.yaw_rate_rps;
    dtSim = S.status === 'running' ? (rdt * 1000 * speed / DUR.execute) * DT : 0;
  }
  robot.g.position.copy(W(...P(rob.x, rob.y))); robot.g.rotation.y = rob.yaw;
  objs.forEach((o, i) => boxMeshes[i] && boxMeshes[i].position.copy(W(...P(o.x, o.y), BS / 2)));
  if (dtSim > 0 || S.status !== 'running' || ph !== 'execute') poseGo2(robot, v, w, dtSim);
  if (S.alarm) { alarmRing.visible = true; alarmRing.position.copy(W(...P(rob.x, rob.y), 0.02)); const p = 0.5 + 0.5 * Math.sin(clock * 9); alarmRing.material.opacity = 0.4 + 0.6 * p; alarmLight.position.copy(W(...P(rob.x, rob.y), 0.8)); alarmLight.intensity = 3 + 5 * p; }
  else alarmLight.intensity = 0;
  led.material.color.setHex(Math.sin(clock * 5) > 0 ? 0xff2a2a : 0x401010);

  $('tT').textContent = (S.t + (ph === 'execute' && S.status === 'running' ? t * DT : 0)).toFixed(1) + ' s';
  $('tB').textContent = String(S.k).padStart(2, '0');
  $('tP').textContent = S.status === 'running' ? ph : S.status;
  $('tPhys').innerHTML = `physics <b>${S.status === 'running' && ph === 'execute' ? 'stepping' : 'paused'}</b>`;
  document.querySelectorAll('#pipe div').forEach((d) => {
    const active = S.status === 'running' || S.status === 'waiting';
    const on = active && d.dataset.ph === ph, done = active && ORDER.indexOf(d.dataset.ph) < ORDER.indexOf(ph);
    d.classList.toggle('on', on); d.classList.toggle('done', done);
    d.querySelector('.fill').style.width = on ? `${t * 100}%` : done ? '100%' : '0';
  });
  placeLabel(label('goal', 'callout accent'), W(...P(goal.x, goal.y), 0.01), ['Goal', `x ${goal.x.toFixed(2)} · y ${goal.y.toFixed(2)} · r ${goal.r.toFixed(2)} m`]);
  placeLabel(label('dimx', 'plain dim'), W(A / 2, -0.62, 0.02), `${A.toFixed(3)} m`);
  placeLabel(label('dimy', 'plain dim'), W(-0.62, A / 2, 0.02), `${RUN.scene.arena_height_m.toFixed(3)} m`);
  placeLabel(label('axx', 'plain'), W(OFF + 0.66, OFF, 0.02), 'x');
  placeLabel(label('axy', 'plain'), W(OFF, OFF + 0.66, 0.02), 'y');
  placeLabel(label('origin', 'plain'), W(OFF - 0.12, OFF - 0.14, 0.02), '(0, 0)');
  for (let m = 0; m <= A; m++) { placeLabel(label('tx' + m, 'plain'), W(m, -0.44, 0.02), `${m - OFF}`); placeLabel(label('ty' + m, 'plain'), W(-0.44, m, 0.02), `${m - OFF}`); }
  const wp = ['score', 'lock'].includes(ph) && !S.alarm && b && S.status !== 'ended';
  for (let k = 0; k < RUN.bundle.horizon_blocks; k++) {
    const st = b && b.plan.selected_states[k];
    placeLabel(label('wp' + k, 'wp'), st ? W(...P(st.x, st.y), 0.05) : new THREE.Vector3(), `+${((k + 1) * DT).toFixed(1)}s`, !!wp && !!st && (k === 0 || k % 2 === 1));
  }
  placeLabel(label('robot', 'callout tall flip'), W(...P(rob.x, rob.y), 0.44 * RS), ['Go2 · ground truth', `x ${rob.x.toFixed(3)} m · y ${rob.y.toFixed(3)} m`, `θ ${f2(rob.yaw)} rad · v ${v.toFixed(2)} m/s`]);
  objs.forEach((o, i) => placeLabel(label('box' + i, 'callout low'), W(...P(o.x, o.y), BS + 0.01), [o.id, `appearance ${o.appearance}`]));
  placeLabel(label('ghost', 'callout accent low'), ghost.g.position.clone().setY(0.44 * RS), ['Locked prediction', `t + ${DT} s · readout`], ghost.g.visible && ph !== 'compare' && !S.alarm);

  applyCam(rdt);
  composer.render();
  requestAnimationFrame(frame);
}

// ---------------------------------------------------------------- controls and live follow
function go(once) {
  if (S.status === 'ended') return;
  const fresh = S.phase === 'score' && S.phaseT >= 1 && S.status === 'paused';
  S.status = 'running'; S.stepOnce = once;
  if (fresh) startBlock();
  updateButtons();
}
$('runBtn').onclick = () => { if (S.status === 'running' && !S.stepOnce) { S.status = 'paused'; updateButtons(); } else go(false); };
$('stepBtn').onclick = () => go(true);
$('resetBtn').onclick = reset;
const runSel = $('runSel');
runSel.innerHTML = (RUNS.length ? RUNS : [LOG.replace(/^\//, '')]).map((p) => `<option value="/${p}"${'/' + p === LOG ? ' selected' : ''}>${p.replace(/^runs\//, '').replace(/\/ui\.jsonl$/, '')}</option>`).join('');
runSel.onchange = () => { location.search = '?log=' + encodeURIComponent(runSel.value); };
window.addEventListener('keydown', (e) => { if (e.code === 'Space' && e.target === document.body) { e.preventDefault(); $('runBtn').click(); } });

resize();
reset();
$('loading').remove();
if (params.get('autoplay') !== '0') go(false);
requestAnimationFrame(frame);
