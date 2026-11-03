/* Motor de procesamiento: corre en un Web Worker, 100% en el navegador.
 * Port de mainv10.py (versión de escritorio v10): mismos modelos, mismos
 * umbrales y mismos pasos, en el mismo orden:
 *   1) Segmentación con u2net_human_seg (pesos en fp16, cálculo en fp32)
 *   2) Rostros con YuNet 2023mar; se elige el que está sobre la persona
 *   3) Máscara: silueta principal, huecos pequeños y puntas de cabello en fondo liso
 *   4) Varón / mujer por el largo del cabello junto a la mandíbula
 *   5) Coronilla por máscara (con tope anatómico) y encuadre por altura de cabeza
 *   6) Composición sobre blanco quitando el color del fondo viejo en los bordes
 *   7) Copia enderezada si la cabeza está inclinada
 * Cada función lleva en su comentario el nombre de su equivalente en mainv10.py.
 * Los redimensionados replican cv2.resize y la rotación replica PIL, para que
 * el resultado coincida con el del escritorio.
 */
importScripts("../vendor/ort.wasm.min.js");

const CACHE_NAME = "fotocheck-modelos-v10";   // cambiar si se cambian los modelos
const MODELS = {
  seg: { url: "../models/u2net_human_seg_f16w.onnx", size: 88044409 },
  yunet: { url: "../models/face_detection_yunet_2023mar.onnx", size: 229740 },
};

// ── Configuración (mismos valores que mainv10.py) ───────────────────────────
const FACE_HEIGHT_RATIO_YUNET = 0.30;
const FACE_CENTER_Y_RATIO = 0.40;
const HEAD_TOP_MARGIN = 0.06;
const ENCUADRES = {
  "varón": { space: 0.025, head: 0.72 },
  "mujer": { space: 0.025, head: 0.60 },
};
const LONG_HAIR_MIN_COVERAGE = 0.5;
const CROWN_MIN_ABOVE_EYES = 0.85;
const REFINAR_CABELLO = true;
const HAIR_BG_MAX_STD = 25, HAIR_BG_MIN_UNIFORM = 0.90;
const HAIR_T0 = 25, HAIR_T1 = 70, HAIR_BAND = 0.12, HAIR_MIN_ALPHA = 0.4;
const CHIN_MARGIN = 0.25, SIDE_MARGIN = 0.03;
const MASK_THRESHOLD = 0.5, HOLE_FILL_MAX_RATIO = 0.02;
const DECONTAMINAR_BORDES = true;
const YUNET_MAX_SIDE = 1280, YUNET_SCORES = [0.6, 0.4], YUNET_NMS = 0.3, YUNET_TOP_K = 5000;
const OTRA_CARA_RATIO = 0.30;
const MIN_FACE_PX = 60, MAX_UPSCALE = 1.6;
const TILT_MIN_DEG = 2.0, TILT_MAX_DEG = 15.0, TILT_EYES_MOUTH_MAX_DIFF = 6.0;
const IMAGENET_MEAN = [0.485, 0.456, 0.406], IMAGENET_STD = [0.229, 0.224, 0.225];
const JPEG_QUALITY = 0.95;

// Solo en la web: las fotos más grandes se reducen antes de procesar para no
// agotar la memoria del navegador. Todo el algoritmo escala con el tamaño.
const MAX_LADO = 2400;
const MIN_LADO_SALIDA = 64, MAX_LADO_SALIDA = 4000;

let sessions = null;
let last = null;   // última foto analizada: permite cambiar el tamaño sin reprocesar

ort.env.wasm.wasmPaths = "../vendor/";
ort.env.logLevel = "error";
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

const post = (type, data = {}, transfer = []) => self.postMessage({ type, ...data }, transfer);

// ═══════════════════════════════════════════════════════════════════════════
// Utilidades numéricas (mismo redondeo que Python / OpenCV)
// ═══════════════════════════════════════════════════════════════════════════

/** round() de Python y cvRound: las mitades van al par. */
function pyRound(v) {
  const f = Math.floor(v), d = v - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}
const trunc = Math.trunc;                     // int() de Python
const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const pct = (v) => `${(v * 100).toFixed(0)}%`;  // f"{v:.0%}"

/** Imagen: w × h × c canales intercalados (Uint8Array o Float32Array). */
const newImg = (w, h, c, Type) => ({ w, h, c, data: new Type(w * h * c) });

// ═══════════════════════════════════════════════════════════════════════════
// cv2.resize (INTER_AREA, INTER_LINEAR, INTER_LANCZOS4)
// ═══════════════════════════════════════════════════════════════════════════

function resize(src, dw, dh, interp) {
  const { w: sw, h: sh } = src;
  const invX = dw / sw, invY = dh / sh;
  const scX = 1 / invX, scY = 1 / invY;
  const iscX = pyRound(scX), iscY = pyRound(scY);
  const areaFast = Math.abs(scX - iscX) < Number.EPSILON && Math.abs(scY - iscY) < Number.EPSILON;
  if (interp === "linear" && areaFast && iscX === 2 && iscY === 2) interp = "area";
  if (interp === "area" && scX >= 1 && scY >= 1) {
    return areaFast ? resizeAreaFast(src, dw, dh, iscX, iscY) : resizeArea(src, dw, dh, scX, scY);
  }
  if (interp === "lanczos4") {
    return resizeSep(src, dw, dh, lanczosTab(sw, dw, scX), lanczosTab(sh, dh, scY), false);
  }
  const area = interp === "area";   // INTER_AREA ampliando: OpenCV lo emula con bilineal
  return resizeSep(src, dw, dh, linearTab(sw, dw, scX, invX, area), linearTab(sh, dh, scY, invY, area), true);
}

/** Índices y pesos de INTER_LINEAR para un eje (resize.cpp). */
function linearTab(sLen, dLen, scale, inv, areaMode) {
  const idx = new Int32Array(dLen * 2), coef = new Float64Array(dLen * 2), icoef = new Int32Array(dLen * 2);
  for (let d = 0; d < dLen; d++) {
    let s, f;
    if (!areaMode) {
      f = Math.fround((d + 0.5) * scale - 0.5); s = Math.floor(f); f = Math.fround(f - s);
    } else {
      s = Math.floor(d * scale);
      f = Math.fround((d + 1) - (s + 1) * inv);
      f = f <= 0 ? 0 : Math.fround(f - Math.floor(f));
    }
    if (s < 0) { f = 0; s = 0; }
    if (s >= sLen - 1) { f = 0; s = sLen - 1; }
    const c0 = Math.fround(1 - f);
    idx[2 * d] = s; idx[2 * d + 1] = Math.min(s + 1, sLen - 1);
    coef[2 * d] = c0; coef[2 * d + 1] = f;
    icoef[2 * d] = pyRound(c0 * 2048); icoef[2 * d + 1] = pyRound(f * 2048);
  }
  return { k: 2, idx, coef, icoef };
}

/** interpolateLanczos4 de OpenCV. */
const S45 = 0.70710678118654752440084436210485;
const LANCZOS_CS = [[1, 0], [-S45, -S45], [0, 1], [S45, -S45], [-1, 0], [S45, S45], [0, -1], [-S45, S45]];
function lanczos4(x, out) {
  if (x < 1.1920929e-7) { out.fill(0); out[3] = 1; return; }
  let sum = 0;
  const y0 = -(x + 3) * Math.PI * 0.25, s0 = Math.sin(y0), c0 = Math.cos(y0);
  for (let i = 0; i < 8; i++) {
    const y = -(x + 3 - i) * Math.PI * 0.25;
    out[i] = Math.fround((LANCZOS_CS[i][0] * s0 + LANCZOS_CS[i][1] * c0) / (y * y));
    sum = Math.fround(sum + out[i]);
  }
  sum = Math.fround(1 / sum);
  for (let i = 0; i < 8; i++) out[i] = Math.fround(out[i] * sum);
}

function lanczosTab(sLen, dLen, scale) {
  const idx = new Int32Array(dLen * 8), coef = new Float64Array(dLen * 8), icoef = new Int32Array(dLen * 8);
  const c = new Float64Array(8);
  for (let d = 0; d < dLen; d++) {
    let f = Math.fround((d + 0.5) * scale - 0.5);
    const s = Math.floor(f); f = Math.fround(f - s);
    lanczos4(f, c);
    for (let k = 0; k < 8; k++) {
      idx[8 * d + k] = clamp(s + k - 3, 0, sLen - 1);
      coef[8 * d + k] = c[k];
      icoef[8 * d + k] = pyRound(c[k] * 2048);
    }
  }
  return { k: 8, idx, coef, icoef };
}

/** Redimensionado separable (bilineal o Lanczos4), con la aritmética de OpenCV. */
function resizeSep(src, dw, dh, tx, ty, linear) {
  const { w: sw, h: sh, c, data } = src;
  const u8 = data instanceof Uint8Array;
  const out = newImg(dw, dh, c, u8 ? Uint8Array : Float32Array);
  const o = out.data;
  // Pasada horizontal de todas las filas de origen
  const hb = new Float64Array(sh * dw * c);
  const kx = tx.k, cx = u8 ? tx.icoef : tx.coef;
  for (let y = 0; y < sh; y++) {
    const row = y * sw * c, hrow = y * dw * c;
    for (let x = 0; x < dw; x++) {
      for (let ch = 0; ch < c; ch++) {
        let v = 0;
        for (let k = 0; k < kx; k++) v += data[row + tx.idx[x * kx + k] * c + ch] * cx[x * kx + k];
        hb[hrow + x * c + ch] = u8 ? v : Math.fround(v);
      }
    }
  }
  // Pasada vertical
  const ky = ty.k, cy = u8 ? ty.icoef : ty.coef;
  const rowLen = dw * c;
  for (let y = 0; y < dh; y++) {
    for (let i = 0; i < rowLen; i++) {
      let v;
      if (u8 && linear) {
        // VResizeLinearVec_32s8u: ((b0·(S0>>4))>>16 + (b1·(S1>>4))>>16 + 2) >> 2
        const s0 = hb[ty.idx[2 * y] * rowLen + i], s1 = hb[ty.idx[2 * y + 1] * rowLen + i];
        v = (Math.floor(cy[2 * y] * (s0 >> 4) / 65536) + Math.floor(cy[2 * y + 1] * (s1 >> 4) / 65536) + 2) >> 2;
      } else if (u8) {
        let s = 0;
        for (let k = 0; k < ky; k++) s += cy[y * ky + k] * hb[ty.idx[y * ky + k] * rowLen + i];
        v = Math.floor((s + 2097152) / 4194304);   // FixedPtCast<int, uchar, 22>
      } else {
        v = 0;
        for (let k = 0; k < ky; k++) v += cy[y * ky + k] * hb[ty.idx[y * ky + k] * rowLen + i];
      }
      o[y * rowLen + i] = u8 ? clamp(v, 0, 255) : v;
    }
  }
  return out;
}

/** Tabla de pesos de INTER_AREA para un eje (computeResizeAreaTab). */
function areaTab(sLen, dLen, scale) {
  const di = [], si = [], al = [];
  for (let d = 0; d < dLen; d++) {
    const f1 = d * scale, f2 = f1 + scale;
    const cell = Math.min(scale, sLen - f1);
    let s1 = Math.ceil(f1), s2 = Math.floor(f2);
    s2 = Math.min(s2, sLen - 1); s1 = Math.min(s1, s2);
    if (s1 - f1 > 1e-3) { di.push(d); si.push(s1 - 1); al.push(Math.fround((s1 - f1) / cell)); }
    for (let s = s1; s < s2; s++) { di.push(d); si.push(s); al.push(Math.fround(1 / cell)); }
    if (f2 - s2 > 1e-3) { di.push(d); si.push(s2); al.push(Math.fround(Math.min(Math.min(f2 - s2, 1), cell) / cell)); }
  }
  return { di, si, al };
}

function resizeArea(src, dw, dh, scX, scY) {
  const { w: sw, c, data } = src;
  const out = newImg(dw, dh, c, Uint8Array), o = out.data;
  const xt = areaTab(sw, dw, scX), yt = areaTab(src.h, dh, scY);
  const rowLen = dw * c;
  const buf = new Float64Array(rowLen), sum = new Float64Array(rowLen);
  let prev = -1;
  const flush = (dy) => { for (let i = 0; i < rowLen; i++) o[dy * rowLen + i] = clamp(pyRound(sum[i]), 0, 255); };
  for (let j = 0; j < yt.di.length; j++) {
    const dy = yt.di[j], sy = yt.si[j], beta = yt.al[j];
    buf.fill(0);
    const row = sy * sw * c;
    for (let k = 0; k < xt.di.length; k++) {
      const dx = xt.di[k] * c, sx = row + xt.si[k] * c, a = xt.al[k];
      for (let ch = 0; ch < c; ch++) buf[dx + ch] += data[sx + ch] * a;
    }
    if (dy !== prev) {
      if (prev >= 0) flush(prev);
      for (let i = 0; i < rowLen; i++) sum[i] = buf[i] * beta;
      prev = dy;
    } else {
      for (let i = 0; i < rowLen; i++) sum[i] += buf[i] * beta;
    }
  }
  if (prev >= 0) flush(prev);
  return out;
}

function resizeAreaFast(src, dw, dh, ix, iy) {
  const { w: sw, c, data } = src;
  const out = newImg(dw, dh, c, Uint8Array), o = out.data;
  const area = ix * iy, inv = Math.fround(1 / area);
  for (let y = 0; y < dh; y++) {
    for (let x = 0; x < dw; x++) {
      for (let ch = 0; ch < c; ch++) {
        let s = 0;
        for (let j = 0; j < iy; j++) {
          const row = (y * iy + j) * sw;
          for (let i = 0; i < ix; i++) s += data[(row + x * ix + i) * c + ch];
        }
        o[(y * dw + x) * c + ch] = area === 4 ? (s + 2) >> 2 : clamp(pyRound(Math.fround(s * inv)), 0, 255);
      }
    }
  }
  return out;
}

// ═══════════════════════════════════════════════════════════════════════════
// cv2.blur, cv2.dilate (elipse) y cv2.connectedComponentsWithStats
// ═══════════════════════════════════════════════════════════════════════════

function reflect101(p, len) {
  if (len === 1) return 0;
  while (p < 0 || p >= len) p = p < 0 ? -p : 2 * len - 2 - p;
  return p;
}

/** cv2.blur normalizado con borde BORDER_REFLECT_101 (núcleo par o impar). */
function boxBlur(src, w, h, c, kw, kh) {
  const tmp = new Float64Array(w * h * c), out = new Float32Array(w * h * c);
  const ax = kw >> 1, ay = kh >> 1;
  const xmap = new Int32Array(w + kw - 1), ymap = new Int32Array(h + kh - 1);
  for (let i = 0; i < xmap.length; i++) xmap[i] = reflect101(i - ax, w);
  for (let i = 0; i < ymap.length; i++) ymap[i] = reflect101(i - ay, h);
  const pre = new Float64Array(Math.max(w + kw, h + kh));
  for (let y = 0; y < h; y++) {
    for (let ch = 0; ch < c; ch++) {
      const row = y * w * c + ch;
      pre[0] = 0;
      for (let i = 0; i < xmap.length; i++) pre[i + 1] = pre[i] + src[row + xmap[i] * c];
      for (let x = 0; x < w; x++) tmp[row + x * c] = pre[x + kw] - pre[x];
    }
  }
  const norm = 1 / (kw * kh);
  for (let x = 0; x < w; x++) {
    for (let ch = 0; ch < c; ch++) {
      const col = x * c + ch;
      pre[0] = 0;
      for (let i = 0; i < ymap.length; i++) pre[i + 1] = pre[i] + tmp[ymap[i] * w * c + col];
      for (let y = 0; y < h; y++) out[y * w * c + col] = (pre[y + kh] - pre[y]) * norm;
    }
  }
  return out;
}

/** cv2.dilate con getStructuringElement(MORPH_ELLIPSE, (2r+1, 2r+1)) sobre 0/1. */
function dilateEllipse(mask, w, h, r) {
  const hw = new Int32Array(2 * r + 1);
  for (let i = 0; i <= 2 * r; i++) {
    const dy = i - r;
    hw[i] = pyRound(r * Math.sqrt((r * r - dy * dy) * (1 / (r * r))));
  }
  const W1 = w + 1, pre = new Int32Array(W1 * h);
  for (let y = 0; y < h; y++) {
    let s = 0;
    for (let x = 0; x < w; x++) { s += mask[y * w + x]; pre[y * W1 + x + 1] = s; }
  }
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const orow = y * w;
    for (let i = 0; i <= 2 * r; i++) {
      const yy = y + i - r;
      if (yy < 0 || yy >= h || pre[yy * W1 + w] === 0) continue;
      const half = hw[i], prow = yy * W1;
      for (let x = 0; x < w; x++) {
        if (out[orow + x]) continue;
        const lo = x - half < 0 ? 0 : x - half, hi = x + half + 1 > w ? w : x + half + 1;
        if (pre[prow + hi] - pre[prow + lo] > 0) out[orow + x] = 1;
      }
    }
  }
  return out;
}

/**
 * cv2.connectedComponentsWithStats: etiquetas numeradas en el orden en que
 * aparece el primer píxel de cada componente (como OpenCV).
 */
function connectedComponents(mask, w, h, conn) {
  const N = w * h, labels = new Int32Array(N);
  const parent = new Int32Array((N >> 1) + 2);
  let next = 1;
  const find = (a) => { let r = a; while (parent[r] !== r) r = parent[r]; while (parent[a] !== r) { const n = parent[a]; parent[a] = r; a = n; } return r; };
  const unite = (a, b) => { a = find(a); b = find(b); if (a < b) parent[b] = a; else if (b < a) parent[a] = b; return a < b ? a : b; };
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!mask[i]) continue;
      let l = 0;
      const vecino = (j) => { const m = labels[j]; if (m) l = l ? unite(l, m) : find(m); };
      if (x > 0) vecino(i - 1);
      if (y > 0) {
        vecino(i - w);
        if (conn === 8) { if (x > 0) vecino(i - w - 1); if (x < w - 1) vecino(i - w + 1); }
      }
      if (!l) { parent[next] = next; l = next++; }
      labels[i] = l;
    }
  }
  const nuevo = new Int32Array(next);
  let n = 1;
  for (let l = 1; l < next; l++) if (find(l) === l) nuevo[l] = n++;
  const area = new Int32Array(n), x0 = new Int32Array(n).fill(w), y0 = new Int32Array(n).fill(h);
  const x1 = new Int32Array(n).fill(-1), y1 = new Int32Array(n).fill(-1);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!labels[i]) continue;
      const l = nuevo[find(labels[i])];
      labels[i] = l; area[l]++;
      if (x < x0[l]) x0[l] = x; if (x > x1[l]) x1[l] = x;
      if (y < y0[l]) y0[l] = y; if (y > y1[l]) y1[l] = y;
    }
  }
  return { n, labels, area, x0, y0, x1, y1 };
}

// ═══════════════════════════════════════════════════════════════════════════
// Modelos
// ═══════════════════════════════════════════════════════════════════════════

async function bajar({ url, size }, cache, onBytes) {
  const abs = new URL(url, self.location.href).href;
  if (cache) {
    const hit = await cache.match(abs);
    if (hit) {
      const b = new Uint8Array(await hit.arrayBuffer());
      if (b.length === size) { onBytes(size); return b; }
    }
  }
  const r = await fetch(abs);
  if (!r.ok) throw new Error(`no se pudo descargar ${url.split("/").pop()} (${r.status})`);
  const reader = r.body.getReader(), parts = [];
  let n = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); n += value.length; onBytes(value.length);
  }
  const bytes = new Uint8Array(n);
  let o = 0;
  for (const p of parts) { bytes.set(p, o); o += p.length; }
  if (cache && n === size) { try { await cache.put(abs, new Response(bytes)); } catch (_) { /* sin espacio */ } }
  return bytes;
}

async function cargarModelos() {
  let cache = null;
  try {
    cache = await caches.open(CACHE_NAME);
    // Modelos de versiones anteriores de la página: liberar el espacio
    for (const k of await caches.keys()) if (k.startsWith("fotocheck-modelos") && k !== CACHE_NAME) await caches.delete(k);
  } catch (_) { /* sin Cache API: se descargan cada vez */ }
  const total = MODELS.seg.size + MODELS.yunet.size;
  let loaded = 0;
  const onBytes = (n) => { loaded += n; post("progress", { loaded, total }); };
  const [seg, yunet] = await Promise.all([bajar(MODELS.seg, cache, onBytes), bajar(MODELS.yunet, cache, onBytes)]);
  post("stage", { text: "Preparando la IA…" });
  const opt = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
  sessions = {
    seg: await ort.InferenceSession.create(seg, opt),
    yunet: await ort.InferenceSession.create(yunet, opt),
  };
}

/** Segmentador.mascara (tipo u2net): alfa float32 H×W en [0, 1]. */
async function mascara(rgb) {
  const { w: W, h: H } = rgb, L = 320, n = L * L;
  const small = resize(rgb, L, L, Math.max(H, W) > L ? "area" : "linear");
  let mx = 0;
  for (const v of small.data) if (v > mx) mx = v;
  const div = Math.max(Math.fround(mx / 255), 1e-6);   // misma normalización que rembg
  const x = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) {
      const v = Math.fround(Math.fround(small.data[i * 3 + c] / 255) / div);
      x[c * n + i] = (v - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
    }
  }
  const s = sessions.seg;
  const res = await s.run({ [s.inputNames[0]]: new ort.Tensor("float32", x, [1, 3, L, L]) });
  const pred = new Float32Array(res[s.outputNames[0]].data);
  for (const k in res) res[k].dispose?.();
  let mn = Infinity, mxp = -Infinity;
  for (const v of pred) { if (v < mn) mn = v; if (v > mxp) mxp = v; }
  const den = Math.max(mxp - mn, 1e-6);
  for (let i = 0; i < n; i++) pred[i] = (pred[i] - mn) / den;
  const alpha = resize({ w: L, h: L, c: 1, data: pred }, W, H, "linear").data;
  for (let i = 0; i < alpha.length; i++) alpha[i] = clamp(alpha[i], 0, 1);
  return alpha;
}

// ═══════════════════════════════════════════════════════════════════════════
// Rostros (cv2.FaceDetectorYN con YuNet 2023mar)
// ═══════════════════════════════════════════════════════════════════════════

const YUNET_OUT = ["cls", "obj", "bbox", "kps"].flatMap((k) => [8, 16, 32].map((s) => `${k}_${s}`));

/** postProcess + NMSBoxes de FaceDetectorYN (umbrales en float, como OpenCV). */
function yunetDecode(out, pw, ph, umbral) {
  const thr = Math.fround(umbral), nms = Math.fround(YUNET_NMS);
  const faces = [];
  for (const s of [8, 16, 32]) {
    const cols = pw / s, rows = ph / s;
    const cls = out[`cls_${s}`], obj = out[`obj_${s}`], bb = out[`bbox_${s}`], kp = out[`kps_${s}`];
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const i = r * cols + c;
        const score = Math.fround(Math.sqrt(Math.fround(clamp(cls[i], 0, 1) * clamp(obj[i], 0, 1))));
        if (score < thr) continue;
        const cx = Math.fround((c + bb[i * 4]) * s), cy = Math.fround((r + bb[i * 4 + 1]) * s);
        const w = Math.fround(Math.fround(Math.exp(bb[i * 4 + 2])) * s), h = Math.fround(Math.fround(Math.exp(bb[i * 4 + 3])) * s);
        const f = [Math.fround(cx - w / 2), Math.fround(cy - h / 2), w, h];
        for (let k = 0; k < 5; k++) {
          f.push(Math.fround((kp[i * 10 + 2 * k] + c) * s), Math.fround((kp[i * 10 + 2 * k + 1] + r) * s));
        }
        f.push(score);
        faces.push(f);
      }
    }
  }
  if (faces.length <= 1) return faces;
  const boxes = faces.map((f) => f.slice(0, 4).map(trunc));
  const orden = faces.map((_, i) => i).filter((i) => faces[i][14] > thr)
    .sort((a, b) => faces[b][14] - faces[a][14]).slice(0, YUNET_TOP_K);
  const keep = [];
  for (const i of orden) {
    const a = boxes[i];
    const ok = keep.every((j) => {
      const b = boxes[j];
      const iw = Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]);
      const ih = Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]);
      const inter = iw > 0 && ih > 0 ? iw * ih : 0;
      const uni = a[2] * a[3] + b[2] * b[3] - inter;
      return !(uni > 0 && Math.fround(inter / uni) > nms);
    });
    if (ok) keep.push(i);
  }
  return keep.map((i) => faces[i]);
}

/** detect_faces: todas las caras, en coordenadas de la imagen. */
async function detectFaces(rgb) {
  const { w: W, h: H } = rgb;
  const esc = Math.min(1, YUNET_MAX_SIDE / Math.max(H, W));
  const small = esc === 1 ? rgb : resize(rgb, trunc(W * esc), trunc(H * esc), "area");
  const sw = small.w, sh = small.h;
  const pw = (trunc((sw - 1) / 32) + 1) * 32, ph = (trunc((sh - 1) / 32) + 1) * 32, n = pw * ph;
  const x = new Float32Array(3 * n);   // BGR 0-255, relleno con ceros abajo y a la derecha
  for (let y = 0; y < sh; y++) {
    for (let xx = 0; xx < sw; xx++) {
      const s = (y * sw + xx) * 3, d = y * pw + xx;
      x[d] = small.data[s + 2]; x[n + d] = small.data[s + 1]; x[2 * n + d] = small.data[s];
    }
  }
  const sess = sessions.yunet;
  const res = await sess.run({ [sess.inputNames[0]]: new ort.Tensor("float32", x, [1, 3, ph, pw]) });
  const out = {};
  for (const k of YUNET_OUT) out[k] = res[k].data;
  let faces = [];
  for (const thr of YUNET_SCORES) {
    faces = yunetDecode(out, pw, ph, thr);
    if (faces.length) break;
  }
  for (const k in res) res[k].dispose?.();
  const k = 1 / esc;
  return faces.map((f) => ({
    x: f[0] * k, y: f[1] * k, w: f[2] * k, h: f[3] * k,
    lm: {
      right_eye: [f[4] * k, f[5] * k], left_eye: [f[6] * k, f[7] * k], nose: [f[8] * k, f[9] * k],
      right_mouth: [f[10] * k, f[11] * k], left_mouth: [f[12] * k, f[13] * k],
    },
    score: f[14],
  }));
}

/** choose_face: grande, con buen score y SOBRE la persona segmentada. */
function chooseFace(caras, alpha, W, H) {
  if (!caras.length) return [null, 0];
  const puntaje = (c) => {
    const cx = trunc(clamp(c.x + c.w / 2, 0, W - 1)), cy = trunc(clamp(c.y + c.h / 2, 0, H - 1));
    return c.w * c.h * c.score * (alpha[cy * W + cx] > MASK_THRESHOLD ? 1.0 : 0.2);
  };
  const ordenadas = caras.map((c) => [puntaje(c), c]).sort((a, b) => b[0] - a[0]).map((p) => p[1]);
  const elegida = ordenadas[0], area = elegida.w * elegida.h;
  const otras = ordenadas.slice(1).filter((c) => c.w * c.h >= OTRA_CARA_RATIO * area).length;
  return [elegida, otras];
}

// ═══════════════════════════════════════════════════════════════════════════
// Máscara, encuadre y composición
// ═══════════════════════════════════════════════════════════════════════════

/** clean_mask: silueta del rostro (o la más grande) + huecos internos pequeños. */
function cleanMask(alpha, W, H, punto) {
  const N = W * H;
  const bin = new Uint8Array(N);
  for (let i = 0; i < N; i++) bin[i] = alpha[i] > MASK_THRESHOLD ? 1 : 0;
  const cc = connectedComponents(bin, W, H, 8);
  let out = alpha;
  if (cc.n > 1) {
    let et = 0;
    if (punto) et = cc.labels[trunc(clamp(punto[1], 0, H - 1)) * W + trunc(clamp(punto[0], 0, W - 1))];
    if (et === 0) {
      et = 1;
      for (let l = 2; l < cc.n; l++) if (cc.area[l] > cc.area[et]) et = l;
    }
    const keep = new Uint8Array(N);
    for (let i = 0; i < N; i++) keep[i] = cc.labels[i] === et ? 1 : 0;
    // Dilatar para no perder los mechones semitransparentes alrededor
    const r = Math.max(3, pyRound(Math.max(W, H) * 0.006));
    const k = dilateEllipse(keep, W, H, r);
    out = new Float32Array(N);
    for (let i = 0; i < N; i++) out[i] = alpha[i] * k[i];
  }

  let areaPersona = 0;
  const huecos = new Uint8Array(N);
  for (let i = 0; i < N; i++) { if (out[i] > MASK_THRESHOLD) areaPersona++; else huecos[i] = 1; }
  if (areaPersona > 0) {
    const hc = connectedComponents(huecos, W, H, 4);
    const rellenar = new Uint8Array(hc.n);
    let alguno = false;
    for (let l = 1; l < hc.n; l++) {
      if (hc.x0[l] === 0 || hc.y0[l] === 0 || hc.x1[l] + 1 >= W || hc.y1[l] + 1 >= H) continue;   // fondo exterior
      if (hc.area[l] <= areaPersona * HOLE_FILL_MAX_RATIO) { rellenar[l] = 1; alguno = true; }
    }
    if (alguno) {
      out = out === alpha ? new Float32Array(alpha) : out;
      for (let i = 0; i < N; i++) if (rellenar[hc.labels[i]]) out[i] = 1.0;
    }
  }
  return out;
}

/** Mediana por canal de los píxeles marcados (np.median sobre valores 0-255). */
function medianaRGB(data, sel, n) {
  const hist = [new Int32Array(256), new Int32Array(256), new Int32Array(256)];
  let cnt = 0;
  for (let i = 0; i < n; i++) {
    if (!sel[i]) continue;
    cnt++;
    for (let c = 0; c < 3; c++) hist[c][data[i * 3 + c]]++;
  }
  const kth = (h, k) => { let a = 0; for (let v = 0; v < 256; v++) { a += h[v]; if (a > k) return v; } return 255; };
  return hist.map((h) => (cnt % 2 ? kth(h, cnt >> 1) : (kth(h, cnt / 2 - 1) + kth(h, cnt / 2)) / 2));
}

/** refine_hair_alpha: recupera puntas de cabello en fondo liso. */
function refineHairAlpha(rgb, alpha, cara) {
  const { w: W, h: H } = rgb;
  const yLim = trunc(Math.min(H, (cara.lm.right_eye[1] + cara.lm.left_eye[1]) / 2));
  if (yLim < 4) return { alpha, ganados: 0 };
  const n = yLim * W, reg = rgb.data;

  const fondo = new Uint8Array(n);
  let cntF = 0;
  for (let i = 0; i < n; i++) if (alpha[i] < 0.05) { fondo[i] = 1; cntF++; }
  if (cntF < 500) return { alpha, ganados: 0 };
  const bgMed = medianaRGB(reg, fondo, n);
  // Las puntas de pelo caen en la zona "fondo": se excluyen del color de fondo
  const limpio = new Uint8Array(n);
  let cntL = 0;
  const suma = [0, 0, 0], suma2 = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    if (!fondo[i]) continue;
    let d = 0;
    for (let c = 0; c < 3; c++) d = Math.max(d, Math.abs(reg[i * 3 + c] - bgMed[c]));
    if (d < 40) {
      limpio[i] = 1; cntL++;
      for (let c = 0; c < 3; c++) { const v = reg[i * 3 + c]; suma[c] += v; suma2[c] += v * v; }
    }
  }
  if (cntL < 500 || cntL < HAIR_BG_MIN_UNIFORM * cntF) return { alpha, ganados: 0 };
  let stdMax = 0;
  for (let c = 0; c < 3; c++) {
    const m = suma[c] / cntL;
    stdMax = Math.max(stdMax, Math.sqrt(Math.max(0, suma2[c] / cntL - m * m)));
  }
  if (stdMax > HAIR_BG_MAX_STD) return { alpha, ganados: 0 };

  // Color de fondo LOCAL (convolución normalizada)
  const k = Math.max(15, trunc(0.25 * cara.w)) | 1;
  const m = new Float32Array(n), rm = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    if (!limpio[i]) continue;
    m[i] = 1;
    for (let c = 0; c < 3; c++) rm[i * 3 + c] = reg[i * 3 + c];
  }
  const den = boxBlur(m, W, yLim, 1, k, k);
  const num = boxBlur(rm, W, yLim, 3, k, k);

  // Solo lo MÁS OSCURO que el fondo y con seguridad (lo dudoso deja velos)
  const aColor = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let bgm, rgm = (reg[i * 3] + reg[i * 3 + 1] + reg[i * 3 + 2]) / 3;
    if (den[i] < 0.02) bgm = (bgMed[0] + bgMed[1] + bgMed[2]) / 3;
    else {
      const dd = Math.max(den[i], 1e-3);
      bgm = (num[i * 3] / dd + num[i * 3 + 1] / dd + num[i * 3 + 2] / dd) / 3;
    }
    let v = clamp((bgm - rgm - HAIR_T0) / (HAIR_T1 - HAIR_T0), 0, 1);
    aColor[i] = v < HAIR_MIN_ALPHA ? 0 : v;
  }

  const r = Math.max(3, trunc(HAIR_BAND * cara.h));
  const persona = new Uint8Array(n);
  for (let i = 0; i < n; i++) persona[i] = alpha[i] > MASK_THRESHOLD ? 1 : 0;
  const banda = dilateEllipse(persona, W, yLim, r);
  const out = new Float32Array(alpha);
  let ganados = 0;
  for (let i = 0; i < n; i++) {
    if (!banda[i]) continue;
    const v = Math.max(alpha[i], aColor[i]);
    if (v > MASK_THRESHOLD && alpha[i] <= MASK_THRESHOLD) ganados++;
    out[i] = v;
  }
  return { alpha: out, ganados };
}

/** person_bbox_from_mask: (x, y, w, h) o null. */
function personBBox(alpha, W, H) {
  let x0 = W, y0 = H, x1 = -1, y1 = -1;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      if (alpha[y * W + x] <= MASK_THRESHOLD) continue;
      if (x < x0) x0 = x; if (x > x1) x1 = x;
      if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
  }
  return x1 < 0 ? null : [x0, y0, x1 - x0 + 1, y1 - y0 + 1];
}

/** detect_long_hair: ¿cabello junto a la mandíbula? → [es_largo, cobertura]. */
function detectLongHair(alpha, W, H, cara) {
  const { x, y, w, h, lm } = cara;
  const bocaY = (lm.right_mouth[1] + lm.left_mouth[1]) / 2;
  const y0 = trunc(Math.max(0, bocaY)), y1 = trunc(Math.min(H, y + h - 0.05 * h));
  if (y1 - y0 < 3) return [false, 0];
  const lados = [];
  for (let [xa, xb] of [[x - 0.35 * w, x - 0.08 * w], [x + 1.08 * w, x + 1.35 * w]]) {
    xa = trunc(Math.max(0, xa)); xb = trunc(Math.min(W, xb));
    if (xb - xa < 3) continue;
    let si = 0;
    for (let yy = y0; yy < y1; yy++) for (let xx = xa; xx < xb; xx++) if (alpha[yy * W + xx] > MASK_THRESHOLD) si++;
    lados.push(si / ((y1 - y0) * (xb - xa)));
  }
  const cob = lados.length ? Math.max(...lados) : 0;
  return [cob >= LONG_HAIR_MIN_COVERAGE, cob];
}

/** find_crown: punto más alto de la cabeza (con tope anatómico) → [y, fuente]. */
function findCrown(alpha, W, H, cara) {
  const { x: fx, y: fy, w: fw, h: fh, lm } = cara;
  const eyeY = (lm.right_eye[1] + lm.left_eye[1]) / 2;
  const d = Math.max(1, (fy + fh) - eyeY);
  const masBaja = eyeY - CROWN_MIN_ABOVE_EYES * d;
  const x0 = trunc(Math.max(0, fx - 0.5 * fw)), x1 = trunc(Math.min(W, fx + 1.5 * fw));
  const yFin = trunc(Math.min(H, Math.max(1, fy + 0.2 * fh)));
  const minimo = Math.max(3, trunc(0.01 * fw));   // ignora motas sueltas
  let crown = yFin;
  for (let y = 0; y < yFin; y++) {
    let filas = 0;
    for (let x = x0; x < x1; x++) if (alpha[y * W + x] > MASK_THRESHOLD) filas++;
    if (filas >= minimo) { crown = y; break; }
  }
  if (crown > masBaja) return [Math.max(0, masBaja), "anatomía"];
  return [crown, "máscara"];
}

/** compute_crop: rectángulo [x0, y0, x1, y1] y modo. ratio = ancho / alto de salida. */
function computeCrop(W, H, face, personBox, crown, headRatio, headSpace, ratio) {
  const { x: fx, y: fy, w: fw, h: fh } = face;
  const fxCenter = fx + fw / 2, fyCenter = fy + fh / 2;
  let modo = "normal";

  const headH = crown !== null ? (fy + fh) - crown : null;
  let cropH;
  if (headH !== null && headH > 0) { cropH = headH / headRatio; modo = "por altura de cabeza"; }
  else cropH = fh / FACE_HEIGHT_RATIO_YUNET;
  let cropW = cropH * ratio;

  if (personBox) {
    const conMargen = personBox[2] * (1 + 2 * SIDE_MARGIN);
    if (conMargen < cropW) { cropW = conMargen; cropH = cropW / ratio; modo = "ajustado al ancho de la persona"; }
  }
  const limitar = () => {
    if (cropH > H) { cropH = H; cropW = cropH * ratio; }
    if (cropW > W) { cropW = W; cropH = cropW / ratio; }
  };
  limitar();

  let top;
  if (crown !== null) top = Math.min(crown - headSpace * cropH, fy - HEAD_TOP_MARGIN * cropH);
  else top = fyCenter - FACE_CENTER_Y_RATIO * cropH;

  const faceBottomNeeded = (fy + fh) + CHIN_MARGIN * fh;
  if (top + cropH < faceBottomNeeded) {
    cropH = faceBottomNeeded - top; cropW = cropH * ratio;
    modo = modo === "normal" ? "ampliado (cabello alto)" : modo + " + ampliado";
  }
  if (personBox) {
    const personBottom = personBox[1] + personBox[3];
    if (personBottom < H - 1) {
      const disponible = personBottom - Math.max(0, top);
      if (disponible > 0 && disponible < cropH) { cropH = disponible; cropW = cropH * ratio; modo += " + sin banda inferior"; }
    }
  }
  limitar();

  // Poco aire sobre la cabeza: si el recorte se sale por abajo, se acorta
  // dejando fijo el borde superior en vez de subirlo.
  if (crown !== null) {
    top = Math.max(0, top);
    if (top + cropH > H) {
      const nuevoH = H - top, minimo = (fy + fh) + CHIN_MARGIN * fh - top;
      if (nuevoH >= minimo) { cropH = nuevoH; cropW = cropH * ratio; modo += " + acortado abajo"; }
    }
  }
  let left = fxCenter - cropW / 2;
  top = Math.max(0, Math.min(top, H - cropH));
  left = Math.max(0, Math.min(left, W - cropW));
  return [[left, top, left + cropW, top + cropH], modo];
}

/** center_crop_rect: sin rostro, centro de la imagen a la proporción de salida. */
function centerCropRect(W, H, ratio) {
  if (W / H > ratio) { const nw = H * ratio; return [(W - nw) / 2, 0, (W + nw) / 2, H]; }
  const nh = W / ratio;
  return [0, (H - nh) / 2, W, (H + nh) / 2];
}

/** _blur_fusion: Fast Foreground Colour Estimation (Forte & Pitié, 2021). */
function blurFusion(image, F, B, alpha, w, h, r) {
  const n = w * h;
  const bA = boxBlur(alpha, w, h, 1, r, r);
  const fa = new Float32Array(n * 3), b1a = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const a = alpha[i];
    for (let c = 0; c < 3; c++) { fa[i * 3 + c] = F[i * 3 + c] * a; b1a[i * 3 + c] = B[i * 3 + c] * (1 - a); }
  }
  const bF = boxBlur(fa, w, h, 3, r, r), bB = boxBlur(b1a, w, h, 3, r, r);
  const outF = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const a = alpha[i], ba = bA[i];
    for (let c = 0; c < 3; c++) {
      const j = i * 3 + c;
      const f = bF[j] / (ba + 1e-5), b = bB[j] / ((1 - ba) + 1e-5);
      bB[j] = b;
      outF[j] = clamp(f + a * (image[j] - a * f - (1 - a) * b), 0, 1);
    }
  }
  return [outF, bB];
}

/** estimate_foreground: color limpio de la persona en los bordes semitransparentes. */
function estimateForeground(image, alpha, w, h) {
  const [F, B] = blurFusion(image, image, image, alpha, w, h, 90);
  return blurFusion(image, F, B, alpha, w, h, 6)[0];
}

/** compose_on_white: recorta, escala al tamaño de salida y compone sobre blanco (RGBA). */
function composeOnWhite(rgb, alpha, rect, TW, TH) {
  const { w: W, h: H } = rgb;
  const [x0, y0, x1, y1] = rect.map(pyRound).map((v, i) => clamp(v, 0, i % 2 ? H : W));
  const cw = x1 - x0, ch = y1 - y0;
  const crop = newImg(cw, ch, 3, Uint8Array), aCrop = newImg(cw, ch, 1, Float32Array);
  for (let y = 0; y < ch; y++) {
    crop.data.set(rgb.data.subarray(((y0 + y) * W + x0) * 3, ((y0 + y) * W + x1) * 3), y * cw * 3);
    aCrop.data.set(alpha.subarray((y0 + y) * W + x0, (y0 + y) * W + x1), y * cw);
  }
  const cR = resize(crop, TW, TH, cw >= TW ? "area" : "lanczos4").data;
  const a = resize(aCrop, TW, TH, "linear").data;
  const n = TW * TH;
  for (let i = 0; i < n; i++) a[i] = clamp(a[i], 0, 1);
  const img = new Float32Array(n * 3);
  for (let i = 0; i < n * 3; i++) img[i] = cR[i] / 255;
  const F = DECONTAMINAR_BORDES ? estimateForeground(img, a, TW, TH) : img;
  const px = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) px[i * 4 + c] = trunc(clamp((F[i * 3 + c] * a[i] + (1 - a[i])) * 255 + 0.5, 0, 255));
    px[i * 4 + 3] = 255;
  }
  return px;
}

/** estimate_tilt: ojos + eje nariz→boca deben coincidir para corregir. */
function estimateTilt(lm) {
  const deg = 180 / Math.PI;
  const angleEyes = Math.atan2(lm.left_eye[1] - lm.right_eye[1], lm.left_eye[0] - lm.right_eye[0]) * deg;
  const axX = (lm.right_mouth[0] + lm.left_mouth[0]) / 2 - lm.nose[0];
  const axY = (lm.right_mouth[1] + lm.left_mouth[1]) / 2 - lm.nose[1];
  const angleAxis = Math.atan2(axX, axY) * deg;
  return [(angleEyes + angleAxis) / 2, Math.abs(angleEyes - angleAxis) <= TILT_EYES_MOUTH_MAX_DIFF];
}

/** Matriz de Image.rotate(angle, expand=True) de PIL. */
function pilRotateMatrix(w, h, angle) {
  const r15 = (v) => Math.round(v * 1e15) / 1e15;
  const rad = -((((angle % 360) + 360) % 360) * Math.PI / 180);
  const m = [r15(Math.cos(rad)), r15(Math.sin(rad)), 0, r15(-Math.sin(rad)), r15(Math.cos(rad)), 0];
  const tr = (x, y) => [m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5]];
  [m[2], m[5]] = tr(-w / 2, -h / 2);
  m[2] += w / 2; m[5] += h / 2;
  const xs = [], ys = [];
  for (const [x, y] of [[0, 0], [w, 0], [w, h], [0, h]]) { const [a, b] = tr(x, y); xs.push(a); ys.push(b); }
  const nw = Math.ceil(Math.max(...xs)) - Math.floor(Math.min(...xs));
  const nh = Math.ceil(Math.max(...ys)) - Math.floor(Math.min(...ys));
  [m[2], m[5]] = tr(-(nw - w) / 2, -(nh - h) / 2);
  return { m, nw, nh };
}

/** Image.rotate con BICUBIC (RGB, relleno blanco) o BILINEAR (alfa, relleno 0). */
function pilRotate(src, angle, bicubic, fill) {
  const { w, h, c, data } = src;
  const { m, nw, nh } = pilRotateMatrix(w, h, angle);
  const out = newImg(nw, nh, c, data.constructor), o = out.data;
  const xc = (v) => (v < 0 ? 0 : v >= w ? w - 1 : v), yc = (v) => (v < 0 ? 0 : v >= h ? h - 1 : v);
  const cub = (v1, v2, v3, v4, d) => v2 + d * (-v1 + v3 + d * (2 * (v1 - v2) + v3 - v4 + d * (-v1 + v2 - v3 + v4)));
  for (let y = 0; y < nh; y++) {
    for (let x = 0; x < nw; x++) {
      const op = (y * nw + x) * c;
      let xin = m[0] * (x + 0.5) + m[1] * (y + 0.5) + m[2];
      let yin = m[3] * (x + 0.5) + m[4] * (y + 0.5) + m[5];
      if (xin < 0 || xin >= w || yin < 0 || yin >= h) { for (let k = 0; k < c; k++) o[op + k] = fill[k]; continue; }
      xin -= 0.5; yin -= 0.5;
      const ix = Math.floor(xin), iy = Math.floor(yin), dx = xin - ix, dy = yin - iy;
      for (let k = 0; k < c; k++) {
        const px = (xx, yy) => data[(yc(yy) * w + xc(xx)) * c + k];
        if (bicubic) {
          const fila = (yy) => cub(px(ix - 1, yy), px(ix, yy), px(ix + 1, yy), px(ix + 2, yy), dx);
          const v = cub(fila(iy - 1), fila(iy), fila(iy + 1), fila(iy + 2), dy);
          o[op + k] = v <= 0 ? 0 : v >= 255 ? 255 : trunc(v);
        } else {
          const v1 = px(ix, iy) + (px(ix + 1, iy) - px(ix, iy)) * dx;
          const v2 = iy + 1 < h ? px(ix, iy + 1) + (px(ix + 1, iy + 1) - px(ix, iy + 1)) * dx : v1;
          o[op + k] = v1 + (v2 - v1) * dy;
        }
      }
    }
  }
  return out;
}

// ═══════════════════════════════════════════════════════════════════════════
// Flujo (frame_photo, warnings_for, process_photo)
// ═══════════════════════════════════════════════════════════════════════════

/** Primera mitad de frame_photo: todo lo que no depende del tamaño de salida. */
async function analizar(rgb, alphaRaw) {
  const { w: W, h: H } = rgb;
  const caras = await detectFaces(rgb);
  const [cara, otras] = chooseFace(caras, alphaRaw, W, H);
  const punto = cara ? [cara.x + cara.w / 2, cara.y + cara.h / 2] : null;
  let alpha = cleanMask(alphaRaw, W, H, punto);
  let ganados = 0;
  if (cara && REFINAR_CABELLO) ({ alpha, ganados } = refineHairAlpha(rgb, alpha, cara));
  const st = { rgb, alpha, cara, otras, ganados, persona: personBBox(alpha, W, H),
               perfil: null, origen: null, coronilla: null, fuente: null };
  if (cara) {
    const [largo, cob] = detectLongHair(alpha, W, H, cara);
    st.perfil = largo ? "mujer" : "varón";
    st.origen = `por cabello ${pct(cob)}`;
    [st.coronilla, st.fuente] = findCrown(alpha, W, H, cara);
  }
  return st;
}

/** Segunda mitad de frame_photo: recorte para un tamaño de salida. */
function encuadrar(st, TW, TH) {
  const { w: W, h: H } = st.rgb, ratio = TW / TH;
  if (!st.cara) return { rect: centerCropRect(W, H, ratio), modo: "sin rostro" };
  const e = ENCUADRES[st.perfil];
  const [rect, m] = computeCrop(W, H, st.cara, st.persona, st.coronilla, e.head, e.space, ratio);
  let modo = `${st.perfil} (${st.origen}), coronilla por ${st.fuente}, ${m}`;
  if (st.ganados) modo += `, puntas de cabello +${st.ganados} px`;
  return { rect, modo };
}

/** warnings_for. escala = lado original / lado procesado (para medir en la foto original). */
function warningsFor(st, rect, TW, escala) {
  const avisos = [];
  if (!st.cara) avisos.push("sin rostro (recorte centrado)");
  else {
    if (st.otras > 0) avisos.push(`${st.otras + 1} caras`);
    if (st.cara.h * escala < MIN_FACE_PX) avisos.push("rostro muy pequeño");
  }
  if (!st.persona) avisos.push("no se segmentó a la persona");
  else if (st.coronilla !== null && st.coronilla <= 1) avisos.push("cabeza cortada en la foto original");
  const cw = (rect[2] - rect[0]) * escala;
  if (cw > 0 && TW / cw > MAX_UPSCALE) avisos.push(`baja resolución (ampliada x${(TW / cw).toFixed(1)})`);
  return avisos;
}

function leerFoto(bitmap) {
  const W0 = bitmap.width, H0 = bitmap.height;
  const cv = new OffscreenCanvas(W0, H0);
  const ctx = cv.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  const d = ctx.getImageData(0, 0, W0, H0).data;
  let rgb = newImg(W0, H0, 3, Uint8Array);
  for (let i = 0, n = W0 * H0; i < n; i++) {
    rgb.data[i * 3] = d[i * 4]; rgb.data[i * 3 + 1] = d[i * 4 + 1]; rgb.data[i * 3 + 2] = d[i * 4 + 2];
  }
  let escala = 1;
  if (Math.max(W0, H0) > MAX_LADO) {
    const e = MAX_LADO / Math.max(W0, H0);
    rgb = resize(rgb, Math.max(1, Math.round(W0 * e)), Math.max(1, Math.round(H0 * e)), "area");
    escala = W0 / rgb.w;
  }
  return { rgb, escala };
}

async function aJpeg(px, w, h) {
  const cv = new OffscreenCanvas(w, h);
  cv.getContext("2d").putImageData(new ImageData(px, w, h), 0, 0);
  return cv.convertToBlob({ type: "image/jpeg", quality: JPEG_QUALITY });
}

async function procesar(bitmap, TW, TH, raw) {
  const t0 = performance.now();
  const { rgb, escala } = leerFoto(bitmap);
  post("stage", { text: "Separando a la persona del fondo…" });
  const alphaRaw = await mascara(rgb);
  post("stage", { text: "Encuadrando el rostro…" });
  const st = await analizar(rgb, alphaRaw);

  // Copia corregida si la cabeza está inclinada de forma confiable
  let corr = null;
  if (st.cara) {
    const [angulo, confiable] = estimateTilt(st.cara.lm);
    if (confiable && Math.abs(angulo) >= TILT_MIN_DEG && Math.abs(angulo) <= TILT_MAX_DEG) {
      post("stage", { text: "Enderezando la cabeza…" });
      const rgbR = pilRotate(rgb, angulo, true, [255, 255, 255]);
      const aR = pilRotate({ w: rgb.w, h: rgb.h, c: 1, data: alphaRaw }, angulo, false, [0]).data;
      for (let i = 0; i < aR.length; i++) aR[i] = clamp(aR[i], 0, 1);
      corr = { st: await analizar(rgbR, aR), angulo };
    }
  }
  last = { st, corr, escala, raw, ms: performance.now() - t0 };
  return render(TW, TH);
}

async function render(TW, TH) {
  if (!last) throw new Error("no hay foto cargada");
  if (!Number.isInteger(TW) || !Number.isInteger(TH) || Math.min(TW, TH) < MIN_LADO_SALIDA || Math.max(TW, TH) > MAX_LADO_SALIDA) {
    throw new Error("tamaño de salida no válido");
  }
  const { st, corr, escala, raw } = last;
  const transfer = [];
  const salida = (s) => { const { rect, modo } = encuadrar(s, TW, TH); return { rect, modo, px: composeOnWhite(s.rgb, s.alpha, rect, TW, TH) }; };
  const a = salida(st);
  const info = {
    salida: [TW, TH], ms: Math.round(last.ms), rostro: !!st.cara, perfil: st.perfil,
    modo: a.modo, rect: a.rect, avisos: warningsFor(st, a.rect, TW, escala),
    escala, procesada: [st.rgb.w, st.rgb.h], corregida: null,
  };
  const res = { info, blob: await aJpeg(a.px, TW, TH) };
  if (raw) { res.pixels = a.px; transfer.push(a.px.buffer); }
  if (corr) {
    const b = salida(corr.st);
    info.corregida = { angulo: corr.angulo, modo: b.modo, rect: b.rect, rostro: !!corr.st.cara };
    res.blobCorregida = await aJpeg(b.px, TW, TH);
    if (raw) { res.pixelsCorregida = b.px; transfer.push(b.px.buffer); }
  }
  return [res, transfer];
}

self.onmessage = async (e) => {
  const { type } = e.data;
  try {
    if (type === "init") {
      if (!sessions) await cargarModelos();
      post("ready");
    } else if (type === "process") {
      if (!sessions) await cargarModelos();
      const [res, transfer] = await procesar(e.data.bitmap, e.data.w, e.data.h, !!e.data.raw);
      post("done", res, transfer);
    } else if (type === "resize") {
      const [res, transfer] = await render(e.data.w, e.data.h);
      post("done", { ...res, resized: true }, transfer);
    }
  } catch (err) {
    post("error", { message: String(err && err.message ? err.message : err) });
  }
};
