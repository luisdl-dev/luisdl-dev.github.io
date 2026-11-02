/* Motor de procesamiento: corre en un Web Worker, 100% en el navegador.
 * Pipeline (equivalente al script de escritorio v9):
 *   1) Segmentación con ormbg (2024, ISNet reentrenado solo con personas)
 *   2) Limpieza de máscara (componente principal, huecos pequeños, suavizado)
 *   3) Detección de rostro YuNet
 *   4) Recorte por altura de cabeza al tamaño elegido (480x640 por defecto)
 */
importScripts("../vendor/ort.wasm.min.js");

const MODEL_VERSION = "v5";   // subir este número si se cambian los modelos (invalida la caché)
const MODELS = {
  seg: { url: "../models/seg_ormbg_int8.onnx", size: 46360719 },
  yunet: { url: "../models/yunet.onnx", size: 229657 },
};

// ── Configuración (mismos valores que el script de escritorio) ──────────────
const DEFAULT_W = 480, DEFAULT_H = 640;
const MIN_LADO_SALIDA = 64, MAX_LADO_SALIDA = 4000;
const MAX_LADO = 1600;
const HEAD_HEIGHT_RATIO = 0.58;
const HEAD_SPACE_RATIO = 0.06;
const HEAD_TOP_MARGIN = 0.06;
const FACE_CENTER_Y_RATIO = 0.40;
const FACE_HEIGHT_RATIO_YUNET = 0.30;
const CHIN_MARGIN = 0.25;
const SIDE_MARGIN = 0.03;
const HOLE_FILL_MAX_RATIO = 0.02;
const YUNET_SCORE_MIN = 0.6;

let sessions = null;
let last = null;   // última foto ya sin fondo: permite cambiar el tamaño sin reprocesar

ort.env.wasm.wasmPaths = "../vendor/";
ort.env.logLevel = "error";
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

const post = (type, data = {}) => self.postMessage({ type, ...data });

// ── Descarga de modelos con progreso y caché persistente ────────────────────
async function fetchModel(key, onBytes) {
  const { url } = MODELS[key];
  const cacheKey = new URL(url, self.location.href).href + "?" + MODEL_VERSION;
  let cache = null;
  try { cache = await caches.open("fotocheck-modelos"); } catch (_) { /* sin Cache API */ }
  if (cache) {
    const hit = await cache.match(cacheKey);
    if (hit) { const buf = await hit.arrayBuffer(); onBytes(buf.byteLength); return buf; }
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`No se pudo descargar el modelo (${res.status})`);
  const reader = res.body.getReader();
  const chunks = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.byteLength; onBytes(value.byteLength);
  }
  const buf = new Uint8Array(got); let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.byteLength; }
  if (cache) { try { await cache.put(cacheKey, new Response(buf.slice().buffer)); } catch (_) {} }
  return buf.buffer;
}

async function loadModels() {
  const total = Object.values(MODELS).reduce((s, m) => s + m.size, 0);
  let loaded = 0;
  const onBytes = (n) => { loaded += n; post("progress", { loaded, total }); };
  const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all", logSeverityLevel: 3 };
  const [h, y] = await Promise.all(["seg", "yunet"].map((k) => fetchModel(k, onBytes)));
  sessions = {
    seg: await ort.InferenceSession.create(h, opts),
    yunet: await ort.InferenceSession.create(y, opts),
  };
}

// ── Utilidades de imagen ────────────────────────────────────────────────────
function drawTo(src, w, h) {
  const c = new OffscreenCanvas(w, h);
  const g = c.getContext("2d", { willReadFrequently: true });
  g.imageSmoothingEnabled = true; g.imageSmoothingQuality = "high";
  g.drawImage(src, 0, 0, w, h);
  return c;
}
const pixels = (c) => c.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, c.width, c.height).data;

function grayToCanvas(gray, w, h) {
  const c = new OffscreenCanvas(w, h), g = c.getContext("2d");
  const im = g.createImageData(w, h);
  for (let k = 0, p = 0; k < gray.length; k++, p += 4) { im.data[p] = im.data[p + 1] = im.data[p + 2] = gray[k]; im.data[p + 3] = 255; }
  g.putImageData(im, 0, 0);
  return c;
}

// ── Segmentación ────────────────────────────────────────────────────────────
async function segment(session, srcCanvas, size, mean, std, escala = 0) {
  const px = pixels(drawTo(srcCanvas, size, size));
  const n = size * size;
  let mx = escala;   // 0 = normalizar por el máximo (estilo rembg)
  if (!mx) { mx = 1e-6; for (let k = 0; k < px.length; k += 4) mx = Math.max(mx, px[k], px[k + 1], px[k + 2]); }
  const t = new Float32Array(3 * n);
  for (let k = 0, p = 0; k < n; k++, p += 4) {
    t[k] = (px[p] / mx - mean[0]) / std[0];
    t[n + k] = (px[p + 1] / mx - mean[1]) / std[1];
    t[2 * n + k] = (px[p + 2] / mx - mean[2]) / std[2];
  }
  const feeds = { [session.inputNames[0]]: new ort.Tensor("float32", t, [1, 3, size, size]) };
  const out = (await session.run(feeds))[session.outputNames[0]].data;
  let mi = Infinity, ma = -Infinity;
  for (let k = 0; k < n; k++) { const v = out[k]; if (v < mi) mi = v; if (v > ma) ma = v; }
  const g = new Uint8ClampedArray(n);
  const span = ma - mi || 1e-9;
  for (let k = 0; k < n; k++) g[k] = Math.floor(((out[k] - mi) / span) * 255);
  // Escalar la máscara al tamaño de la foto
  const W = srcCanvas.width, H = srcCanvas.height;
  const up = pixels(drawTo(grayToCanvas(g, size, size), W, H));
  const m = new Uint8Array(W * H);
  for (let k = 0, p = 0; k < m.length; k++, p += 4) m[k] = up[p];
  return m;
}

// ── Filtros (equivalentes a cv2.boxFilter normalizado, borde replicado) ─────
function boxFilter(src, W, H, r) {
  const tmp = new Float32Array(W * H), out = new Float32Array(W * H), d = 2 * r + 1;
  for (let y = 0; y < H; y++) {
    const row = y * W; let s = 0;
    for (let x = -r; x <= r; x++) s += src[row + Math.min(W - 1, Math.max(0, x))];
    for (let x = 0; x < W; x++) {
      tmp[row + x] = s;
      s += src[row + Math.min(W - 1, x + r + 1)] - src[row + Math.max(0, x - r)];
    }
  }
  for (let x = 0; x < W; x++) {
    let s = 0;
    for (let y = -r; y <= r; y++) s += tmp[Math.min(H - 1, Math.max(0, y)) * W + x];
    for (let y = 0; y < H; y++) {
      out[y * W + x] = s / (d * d);
      s += tmp[Math.min(H - 1, y + r + 1) * W + x] - tmp[Math.max(0, y - r) * W + x];
    }
  }
  return out;
}

function guided(gray, p8, W, H, r, eps = 1e-3) {
  const n = W * H, I = new Float32Array(n), p = new Float32Array(n), II = new Float32Array(n), Ip = new Float32Array(n);
  for (let k = 0; k < n; k++) { I[k] = gray[k] / 255; p[k] = p8[k] / 255; II[k] = I[k] * I[k]; Ip[k] = I[k] * p[k]; }
  const mI = boxFilter(I, W, H, r), mp = boxFilter(p, W, H, r), mII = boxFilter(II, W, H, r), mIp = boxFilter(Ip, W, H, r);
  const a = new Float32Array(n), b = new Float32Array(n);
  for (let k = 0; k < n; k++) {
    a[k] = (mIp[k] - mI[k] * mp[k]) / (mII[k] - mI[k] * mI[k] + eps);
    b[k] = mp[k] - a[k] * mI[k];
  }
  const ma = boxFilter(a, W, H, r), mb = boxFilter(b, W, H, r);
  const q = new Uint8Array(n);
  for (let k = 0; k < n; k++) q[k] = Math.floor(Math.min(1, Math.max(0, ma[k] * I[k] + mb[k])) * 255);
  return q;
}

// Decide los píxeles dudosos por color: ¿se parecen más a la persona o al fondo cercanos?
function colorMatte(rgba, m, W, H, fgT = 235, bgT = 12) {
  const n = W * H, R = Math.max(4, Math.round(Math.max(W, H) / 40));
  const fg = new Float32Array(n), bg = new Float32Array(n);
  for (let k = 0; k < n; k++) { fg[k] = m[k] >= fgT ? 1 : 0; bg[k] = m[k] <= bgT ? 1 : 0; }
  const nf = boxFilter(fg, W, H, R), nb = boxFilter(bg, W, H, R);
  const F = [], B = [];
  for (let c = 0; c < 3; c++) {
    const If = new Float32Array(n), Ib = new Float32Array(n);
    for (let k = 0; k < n; k++) { const v = rgba[k * 4 + c] / 255; If[k] = v * fg[k]; Ib[k] = v * bg[k]; }
    F.push(boxFilter(If, W, H, R)); B.push(boxFilter(Ib, W, H, R));
  }
  const out = new Uint8Array(n);
  for (let k = 0; k < n; k++) {
    let v = m[k] / 255;
    if (fg[k]) v = 1;
    else if (bg[k]) v = 0;
    else if (nf[k] > 1e-3 && nb[k] > 1e-3) {
      const df = Math.max(nf[k], 1e-4), db = Math.max(nb[k], 1e-4);
      let num = 0, den = 0;
      for (let c = 0; c < 3; c++) {
        const Fc = F[c][k] / df, Bc = B[c][k] / db, I = rgba[k * 4 + c] / 255, d = Fc - Bc;
        num += (I - Bc) * d; den += d * d;
      }
      v = Math.min(1, Math.max(0, num / Math.max(den, 1e-4)));
    }
    out[k] = Math.floor(v * 255);
  }
  return out;
}

// ── Limpieza de máscara (componente principal, huecos, suavizado) ──────────
function labelComponents(bin, W, H, eight) {
  const lab = new Int32Array(W * H), areas = [0], stack = new Int32Array(W * H);
  let next = 0;
  for (let s = 0; s < bin.length; s++) {
    if (!bin[s] || lab[s]) continue;
    next++; let area = 0, top = 0;
    stack[top++] = s; lab[s] = next;
    let minX = W, minY = H, maxX = -1, maxY = -1;
    while (top) {
      const k = stack[--top], x = k % W, y = (k / W) | 0; area++;
      if (x < minX) minX = x; if (x > maxX) maxX = x; if (y < minY) minY = y; if (y > maxY) maxY = y;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        if (!dx && !dy) continue;
        if (!eight && dx && dy) continue;
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
        const q = ny * W + nx;
        if (bin[q] && !lab[q]) { lab[q] = next; stack[top++] = q; }
      }
    }
    areas.push({ area, border: minX === 0 || minY === 0 || maxX === W - 1 || maxY === H - 1 });
  }
  return { lab, comps: areas };
}

function cleanMask(a, W, H) {
  const n = W * H;
  let bin = new Uint8Array(n);
  for (let k = 0; k < n; k++) bin[k] = a[k] > 127 ? 1 : 0;
  const { lab, comps } = labelComponents(bin, W, H, true);
  if (comps.length > 2) {
    let best = 1;
    for (let i = 2; i < comps.length; i++) if (comps[i].area > comps[best].area) best = i;
    for (let k = 0; k < n; k++) if (lab[k] !== best) a[k] = 0;
  }
  let personArea = 0;
  for (let k = 0; k < n; k++) { bin[k] = a[k] > 127 ? 0 : 1; personArea += 1 - bin[k]; }
  if (personArea > 0) {
    const holes = labelComponents(bin, W, H, false);
    const fill = new Uint8Array(holes.comps.length);
    for (let i = 1; i < holes.comps.length; i++) {
      const c = holes.comps[i];
      if (!c.border && c.area <= personArea * HOLE_FILL_MAX_RATIO) fill[i] = 1;
    }
    for (let k = 0; k < n; k++) if (fill[holes.lab[k]]) a[k] = 255;
  }
  // Suavizado 3x3 [1/4, 1/2, 1/4] (borde reflejado, como cv2.GaussianBlur)
  const refl = (i, L) => (i < 0 ? -i : i >= L ? 2 * L - i - 2 : i);
  const t = new Float32Array(n), o = new Uint8Array(n);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
    t[y * W + x] = 0.25 * a[y * W + refl(x - 1, W)] + 0.5 * a[y * W + x] + 0.25 * a[y * W + refl(x + 1, W)];
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
    o[y * W + x] = Math.round(0.25 * t[refl(y - 1, H) * W + x] + 0.5 * t[y * W + x] + 0.25 * t[refl(y + 1, H) * W + x]);
  return o;
}

// ── YuNet (decodificación equivalente a cv2.FaceDetectorYN) ─────────────────
async function detectFace(rgba, W, H) {
  const PW = Math.ceil(W / 32) * 32, PH = Math.ceil(H / 32) * 32, n = PW * PH;
  const t = new Float32Array(3 * n);   // BGR, 0-255, relleno con ceros abajo/derecha
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const p = (y * W + x) * 4, k = y * PW + x;
    t[k] = rgba[p + 2]; t[n + k] = rgba[p + 1]; t[2 * n + k] = rgba[p];
  }
  const out = await sessions.yunet.run({ input: new ort.Tensor("float32", t, [1, 3, PH, PW]) });
  let best = null;
  for (const s of [8, 16, 32]) {
    const cols = PW / s, cls = out[`cls_${s}`].data, obj = out[`obj_${s}`].data, bb = out[`bbox_${s}`].data;
    for (let i = 0; i < cls.length; i++) {
      const sc = Math.sqrt(Math.min(1, Math.max(0, cls[i])) * Math.min(1, Math.max(0, obj[i])));
      if (!best || sc > best.score) {
        const r = Math.floor(i / cols), c = i % cols;
        const cx = (c + bb[i * 4]) * s, cy = (r + bb[i * 4 + 1]) * s;
        const w = Math.exp(bb[i * 4 + 2]) * s, h = Math.exp(bb[i * 4 + 3]) * s;
        best = { score: sc, x: Math.trunc(cx - w / 2), y: Math.trunc(cy - h / 2), w: Math.trunc(w), h: Math.trunc(h) };
      }
    }
  }
  return best && best.score >= YUNET_SCORE_MIN ? best : null;
}

// ── Recorte (mismo algoritmo que crop_to_fotocheck del escritorio) ──────────
function personBBox(rgba, W, H) {
  let minX = W, minY = H, maxX = -1, maxY = -1;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const p = (y * W + x) * 4;
    if (rgba[p] < 240 || rgba[p + 1] < 240 || rgba[p + 2] < 240) {
      if (x < minX) minX = x; if (x > maxX) maxX = x; if (y < minY) minY = y; if (y > maxY) maxY = y;
    }
  }
  return maxX < 0 ? null : { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 };
}

function cropRect(W, H, face, pb, RATIO) {
  const { x: fx, y: fy, w: fw, h: fh } = face;
  const fxc = fx + fw / 2, fyc = fy + fh / 2;
  let modo = "normal", ch, cw;
  const headH = pb ? fy + fh - pb.y : null;
  if (headH !== null && headH > 0) { ch = headH / HEAD_HEIGHT_RATIO; modo = "por altura de cabeza"; }
  else ch = fh / FACE_HEIGHT_RATIO_YUNET;
  cw = ch * RATIO;
  if (pb) {
    const pwm = pb.w * (1 + 2 * SIDE_MARGIN);
    if (pwm < cw) { cw = pwm; ch = cw / RATIO; modo = "ajustado al ancho de la persona"; }
  }
  if (ch > H) { ch = H; cw = ch * RATIO; }
  if (cw > W) { cw = W; ch = cw / RATIO; }
  let top;
  if (pb) top = Math.min(pb.y - HEAD_SPACE_RATIO * ch, fy - HEAD_TOP_MARGIN * ch);
  else top = fyc - FACE_CENTER_Y_RATIO * ch;
  const need = fy + fh + CHIN_MARGIN * fh;
  if (top + ch < need) { ch = need - top; cw = ch * RATIO; modo += " + ampliado"; }
  if (pb) {
    const pBottom = pb.y + pb.h;
    if (pBottom < H - 1) {
      const disp = pBottom - Math.max(0, top);
      if (disp > 0 && disp < ch) { ch = disp; cw = ch * RATIO; modo += " + sin banda inferior"; }
    }
  }
  if (ch > H) { ch = H; cw = ch * RATIO; }
  if (cw > W) { cw = W; ch = cw / RATIO; }
  let left = fxc - cw / 2;
  top = Math.max(0, Math.min(top, H - ch));
  left = Math.max(0, Math.min(left, W - cw));
  // Formatos más anchos que la foto (p. ej. horizontal desde una foto vertical):
  // si limitar el ancho dejó el mentón fuera, se prioriza la cabeza completa y se
  // extiende con blanco a los costados (el fondo ya es blanco, no se nota).
  if (top + ch < need && ch < H) {
    ch = Math.min(H, need - top);
    top = Math.max(0, Math.min(top, H - ch));
    if (top + ch < need) { top = Math.max(0, need - ch); }
    cw = ch * RATIO;
    left = fxc - cw / 2;
    if (cw <= W) left = Math.max(0, Math.min(left, W - cw));
    modo += " + relleno lateral";
  }
  return { x0: Math.round(left), y0: Math.round(top), x1: Math.round(left + cw), y1: Math.round(top + ch), modo };
}

function fallbackRect(W, H, RATIO) {
  if (W / H > RATIO) { const nw = Math.trunc(H * RATIO), x0 = Math.trunc((W - nw) / 2); return { x0, y0: 0, x1: x0 + nw, y1: H, modo: "sin rostro (recorte centrado)" }; }
  const nh = Math.trunc(W / RATIO), y0 = Math.trunc((H - nh) / 2);
  return { x0: 0, y0, x1: W, y1: y0 + nh, modo: "sin rostro (recorte centrado)" };
}

// ── Pipeline completo ───────────────────────────────────────────────────────
async function processPhoto(bitmap, outW, outH) {
  const t0 = performance.now();
  let W = bitmap.width, H = bitmap.height;
  const k = Math.min(1, MAX_LADO / Math.max(W, H));
  W = Math.round(W * k); H = Math.round(H * k);
  const src = drawTo(bitmap, W, H);
  const rgba = pixels(src);
  const n = W * H;

  post("stage", { text: "Separando a la persona del fondo…" });
  const mh = await segment(sessions.seg, src, 1024, [0, 0, 0], [1, 1, 1], 255);

  // La máscara de ormbg ya trae bordes y cabello finos: el refinado por color
  // (pensado para u2net) le metía bordes dentados. Solo se limpia.
  const alpha = cleanMask(mh, W, H);
  // Borde más firme: lo casi transparente (halo de fondo) se va, lo casi opaco queda entero
  for (let q = 0; q < n; q++) alpha[q] = Math.max(0, Math.min(255, (alpha[q] - 60) * 255 / 140));

  // Color de la persona cerca del borde (promedio local de píxeles seguros):
  // en el borde semitransparente se usa ese color en vez del fondo que se cuela
  const R = Math.max(3, Math.round(Math.max(W, H) / 80));
  const fg = new Float32Array(n), cr = new Float32Array(n), cgr = new Float32Array(n), cb = new Float32Array(n);
  for (let q = 0, p = 0; q < n; q++, p += 4) if (alpha[q] >= 240) { fg[q] = 1; cr[q] = rgba[p]; cgr[q] = rgba[p + 1]; cb[q] = rgba[p + 2]; }
  const nF = boxFilter(fg, W, H, R), sR = boxFilter(cr, W, H, R), sG = boxFilter(cgr, W, H, R), sB = boxFilter(cb, W, H, R);

  const comp = new Uint8ClampedArray(n * 4);
  for (let q = 0, p = 0; q < n; q++, p += 4) {
    const a = alpha[q] / 255;
    let r = rgba[p], g = rgba[p + 1], b = rgba[p + 2];
    if (alpha[q] > 10 && alpha[q] < 240 && nF[q] > 1e-3) {
      const k = nF[q];
      r = (sR[q] / k) * (1 - a) + r * a; g = (sG[q] / k) * (1 - a) + g * a; b = (sB[q] / k) * (1 - a) + b * a;
    }
    comp[p] = r * a + 255 * (1 - a);
    comp[p + 1] = g * a + 255 * (1 - a);
    comp[p + 2] = b * a + 255 * (1 - a);
    comp[p + 3] = 255;
  }

  post("stage", { text: "Encuadrando el rostro…" });
  const face = await detectFace(comp, W, H);
  const full = new OffscreenCanvas(W, H);
  full.getContext("2d").putImageData(new ImageData(comp, W, H), 0, 0);
  last = { full, W, H, face, pb: face ? personBBox(comp, W, H) : null,
           ms: Math.round(performance.now() - t0) };
  return render(outW, outH);
}

function sizeOK(w, h) {
  return Number.isInteger(w) && Number.isInteger(h) &&
         w >= MIN_LADO_SALIDA && h >= MIN_LADO_SALIDA && w <= MAX_LADO_SALIDA && h <= MAX_LADO_SALIDA;
}

// Recorta la última foto al tamaño pedido (rápido: no vuelve a quitar el fondo)
async function render(outW, outH) {
  if (!sizeOK(outW, outH)) throw new Error(`tamaño no válido (${outW} × ${outH})`);
  const { full, W, H, face, pb } = last, ratio = outW / outH;
  const rect = face ? cropRect(W, H, face, pb, ratio) : fallbackRect(W, H, ratio);
  const outC = new OffscreenCanvas(outW, outH);
  const g = outC.getContext("2d");
  g.fillStyle = "#fff"; g.fillRect(0, 0, outW, outH);
  g.imageSmoothingEnabled = true; g.imageSmoothingQuality = "high";
  // Se arma el recorte sobre un lienzo blanco: si sale de la foto, queda blanco
  const cw = rect.x1 - rect.x0, ch = rect.y1 - rect.y0;
  const crop = new OffscreenCanvas(cw, ch), cg = crop.getContext("2d");
  cg.fillStyle = "#fff"; cg.fillRect(0, 0, cw, ch);
  cg.drawImage(full, -rect.x0, -rect.y0);
  g.drawImage(crop, 0, 0, cw, ch, 0, 0, outW, outH);
  const blob = await outC.convertToBlob({ type: "image/jpeg", quality: 0.95 });
  return {
    blob,
    info: {
      ms: last.ms,
      rostro: !!face, modo: rect.modo, rect, size: [W, H], face, salida: [outW, outH],
    },
  };
}

self.onmessage = async (e) => {
  const { type } = e.data;
  try {
    if (type === "init") {
      if (!sessions) await loadModels();
      post("ready");
    } else if (type === "process") {
      const { bitmap, w, h } = e.data;
      const res = await processPhoto(bitmap, w || DEFAULT_W, h || DEFAULT_H);
      bitmap.close && bitmap.close();
      post("done", res);
    } else if (type === "resize") {
      if (!last) return;
      post("done", { ...(await render(e.data.w, e.data.h)), resized: true });
    }
  } catch (err) {
    post("error", { message: String(err && err.message ? err.message : err) });
  }
};
