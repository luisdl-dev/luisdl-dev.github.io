(() => {
  const $ = (id) => document.getElementById(id);
  const drop = $("drop"), file = $("file"), empty = $("empty"), result = $("result");
  const busy = $("busy"), fill = $("fill"), stage = $("stage"), meta = $("meta");
  const download = $("download"), again = $("again"), firstLoad = $("first-load");
  const straight = $("straight"), notes = $("notes");
  const inW = $("out-w"), inH = $("out-h"), resetSize = $("reset-size"), pasteKey = $("paste-key");
  // Antes y después + ajustes
  const editor = $("editor"), before = $("before"), cropBox = $("crop-box"), after = $("after"), afterCap = $("after-cap");
  const encHint = $("enc-hint"), colorHint = $("color-hint"), download2 = $("download2");
  const radios = [...document.querySelectorAll('input[name="encuadre"]')];
  const SLIDERS = ["brillo", "contraste", "saturacion", "temperatura"];

  const DEF_W = 480, DEF_H = 640, MIN = 64, MAX = 4000;
  const NEUTRO = { brillo: 0, contraste: 0, saturacion: 0, temperatura: 0 };

  // Atajo de pegar según el sistema
  if (/Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent)) pasteKey.textContent = "⌘ + V";

  const supported = typeof OffscreenCanvas !== "undefined" && typeof Worker !== "undefined" && typeof createImageBitmap !== "undefined";
  if (!supported) {
    setMeta("Tu navegador no es compatible. Usa Chrome, Edge, Firefox o Safari actualizados.", "err");
    drop.setAttribute("aria-disabled", "true");
    return;
  }

  // ── Tamaño de salida ────────────────────────────────────────────────────
  const store = {
    get() { try { return JSON.parse(localStorage.getItem("fotocheck-tamano")); } catch (_) { return null; } },
    set(v) { try { localStorage.setItem("fotocheck-tamano", JSON.stringify(v)); } catch (_) {} },
  };
  const saved = store.get();
  if (saved && valid(saved.w, saved.h)) { inW.value = saved.w; inH.value = saved.h; }

  function valid(w, h) { return Number.isInteger(w) && Number.isInteger(h) && w >= MIN && h >= MIN && w <= MAX && h <= MAX; }
  function size() { return { w: Number(inW.value), h: Number(inH.value) }; }
  const sizeText = (w, h) => `${w} × ${h}`;

  function applySize() {
    const { w, h } = size();
    const ok = valid(w, h);
    inW.setAttribute("aria-invalid", String(!ok)); inH.setAttribute("aria-invalid", String(!ok));
    if (!ok) { setMeta(`El ancho y el alto deben ser números enteros entre ${MIN} y ${MAX} px.`, "err"); return false; }
    // La ventana de la tarjeta toma la forma del tamaño elegido (limitada para que no se deforme el diseño)
    const r = Math.min(2, Math.max(0.5, w / h));
    drop.style.aspectRatio = `${r}`;
    resetSize.hidden = w === DEF_W && h === DEF_H;
    store.set({ w, h });
    return true;
  }

  let resizeTimer = null;
  function onSizeInput() {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (!applySize()) return;
      const { w, h } = size();
      if (hasResult() && !working) pedirRender("Ajustando tamaño…");
      else if (!working) setMeta(`Sale en ${sizeText(w, h)} px, formato JPG`);
    }, 350);
  }
  inW.addEventListener("input", onSizeInput); inH.addEventListener("input", onSizeInput);
  resetSize.addEventListener("click", () => { inW.value = DEF_W; inH.value = DEF_H; onSizeInput(); });

  // ── Motor ───────────────────────────────────────────────────────────────
  const VERSION = "2.1", VERSION_FECHA = "3 nov 2026";   // subir en cada cambio publicado
  document.getElementById("ver").textContent = `${VERSION} (${VERSION_FECHA})`;
  const worker = new Worker(`js/worker.js?v=${VERSION}`);
  let ready = false, initSent = false, working = false, pending = null, lastProgress = null;
  let rendering = false, renderQueued = null;   // ajustes de tamaño, encuadre o color
  // Resultado actual: foto normal y, si la cabeza estaba inclinada, la enderezada
  let urls = { normal: null, corregida: null }, verCorregida = false, stem = "foto-fotocheck", grados = 0;
  let info = null, beforeUrl = null, proc = null;
  const opts = { encuadre: "recomendado", color: { ...NEUTRO } };

  function setMeta(text, cls) { meta.textContent = text; meta.className = "meta" + (cls ? " " + cls : ""); }
  function showBusy(text, pct) {
    busy.hidden = false; stage.textContent = text;
    if (pct == null) { fill.classList.add("indet"); fill.style.width = ""; }
    else { fill.classList.remove("indet"); fill.style.width = pct + "%"; }
  }
  const hasResult = () => drop.classList.contains("has-result");
  const mb = (b) => (b / 1e6).toFixed(0);
  const progreso = (m) => [`${mb(m.loaded)} de ${mb(m.total)} MB`, Math.min(100, Math.round((m.loaded / m.total) * 100))];

  // La IA se descarga apenas se abre la página (salvo con ahorro de datos)
  function iniciarCarga() {
    if (initSent) return;
    initSent = true;
    worker.postMessage({ type: "init" });
  }

  function mostrar() {
    const corr = verCorregida && urls.corregida;
    const url = corr ? urls.corregida : urls.normal;
    result.src = url; after.src = url;
    download.href = url; download2.href = url;
    const nombre = stem + (corr ? "_corregida" : "") + ".jpg";
    download.setAttribute("download", nombre); download2.setAttribute("download", nombre);
    straight.textContent = corr ? "Ver original" : `Ver enderezada (${grados}°)`;
    straight.setAttribute("aria-pressed", String(!!corr));
    if (!info) return;
    const [w, h] = info.salida;
    afterCap.textContent = `Después · ${sizeText(w, h)} px${corr ? ", enderezada" : ""}`;
    // Recuadro sobre la foto original: la parte que se usó (la enderezada está girada, no coincide)
    cropBox.hidden = !!corr;
    if (!corr) {
      const [W, H] = info.procesada, [x0, y0, x1, y1] = info.rect;
      Object.assign(cropBox.style, { left: `${x0 / W * 100}%`, top: `${y0 / H * 100}%`, width: `${(x1 - x0) / W * 100}%`, height: `${(y1 - y0) / H * 100}%` });
    }
  }
  straight.addEventListener("click", () => { verCorregida = !verCorregida; mostrar(); });

  function limpiarResultado() {
    for (const k in urls) if (urls[k]) URL.revokeObjectURL(urls[k]);
    urls = { normal: null, corregida: null };
    if (beforeUrl) { URL.revokeObjectURL(beforeUrl); beforeUrl = null; }
    info = null;
    result.hidden = true; empty.hidden = false; drop.classList.remove("has-result");
    download.hidden = true; again.hidden = true; straight.hidden = true; notes.hidden = true;
    editor.hidden = true;
  }

  function textoListo(w, h) {
    let t = `Listo en ${(proc.ms / 1000).toFixed(1).replace(".", ",")} s, ${sizeText(w, h)} px`;
    if (!proc.rostro) t += ". No se detectó un rostro: se recortó al centro";
    else t += `. Encuadre de ${proc.perfil}`;
    return t;
  }

  function aplicarResultado(m) {
    info = m.info;
    window.__ultimo = info;   // datos del último resultado (útil para pruebas)
    for (const k in urls) if (urls[k]) URL.revokeObjectURL(urls[k]);
    urls = { normal: URL.createObjectURL(m.blob), corregida: m.blobCorregida ? URL.createObjectURL(m.blobCorregida) : null };
    if (!m.rerender || !urls.corregida) verCorregida = false;
    grados = info.corregida ? Math.abs(Math.round(info.corregida.angulo)) : 0;
    straight.hidden = !urls.corregida;
    mostrar();
    result.hidden = false; empty.hidden = true;
    drop.classList.add("has-result");
    download.hidden = false; again.hidden = false;
    const [w, h] = info.salida;
    result.alt = `Foto procesada de ${w} por ${h} píxeles`;
    setMeta(textoListo(w, h), proc.rostro ? "ok" : "");
    // Avisos de la versión de escritorio (revisar_v2.txt), salvo el de sin rostro que ya va arriba
    const avisos = info.avisos.filter((a) => !a.startsWith("sin rostro"));
    notes.hidden = !avisos.length;
    notes.textContent = avisos.length ? `Conviene revisar: ${avisos.join(", ")}.` : "";
    actualizarEncuadres();
  }

  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === "progress") {
      lastProgress = m;
      const [txt, pct] = progreso(m);
      if (working) showBusy(`Descargando la IA: ${txt}`, pct);
      else firstLoad.textContent = `Descargando la IA en segundo plano: ${txt}. Puedes ir eligiendo tu foto.`;
    } else if (m.type === "ready") {
      ready = true; firstLoad.hidden = true;
      if (pending) { const p = pending; pending = null; run(p); }
    } else if (m.type === "stage") {
      if (working) showBusy(m.text, null);
      else firstLoad.textContent = m.text;
    } else if (m.type === "done" && m.rerender) {
      rendering = false; busy.hidden = true;
      aplicarResultado(m);
      if (renderQueued !== null) { const t = renderQueued; renderQueued = null; pedirRender(t); }
    } else if (m.type === "done") {
      working = false; busy.hidden = true;
      proc = { ms: m.info.ms, rostro: m.info.rostro, perfil: m.info.perfil };
      // Cada foto nueva empieza con el flujo por defecto (el de la versión de escritorio)
      opts.encuadre = "recomendado"; opts.color = { ...NEUTRO };
      sincronizarControles(); colorHint.textContent = "";
      aplicarResultado(m);
      before.src = beforeUrl;
      editor.hidden = false;
      const quieto = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      editor.scrollIntoView({ behavior: quieto ? "auto" : "smooth", block: "start" });
      // Si el tamaño cambió mientras se procesaba, ajustar al tamaño actual
      const cur = size(), [w, h] = m.info.salida;
      if (valid(cur.w, cur.h) && (cur.w !== w || cur.h !== h)) onSizeInput();
    } else if (m.type === "error") {
      busy.hidden = true;
      if (working) {
        working = false; pending = null;
        if (!ready) initSent = false;   // se reintenta la descarga con la próxima foto
        setMeta(`No se pudo procesar: ${m.message}. Prueba con otra foto en JPG o PNG.`, "err");
      } else if (rendering) {
        rendering = false; renderQueued = null;
        setMeta(`No se pudo ajustar la foto: ${m.message}.`, "err");
      } else {
        initSent = false;
        firstLoad.textContent = "No se pudo descargar la IA ahora. Se intentará de nuevo cuando subas una foto.";
      }
    }
  };
  worker.onerror = () => {
    working = false; rendering = false; busy.hidden = true;
    setMeta("El procesador se detuvo. Recarga la página e inténtalo de nuevo.", "err");
  };

  async function accept(f, origen) {
    if (!f || working || rendering) return;
    if (!/^image\/(jpeg|png|webp)$/.test(f.type)) { setMeta("Ese archivo no es una foto JPG, PNG o WebP.", "err"); return; }
    if (!applySize()) return;
    let bitmap;
    try { bitmap = await createImageBitmap(f, { imageOrientation: "from-image" }); }
    catch (_) { setMeta("No se pudo abrir la foto. Prueba con otro archivo.", "err"); return; }
    // Una imagen pegada desde el portapapeles no trae nombre real
    stem = origen === "pegada" ? "foto-fotocheck" : (f.name.replace(/\.[^.]+$/, "") || "foto-fotocheck");
    if (beforeUrl) URL.revokeObjectURL(beforeUrl);
    beforeUrl = URL.createObjectURL(f);
    working = true; setMeta(origen === "pegada" ? "Foto pegada. Procesando…" : "Procesando…");
    const job = { bitmap };
    if (!ready) {
      pending = job;
      if (lastProgress) { const [txt, pct] = progreso(lastProgress); showBusy(`Descargando la IA: ${txt}`, pct); }
      else showBusy("Preparando…", 0);
      iniciarCarga();
    } else run(job);
  }
  function run(job) {
    const { w, h } = size();
    showBusy("Procesando…", null);
    worker.postMessage({ type: "process", bitmap: job.bitmap, w, h }, [job.bitmap]);
  }

  /** Vuelve a componer la foto con el tamaño, encuadre y color elegidos (sin repetir la IA). */
  function pedirRender(texto) {
    if (!hasResult() || working) return;
    if (rendering) { renderQueued = texto || ""; return; }
    const { w, h } = size();
    if (!valid(w, h)) return;
    rendering = true;
    if (texto) showBusy(texto, null);
    worker.postMessage({ type: "render", w, h, encuadre: opts.encuadre, color: { ...opts.color } });
  }

  // ── Encuadre ────────────────────────────────────────────────────────────
  function actualizarEncuadres() {
    const enc = info.encuadres;
    for (const r of radios) r.disabled = !enc[r.value];
    if (!info.rostro) encHint.textContent = "No se detectó un rostro, así que el recorte es centrado.";
    else if (!enc.cuerpo && !enc["mas-cuerpo"]) encHint.textContent = "La foto no muestra más cuerpo del que ya se ve.";
    else if (!enc["mas-cuerpo"]) encHint.textContent = "La foto no da para «Más cuerpo»: ya se usa todo lo que muestra.";
    else encHint.textContent = "";
  }
  for (const r of radios) r.addEventListener("change", () => { if (r.checked) { opts.encuadre = r.value; pedirRender(null); } });

  // ── Color ───────────────────────────────────────────────────────────────
  const signo = (v) => (v > 0 ? `+${v}` : `${v}`);
  function sincronizarControles() {
    for (const r of radios) r.checked = r.value === opts.encuadre;
    for (const k of SLIDERS) { $(`c-${k}`).value = opts.color[k]; $(`o-${k}`).textContent = signo(opts.color[k]); }
  }
  let colorTimer = null;
  for (const k of SLIDERS) {
    const el = $(`c-${k}`);
    el.addEventListener("input", () => {
      opts.color[k] = Number(el.value); $(`o-${k}`).textContent = signo(opts.color[k]);
      colorHint.textContent = "";
      clearTimeout(colorTimer); colorTimer = setTimeout(() => pedirRender(null), 60);
    });
  }
  $("c-auto").addEventListener("click", () => {
    if (!info) return;
    opts.color = { ...info.colorAuto };
    sincronizarControles();
    const nada = SLIDERS.every((k) => !opts.color[k]);
    colorHint.textContent = nada ? "El color ya estaba equilibrado: no hizo falta cambiarlo." : "Ajuste automático aplicado. Puedes afinarlo con los controles.";
    pedirRender(null);
  });
  $("c-reset").addEventListener("click", () => {
    opts.color = { ...NEUTRO }; sincronizarControles(); colorHint.textContent = ""; pedirRender(null);
  });

  function reset() {
    limpiarResultado();
    const { w, h } = size(); setMeta(`Sale en ${sizeText(w, h)} px, formato JPG`);
    file.value = ""; file.click();
  }

  drop.addEventListener("click", () => { if (!hasResult() && !working) file.click(); });
  drop.addEventListener("keydown", (e) => { if ((e.key === "Enter" || e.key === " ") && !hasResult()) { e.preventDefault(); file.click(); } });
  file.addEventListener("change", () => accept(file.files[0], "archivo"));
  again.addEventListener("click", reset);
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("drag"); }));
  drop.addEventListener("drop", (e) => accept(e.dataTransfer.files[0], "archivo"));

  // Pegar con Ctrl + V: captura de pantalla, imagen copiada del navegador o archivo copiado
  window.addEventListener("paste", (e) => {
    const cd = e.clipboardData; if (!cd) return;
    const f = [...(cd.files || [])].find((x) => x.type.startsWith("image/")) ||
              [...(cd.items || [])].find((i) => i.kind === "file" && i.type.startsWith("image/"))?.getAsFile();
    if (!f) return;                       // texto normal: que siga su curso (p. ej. en los campos de tamaño)
    e.preventDefault();
    if (working || rendering) { setMeta("Espera a que termine la foto actual para pegar otra."); return; }
    if (hasResult()) limpiarResultado();
    accept(f, "pegada");
  });

  applySize();
  const s0 = size(); setMeta(`Sale en ${sizeText(s0.w, s0.h)} px, formato JPG`);
  sincronizarControles();
  if (!(navigator.connection && navigator.connection.saveData)) iniciarCarga();
})();
