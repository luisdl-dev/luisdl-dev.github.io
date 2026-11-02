(() => {
  const $ = (id) => document.getElementById(id);
  const drop = $("drop"), file = $("file"), empty = $("empty"), result = $("result");
  const busy = $("busy"), fill = $("fill"), stage = $("stage"), meta = $("meta");
  const download = $("download"), again = $("again"), firstLoad = $("first-load");
  const inW = $("out-w"), inH = $("out-h"), resetSize = $("reset-size"), pasteKey = $("paste-key");

  const DEF_W = 480, DEF_H = 640, MIN = 64, MAX = 4000;

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
      if (hasResult() && !working) { working = true; showBusy("Ajustando tamaño…", null); worker.postMessage({ type: "resize", w, h }); }
      else if (!working) setMeta(`Sale en ${sizeText(w, h)} px, formato JPG`);
    }, 350);
  }
  inW.addEventListener("input", onSizeInput); inH.addEventListener("input", onSizeInput);
  resetSize.addEventListener("click", () => { inW.value = DEF_W; inH.value = DEF_H; onSizeInput(); });

  // ── Motor ───────────────────────────────────────────────────────────────
  const worker = new Worker("js/worker.js");
  let ready = false, working = false, pending = null, lastUrl = null;

  function setMeta(text, cls) { meta.textContent = text; meta.className = "meta" + (cls ? " " + cls : ""); }
  function showBusy(text, pct) {
    busy.hidden = false; stage.textContent = text;
    if (pct == null) { fill.classList.add("indet"); fill.style.width = ""; }
    else { fill.classList.remove("indet"); fill.style.width = pct + "%"; }
  }
  const hasResult = () => drop.classList.contains("has-result");
  const mb = (b) => (b / 1e6).toFixed(0);

  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === "progress") {
      const pct = Math.min(100, Math.round((m.loaded / m.total) * 100));
      showBusy(`Descargando modelos: ${mb(m.loaded)} de ${mb(m.total)} MB`, pct);
    } else if (m.type === "ready") {
      ready = true; firstLoad.hidden = true;
      if (pending) { const p = pending; pending = null; run(p); }
    } else if (m.type === "stage") {
      showBusy(m.text, null);
    } else if (m.type === "done") {
      working = false; busy.hidden = true;
      window.__ultimo = m.info;   // datos del último resultado (útil para pruebas)
      if (lastUrl) URL.revokeObjectURL(lastUrl);
      lastUrl = URL.createObjectURL(m.blob);
      result.src = lastUrl; result.hidden = false; empty.hidden = true;
      drop.classList.add("has-result");
      download.href = lastUrl; download.hidden = false; again.hidden = false;
      const [w, h] = m.info.salida;
      result.alt = `Foto procesada de ${w} por ${h} píxeles`;
      let t;
      if (m.resized) t = `Listo en ${sizeText(w, h)} px`;
      else {
        t = `Listo en ${(m.info.ms / 1000).toFixed(1).replace(".", ",")} s, ${sizeText(w, h)} px`;
        if (!m.info.rostro) t += ". No se detectó un rostro: se recortó al centro";
        else if (m.info.fondoDificil) t += ". Fondo difícil: se combinaron dos modelos";
      }
      setMeta(t, m.info.rostro ? "ok" : "");
      // Si el tamaño cambió mientras se procesaba, ajustar al tamaño actual
      const cur = size();
      if (valid(cur.w, cur.h) && (cur.w !== w || cur.h !== h)) onSizeInput();
    } else if (m.type === "error") {
      working = false; busy.hidden = true;
      setMeta(`No se pudo procesar: ${m.message}. Prueba con otra foto en JPG o PNG.`, "err");
    }
  };
  worker.onerror = () => {
    working = false; busy.hidden = true;
    setMeta("El procesador se detuvo. Recarga la página e inténtalo de nuevo.", "err");
  };

  async function accept(f, origen) {
    if (!f || working) return;
    if (!/^image\/(jpeg|png|webp)$/.test(f.type)) { setMeta("Ese archivo no es una foto JPG, PNG o WebP.", "err"); return; }
    if (!applySize()) return;
    let bitmap;
    try { bitmap = await createImageBitmap(f, { imageOrientation: "from-image" }); }
    catch (_) { setMeta("No se pudo abrir la foto. Prueba con otro archivo.", "err"); return; }
    // Una imagen pegada desde el portapapeles no trae nombre real
    const stem = origen === "pegada" ? "foto-fotocheck" : (f.name.replace(/\.[^.]+$/, "") || "foto-fotocheck");
    download.setAttribute("download", stem + ".jpg");
    working = true; setMeta(origen === "pegada" ? "Foto pegada. Procesando…" : "Procesando…");
    const job = { bitmap };
    if (!ready) { pending = job; showBusy("Preparando…", 0); worker.postMessage({ type: "init" }); }
    else run(job);
  }
  function run(job) {
    const { w, h } = size();
    showBusy("Procesando…", null);
    worker.postMessage({ type: "process", bitmap: job.bitmap, w, h }, [job.bitmap]);
  }

  function reset() {
    result.hidden = true; empty.hidden = false; drop.classList.remove("has-result");
    download.hidden = true; again.hidden = true;
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
    if (working) { setMeta("Espera a que termine la foto actual para pegar otra."); return; }
    if (hasResult()) { result.hidden = true; empty.hidden = false; drop.classList.remove("has-result"); download.hidden = true; again.hidden = true; }
    accept(f, "pegada");
  });

  applySize();
  const s0 = size(); setMeta(`Sale en ${sizeText(s0.w, s0.h)} px, formato JPG`);
})();
