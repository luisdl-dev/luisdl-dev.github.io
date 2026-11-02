(() => {
  const $ = (id) => document.getElementById(id);
  const drop = $("drop"), file = $("file"), empty = $("empty"), result = $("result");
  const busy = $("busy"), fill = $("fill"), stage = $("stage"), meta = $("meta");
  const download = $("download"), again = $("again"), firstLoad = $("first-load");

  const supported = typeof OffscreenCanvas !== "undefined" && typeof Worker !== "undefined" && typeof createImageBitmap !== "undefined";
  if (!supported) {
    setMeta("Tu navegador no es compatible. Usa Chrome, Edge, Firefox o Safari actualizados.", "err");
    drop.setAttribute("aria-disabled", "true");
    return;
  }

  const worker = new Worker("js/worker.js");
  let ready = false, working = false, pending = null, lastUrl = null;

  function setMeta(text, cls) { meta.textContent = text; meta.className = "meta" + (cls ? " " + cls : ""); }
  function showBusy(text, pct) {
    busy.hidden = false; stage.textContent = text;
    if (pct == null) { fill.classList.add("indet"); fill.style.width = ""; }
    else { fill.classList.remove("indet"); fill.style.width = pct + "%"; }
  }
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
      const s = (m.info.ms / 1000).toFixed(1).replace(".", ",");
      let t = `Listo en ${s} s`;
      if (!m.info.rostro) t += ". No se detectó un rostro: se recortó al centro";
      else if (m.info.fondoDificil) t += ". Fondo difícil: se combinaron dos modelos";
      setMeta(t, m.info.rostro ? "ok" : "");
    } else if (m.type === "error") {
      working = false; busy.hidden = true;
      setMeta(`No se pudo procesar: ${m.message}. Prueba con otra foto en JPG o PNG.`, "err");
    }
  };
  worker.onerror = () => {
    working = false; busy.hidden = true;
    setMeta("El procesador se detuvo. Recarga la página e inténtalo de nuevo.", "err");
  };

  async function accept(f) {
    if (!f || working) return;
    if (!/^image\/(jpeg|png|webp)$/.test(f.type)) { setMeta("Ese archivo no es una foto JPG, PNG o WebP.", "err"); return; }
    let bitmap;
    try { bitmap = await createImageBitmap(f, { imageOrientation: "from-image" }); }
    catch (_) { setMeta("No se pudo abrir la foto. Prueba con otro archivo.", "err"); return; }
    const stem = f.name.replace(/\.[^.]+$/, "") || "foto";
    download.setAttribute("download", stem + ".jpg");
    working = true; setMeta("Procesando…");
    const job = { bitmap };
    if (!ready) { pending = job; showBusy("Preparando…", 0); worker.postMessage({ type: "init" }); }
    else run(job);
  }
  function run(job) { showBusy("Procesando…", null); worker.postMessage({ type: "process", bitmap: job.bitmap }, [job.bitmap]); }

  function reset() {
    result.hidden = true; empty.hidden = false; drop.classList.remove("has-result");
    download.hidden = true; again.hidden = true; setMeta("Sale en 480 × 640, formato JPG");
    file.value = ""; file.click();
  }

  drop.addEventListener("click", () => { if (!drop.classList.contains("has-result") && !working) file.click(); });
  drop.addEventListener("keydown", (e) => { if ((e.key === "Enter" || e.key === " ") && !drop.classList.contains("has-result")) { e.preventDefault(); file.click(); } });
  file.addEventListener("change", () => accept(file.files[0]));
  again.addEventListener("click", reset);
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("drag"); }));
  drop.addEventListener("drop", (e) => accept(e.dataTransfer.files[0]));
  window.addEventListener("paste", (e) => { const it = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/")); if (it) accept(it.getAsFile()); });
})();
