/* Dashboard de Ottoman: dispositivos, chat, lectura emocional y entrenamiento. */
const $ = (id) => document.getElementById(id);
const COLOR = { ALEGRIA: "#d9e24a", TRISTEZA: "#60a5fa", ENOJO: "#f87171", ANSIEDAD: "#c084fc", NEUTRO: "#94a3b8" };
const NOMBRE = { ALEGRIA: "Alegría", TRISTEZA: "Tristeza", ENOJO: "Enojo", ANSIEDAD: "Ansiedad", NEUTRO: "Neutro" };
const ORDEN = ["ALEGRIA", "TRISTEZA", "ENOJO", "ANSIEDAD", "NEUTRO"];
const RASGOS_VISIBLES = ["sonrisa", "ceño", "cejas_internas", "boca_abajo", "ojos_abiertos", "cejas_arriba", "labios_apretados", "boca_abierta"];

const disp = {
  camara: { activa: false, deseada: true, stream: null, motivo: "buscando…" },
  mic: { activa: false, deseada: true, stream: null, motivo: "buscando…" },
};
const st = { ocupado: false, grabando: false, destinoGrabacion: "chat", trozos: [], tInicio: 0, actitudOjos: "reposo",
             captura: null, mod: "cara", etiqueta: "NEUTRO", info: null, hablando: false };
let audioCtx = null, procesador = null, analizadorSalida = null;

const pct = (v) => Math.round(v * 100) + "%";
async function api(ruta, opciones) {
  const r = await fetch(ruta, opciones);
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.detail || `Error ${r.status}`);
  return d;
}
const cuerpo = (modo) => fetch("/api/cuerpo", { method: "POST", headers: { "Content-Type": "application/json" },
                                              body: JSON.stringify({ modo }) }).catch(() => {});

/* ===================== Dispositivos ===================== */
function pintarChips() {
  for (const [tipo, id, nombre] of [["camara", "chip-camara", "Cámara"], ["mic", "chip-mic", "Micrófono"]]) {
    const d = disp[tipo], chip = $(id);
    chip.className = "chip " + (d.activa ? "ok" : d.deseada ? "mal" : "aviso");
    chip.querySelector(".txt").textContent = `${nombre}: ${d.activa ? "activa" : d.deseada ? d.motivo : "apagada"}`;
  }
  $("camara-vacia").hidden = disp.camara.activa;
  $("btn-hablar").disabled = !disp.mic.activa || (st.ocupado && !st.grabando);
  if (!disp.camara.activa) { $("encuadre").textContent = "Sin cámara"; limpiarCapa(); }
  $("texto").placeholder = disp.mic.activa ? "Escribí, o tocá el micrófono para hablar…" : "Escribí lo que quieras contarle…";
}

function detener(tipo) {
  const d = disp[tipo];
  if (d.stream) d.stream.getTracks().forEach((t) => t.stop());
  d.stream = null; d.activa = false;
  if (tipo === "camara") $("video").srcObject = null;
  if (tipo === "mic" && procesador) { procesador.disconnect(); procesador = null; }
}

async function conectar(tipo) {
  const d = disp[tipo];
  detener(tipo);
  if (!navigator.mediaDevices?.getUserMedia) { d.motivo = "el navegador no lo permite"; return; }
  try {
    const pedido = tipo === "camara" ? { video: { width: { ideal: 640 }, height: { ideal: 480 } } }
                                     : { audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } };
    d.stream = await navigator.mediaDevices.getUserMedia(pedido);
    d.activa = true; d.motivo = "";
    d.stream.getTracks().forEach((t) => (t.onended = () => { detener(tipo); d.motivo = "se desconectó"; pintarChips(); }));
    if (tipo === "camara") $("video").srcObject = d.stream; else prepararMic(d.stream);
  } catch (e) {
    d.motivo = e.name === "NotAllowedError" ? "permiso denegado"
             : e.name === "NotFoundError" || e.name === "OverconstrainedError" ? "no detectada"
             : e.name === "NotReadableError" ? "en uso por otro programa" : "no disponible";
  }
}

async function buscarDispositivos() {
  $("btn-buscar").disabled = true; $("btn-buscar").textContent = "Buscando…";
  for (const tipo of ["camara", "mic"]) if (disp[tipo].deseada && !disp[tipo].activa) await conectar(tipo);
  pintarChips();
  $("btn-buscar").disabled = false; $("btn-buscar").textContent = "Buscar dispositivos";
}

async function alternar(tipo) {
  const d = disp[tipo];
  if (d.activa) { d.deseada = false; detener(tipo); }
  else { d.deseada = true; await conectar(tipo); }
  pintarChips();
}

/* ===================== Micrófono ===================== */
function contexto() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === "suspended") audioCtx.resume();
  return audioCtx;
}
function prepararMic(stream) {
  const ctx = contexto();
  const fuente = ctx.createMediaStreamSource(stream);
  procesador = ctx.createScriptProcessor(4096, 1, 1);
  const mudo = ctx.createGain(); mudo.gain.value = 0;
  procesador.onaudioprocess = (ev) => {
    if (!st.grabando) return;
    const x = ev.inputBuffer.getChannelData(0);
    st.trozos.push(new Float32Array(x));
    let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * x[i];
    $("nivel").firstElementChild.style.width = Math.min(100, Math.sqrt(s / x.length) * 500) + "%";
    if ((performance.now() - st.tInicio) / 1000 > 40) terminarGrabacion();
  };
  fuente.connect(procesador); procesador.connect(mudo); mudo.connect(ctx.destination);
}
function aWav(trozos, fsOrigen) {
  const total = trozos.reduce((n, t) => n + t.length, 0);
  const x = new Float32Array(total); let o = 0;
  for (const t of trozos) { x.set(t, o); o += t.length; }
  const fs = 16000, paso = fsOrigen / fs, n = Math.floor(total / paso);
  const buf = new ArrayBuffer(44 + n * 2), v = new DataView(buf);
  const txt = (p, s) => { for (let i = 0; i < s.length; i++) v.setUint8(p + i, s.charCodeAt(i)); };
  txt(0, "RIFF"); v.setUint32(4, 36 + n * 2, true); txt(8, "WAVEfmt "); v.setUint32(16, 16, true);
  v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, fs, true); v.setUint32(28, fs * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true); txt(36, "data"); v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {
    const a = Math.floor(i * paso), b = Math.min(total, Math.floor((i + 1) * paso));
    let s = 0; for (let j = a; j < b; j++) s += x[j];
    const m = Math.max(-1, Math.min(1, s / Math.max(1, b - a)));
    v.setInt16(44 + i * 2, m * 32767, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}
function empezarGrabacion(destino) {
  if (!disp.mic.activa || st.grabando) return;
  contexto(); detenerVoz();
  st.trozos = []; st.grabando = true; st.destinoGrabacion = destino; st.tInicio = performance.now();
  if (destino === "chat") {
    $("btn-hablar").classList.add("grabando"); $("nivel").hidden = false;
    $("estado-robot").textContent = "Te escucha";
    Ojos.fijar("escuchando"); cuerpo("escuchando");
  }
}
function terminarGrabacion() {
  if (!st.grabando) return null;
  st.grabando = false;
  const dur = (performance.now() - st.tInicio) / 1000;
  const wav = aWav(st.trozos, audioCtx.sampleRate);
  $("btn-hablar").classList.remove("grabando"); $("nivel").hidden = true;
  if (st.destinoGrabacion === "chat") {
    if (dur < 0.7) { avisar("Fue muy cortito. Tocá el micrófono, hablá y volvé a tocarlo para enviar."); volverAReposo(); return null; }
    enviarTurno({ wav, dur });
  }
  return { wav, dur };
}

/* ===================== Cámara ===================== */
const lienzo = document.createElement("canvas"); lienzo.width = 320; lienzo.height = 240;
let enVuelo = false;
function limpiarCapa() { const c = $("capa"); c.getContext("2d").clearRect(0, 0, c.width, c.height); }
async function cicloCamara() {
  const video = $("video");
  if (!disp.camara.activa || enVuelo || video.readyState < 2) return;
  enVuelo = true;
  try {
    lienzo.getContext("2d").drawImage(video, 0, 0, 320, 240);
    const blob = await new Promise((r) => lienzo.toBlob(r, "image/jpeg", 0.72));
    let ruta = "/api/cara";
    const cap = st.captura;
    if (cap) ruta += `?etiqueta=${cap.etiqueta}&rafaga=${cap.rafaga}`;
    const r = await api(ruta, { method: "POST", body: blob, headers: { "Content-Type": "image/jpeg" } });
    pintarCara(r);
    if (cap && r.guardada) { cap.n++; $("ent-progreso").textContent = `Capturando… ${cap.n} cuadros`; }
    if (cap && performance.now() > cap.hasta) finCaptura();
  } catch (e) { /* un cuadro perdido no importa */ }
  enVuelo = false;
}
/* Perfiles: quién está frente a la cámara */
function pintarPerfil() {
  const p = st.info?.perfiles; if (!p) return;
  const activo = p.lista.find((x) => x.id === p.activo);
  $("chip-perfil").className = "chip " + (st.persona === "desconocido" ? "aviso" : "ok");
  $("chip-perfil").querySelector(".txt").textContent = `Perfil: ${activo?.nombre || "—"}` + (st.persona === "desconocido" ? " (no reconocido)" : "");
  $("ent-perfil").innerHTML = p.lista.map((x) => `<option value="${x.id}" ${x.id === p.activo ? "selected" : ""}>${x.nombre}</option>`).join("");
  $("chk-reconocer").checked = st.info.reconocer !== false;
  $("ent-perfil-nota").textContent = activo?.tiene_firma
    ? `Ottoman ya conoce la cara de ${activo.nombre}. Cada captura de «Mi cara» lo refuerza.`
    : `Ottoman todavía no conoce la cara de ${activo?.nombre || "este perfil"}: capturá una ráfaga en «Mi cara» y va a poder reconocerla.`;
}
async function refrescarPerfiles() { try { st.info = await api("/api/estado"); pintarEstadoGeneral(); } catch (e) {} }
async function crearPerfil(registrarCara) {
  const nombre = prompt("¿Cómo se llama la persona del perfil nuevo?");
  if (!nombre || !nombre.trim()) return;
  try {
    await api("/api/perfiles", { method: "POST", headers: { "Content-Type": "application/json" },
                                 body: JSON.stringify({ nombre, registrar_cara: !!registrarCara }) });
    st.persona = "reconocido"; $("aviso-persona").hidden = true;
    await refrescarPerfiles(); if ($("dlg-entrenar").open) refrescarEntrenamiento();
    avisar(`Perfil «${nombre.trim()}» creado. Hasta que entrene, Ottoman usa las reglas generales para leer su cara y su voz.`);
  } catch (e) { alert(e.message); }
}
function pintarPersona(persona) {
  if (!persona) return;
  const previo = st.persona; st.persona = persona.estado;
  if (persona.estado === "cambio") { avisar(`Reconocí a ${persona.nombre}: cambié a su perfil.`); refrescarPerfiles(); }
  const aviso = $("aviso-persona");
  if (persona.estado === "desconocido") {
    $("aviso-persona-texto").textContent = `No reconozco esta cara como la de ${persona.nombre}. Mientras tanto leo las expresiones con las reglas generales.`;
    $("btn-soy-yo").textContent = `Soy ${persona.nombre}`;
    aviso.hidden = false;
  } else aviso.hidden = true;
  if (previo !== st.persona) pintarPerfil();
}
function pintarCara(r) {
  pintarPersona(r.persona);
  const c = $("capa"), g = c.getContext("2d");
  g.clearRect(0, 0, c.width, c.height);
  const lista = $("rasgos");
  if (!r.detectada) {
    $("encuadre").textContent = "No veo tu cara";
    $("fuente-cara").textContent = "";
    return;
  }
  const [x0, y0, x1, y1] = r.caja;
  g.strokeStyle = r.encuadre === "Bien encuadrado" ? "#4ade80" : "#fbbf24"; g.lineWidth = 2;
  g.beginPath(); g.roundRect(x0 * c.width, y0 * c.height, (x1 - x0) * c.width, (y1 - y0) * c.height, 10); g.stroke();
  $("encuadre").textContent = r.encuadre;
  if (!lista.dataset.listo) {
    lista.innerHTML = RASGOS_VISIBLES.map((k) => `<li data-k="${k}"><span></span><span class="medidor"><i></i></span></li>`).join("");
    lista.dataset.listo = "1";
  }
  for (const rasgo of r.rasgos) {
    const li = lista.querySelector(`li[data-k="${rasgo.clave}"]`);
    if (!li) continue;
    li.firstElementChild.textContent = rasgo.nombre;
    li.querySelector("i").style.width = Math.min(100, rasgo.valor * 130) + "%";
  }
  const top = ORDEN.reduce((a, b) => (r.probs[a] >= r.probs[b] ? a : b));
  $("fuente-cara").innerHTML = `Lectura instantánea de la cara: <strong style="color:${COLOR[top]}">${NOMBRE[top]} ${pct(r.probs[top])}</strong> · ${r.fuente}`;
}

/* ===================== Chat y turno ===================== */
function mensaje(clase, texto, meta) {
  $("chat-vacio")?.remove();
  const m = document.createElement("div");
  m.className = "msj " + clase; m.textContent = texto;
  if (meta) { const s = document.createElement("span"); s.className = "meta"; s.textContent = meta; m.appendChild(s); }
  $("mensajes").appendChild(m); $("mensajes").scrollTop = $("mensajes").scrollHeight;
  return m;
}
const avisar = (t) => mensaje("sistema", t);

function ocupar(si) {
  st.ocupado = si;
  $("btn-enviar").disabled = si; $("texto").disabled = si;
  $("btn-hablar").disabled = !disp.mic.activa || si;
}
function volverAReposo() {
  Ojos.fijar("reposo"); cuerpo("reposo");
  $("estado-robot").textContent = "En reposo";
}

async function enviarTurno({ texto, wav, dur }) {
  if (st.ocupado) return;
  ocupar(true); detenerVoz();
  const datos = new FormData();
  if (wav) datos.append("audio", wav, "voz.wav"); else datos.append("texto", texto);
  datos.append("usar_camara", disp.camara.activa ? "true" : "false");
  datos.append("voz_caricatura", $("chk-caricatura").checked ? "true" : "false");
  datos.append("ventana", String(wav ? Math.max(4, Math.min(20, dur + 1.5)) : 6));
  const propio = wav ? mensaje("usuario pensando", "Transcribiendo") : mensaje("usuario", texto);
  const espera = mensaje("robot pensando", "Ottoman está pensando");
  Ojos.fijar("pensando"); $("estado-robot").textContent = "Pensando";
  try {
    const r = await api("/api/turno", { method: "POST", body: datos });
    espera.remove();
    if (!r.entendido) { propio.remove(); avisar(r.aviso); volverAReposo(); return; }
    propio.classList.remove("pensando"); propio.textContent = r.usuario;
    pintarVeredicto(r); pintarDecision(r);
    mensaje("robot", r.respuesta, `${r.decision.nombre} · ${r.origen_respuesta}`);
    $("origen-llm").textContent = r.origen_respuesta;
    if (r.llm_error) avisar(`Gemini no respondió (${r.llm_error}). Usé el respaldo local.`);
    st.actitudOjos = r.decision.ojos; st.opcionesOjos = { cuidado: r.decision.actitud === "CUIDADO", expresividad: r.decision.expresividad };
    Ojos.fijar(st.actitudOjos, st.opcionesOjos);
    const disenos = (r.frases || []).map((f) => Ojos.nombre(Ojos.disenoDe(f.gesto))).filter((d, i, l) => d !== l[i - 1]);
    $("mod-cara").textContent = disenos.join(" → ") || `Ojos «${r.decision.ojos}»`;
    await decir(r);
  } catch (e) {
    espera.remove(); propio.classList.remove("pensando");
    avisar("Algo falló al procesar el mensaje: " + e.message); volverAReposo();
  } finally { ocupar(false); $("texto").focus(); }
}

/* ===================== Voz de Ottoman ===================== */
let reproductor = null, cicloEnergia = 0, temporizadores = [];
/* Guion del turno: gesto de apertura, un gesto por frase (repartidos según el largo de cada
   una sobre la duración del audio) y cierre. Lo ejecutan el cuerpo 3D y los ojos a la vez. */
function lanzarGuion(r, duracion) {
  const frases = r.frases || [];
  const total = frases.reduce((n, f) => n + f.texto.length, 0) || 1;
  const guion = [{ t: 0, gesto: "apertura" }];
  let acumulado = 0;
  for (const f of frases) {
    guion.push({ t: Math.max(0.45, (acumulado / total) * duracion), gesto: f.gesto });
    acumulado += f.texto.length;
  }
  fetch("/api/cuerpo", { method: "POST", headers: { "Content-Type": "application/json" },
                         body: JSON.stringify({ modo: "hablando", guion }) }).catch(() => {});
  temporizadores.forEach(clearTimeout);
  temporizadores = guion.map((g, i) => setTimeout(
    () => Ojos.gesto(g.gesto, (guion[i + 1] ? guion[i + 1].t : duracion) - g.t), g.t * 1000));
}
function detenerVoz() {
  if (reproductor) { reproductor.pause(); reproductor = null; }
  if (window.speechSynthesis) speechSynthesis.cancel();
  cancelAnimationFrame(cicloEnergia);
  temporizadores.forEach(clearTimeout); Ojos.gesto(null);
  if (st.hablando) { st.hablando = false; volverAReposo(); }
}
function alEmpezarAHablar(r, duracion) {
  st.hablando = true; Ojos.fijar(st.actitudOjos, st.opcionesOjos); $("estado-robot").textContent = "Hablando";
  lanzarGuion(r, duracion);
}
function alTerminarDeHablar(r) {
  if (!st.hablando) return;
  st.hablando = false; cancelAnimationFrame(cicloEnergia);
  temporizadores.forEach(clearTimeout); Ojos.gesto(null);
  const ultima = r?.frases?.[r.frases.length - 1];
  if (ultima?.gesto === "preguntar" || ultima?.gesto === "ofrecer") {
    // Cierre del turno: si terminó con una pregunta o una invitación, se queda atento
    Ojos.fijar("escuchando"); cuerpo("escuchando"); $("estado-robot").textContent = "Espera tu respuesta";
  } else volverAReposo();
}
/* Voz de caricatura: megáfono de juguete (pasa-banda 650-4350 Hz), saturación (ganancia 1.55
   con recorte en ±0.75) y eco de cuatro retardos (45, 90, 140 y 210 ms). Después, un filtro
   de "color" según la actitud: más cálido al calmar, más brillante al festejar. */
function cadenaCaricatura(ctx, fuente, timbre) {
  const pasaAltos = ctx.createBiquadFilter(); pasaAltos.type = "highpass"; pasaAltos.frequency.value = 650; pasaAltos.Q.value = 0.707;
  const pasaBajos = ctx.createBiquadFilter(); pasaBajos.type = "lowpass"; pasaBajos.frequency.value = 4350; pasaBajos.Q.value = 0.707;
  const pre = ctx.createGain(); pre.gain.value = 1.55;
  const recorte = ctx.createWaveShaper();
  const curva = new Float32Array(2048);
  for (let i = 0; i < curva.length; i++) curva[i] = Math.max(-0.75, Math.min(0.75, (i / (curva.length - 1)) * 2 - 1));
  recorte.curve = curva;
  const mezcla = ctx.createGain(); mezcla.gain.value = 0.62;
  fuente.connect(pasaAltos); pasaAltos.connect(pasaBajos); pasaBajos.connect(pre); pre.connect(recorte); recorte.connect(mezcla);
  for (const [seg, ganancia] of [[0.045, 0.32], [0.090, 0.22], [0.140, 0.15], [0.210, 0.09]]) {
    const retardo = ctx.createDelay(0.5); retardo.delayTime.value = seg;
    const g = ctx.createGain(); g.gain.value = ganancia;
    recorte.connect(retardo); retardo.connect(g); g.connect(mezcla);
  }
  let salida = mezcla;
  if (timbre === "calido" || timbre === "brillante") {
    const color = ctx.createBiquadFilter();
    if (timbre === "calido") { color.type = "lowpass"; color.frequency.value = 2600; color.Q.value = 0.707; }
    else { color.type = "highshelf"; color.frequency.value = 1500; color.gain.value = 3; }
    mezcla.connect(color); salida = color;
  }
  salida.connect(ctx.destination);
}
function decir(r) {
  return new Promise((listo) => {
    const fin = () => { alTerminarDeHablar(r); listo(); };
    const estimada = Math.max(2, r.respuesta.length / 14);
    if (r.audio) {
      const a = new Audio(r.audio); reproductor = a;
      try {
        const ctx = contexto();
        const fuente = ctx.createMediaElementSource(a);
        if (r.caricatura) cadenaCaricatura(ctx, fuente, r.decision.timbre); else fuente.connect(ctx.destination);
      } catch (e) { /* sin efectos: suena la voz tal como viene */ }
      a.onplay = () => alEmpezarAHablar(r, isFinite(a.duration) && a.duration > 0 ? a.duration : estimada);
      a.onended = fin; a.onerror = () => decirConNavegador(r, fin);
      a.play().catch(() => decirConNavegador(r, fin));
    } else decirConNavegador(r, fin);
  });
}
function decirConNavegador(r, fin) {
  if (!window.speechSynthesis) { fin(); return; }
  const u = new SpeechSynthesisUtterance(r.respuesta);
  const voces = speechSynthesis.getVoices();
  u.voice = voces.find((v) => /es[-_]AR/i.test(v.lang)) || voces.find((v) => /^es/i.test(v.lang)) || null;
  u.lang = u.voice?.lang || "es-AR";
  const extra = r.caricatura ? { v: 14, t: 20 } : { v: 0, t: 0 };
  u.rate = 1 + (r.decision.voz.velocidad + extra.v) / 100; u.pitch = 1 + (r.decision.voz.tono + extra.t) / 50;
  u.volume = Math.min(1, 0.9 + r.decision.voz.volumen / 100);
  u.onstart = () => alEmpezarAHablar(r, Math.max(2, r.respuesta.length / (14 * u.rate))); u.onend = fin; u.onerror = fin;
  speechSynthesis.speak(u);
}

/* ===================== Paneles de lectura ===================== */
function pintarVeredicto(r) {
  const v = r.veredicto;
  $("v-emocion").textContent = NOMBRE[v.emocion]; $("v-emocion").style.color = COLOR[v.emocion];
  $("v-detalle").textContent = `Confianza ${pct(v.confianza)} · intensidad ${pct(v.intensidad)}`;
  $("veredicto-tiempo").textContent = `resuelto en ${r.segundos.toFixed(1)} s`;
  $("v-barras").innerHTML = ORDEN.map((e) =>
    `<li><span>${NOMBRE[e]}</span><span class="medidor"><i style="width:${v.probs[e] * 100}%;background:${COLOR[e]}"></i></span><span class="num">${pct(v.probs[e])}</span></li>`).join("");
  const fuentes = [["texto", "Tus palabras"], ["cara", "Tu cara"], ["voz", "Tu voz"]];
  $("v-senales").innerHTML = fuentes.map(([k, nombre]) => {
    const m = v.por_modalidad[k];
    if (!m) return `<li class="falta"><span class="quien">${nombre}</span><span>No disponible</span></li>`;
    return `<li><span class="quien">${nombre} · pesa ${pct(m.peso)}</span><span style="color:${COLOR[m.emocion]}">${NOMBRE[m.emocion]} ${pct(m.prob)}</span></li>`;
  }).join("");
  const nota = $("v-nota");
  if (v.incongruencia) {
    const i = v.incongruencia;
    nota.textContent = `Tus palabras suenan a ${NOMBRE[i.texto].toLowerCase()}, pero tu ${i.otra} muestra ${NOMBRE[i.emocion_otra].toLowerCase()}. Ottoman pregunta antes de asumir.`;
    nota.hidden = false;
  } else nota.hidden = true;
}
function pintarReglas(reglas, mods, porque) {
  $("reglas").innerHTML = reglas.map((r) =>
    `<li class="${r.elegida ? "elegida" : ""} ${r.cumple ? "cumple" : ""}"><span class="id">${r.id}</span><span class="si">${r.si}</span><span class="entonces">→ ${r.actitud_nombre}</span></li>`).join("");
  $("mods").innerHTML = mods.map((m) =>
    `<li class="${m.activo ? "activo" : ""}"><span class="id">${m.id}</span><span>${m.si}<span class="accion">→ ${m.entonces}</span></span></li>`).join("");
  $("regla-porque").textContent = porque || "Se evalúan en orden y gana la primera que se cumple. La actitud elegida gobierna el cuerpo, la cara, el léxico y la voz.";
}
function pintarDecision(r) {
  const d = r.decision;
  $("actitud-nombre").textContent = d.nombre; $("actitud-desc").textContent = d.descripcion;
  $("actitud-desc").classList.remove("apagado");
  const signo = (n) => (n > 0 ? "+" : "") + n;
  const gestos = (r.frases || []).map((f) => f.nombre).join(" → ");
  $("mod-cuerpo").textContent = `${gestos || "Gestos de la actitud"} · expresividad ${pct(d.expresividad)}`;
  $("mod-lexico").textContent = r.texto.registro ? `Se acerca a tu forma de hablar: ${r.texto.registro}` : "Tono propio de la actitud";
  $("mod-voz").textContent = `${r.caricatura ? "Caricatura" : "Natural"}, color ${d.timbre === "calido" ? "cálido" : d.timbre} · velocidad ${signo(d.voz.velocidad)}% · tono ${signo(d.voz.tono)} Hz`;
  pintarReglas(d.reglas, d.modificadores, `Se aplicó ${d.regla}: ${d.porque}`);
}

/* ===================== Entrenamiento ===================== */
const AYUDA = {
  cara: "Elegí una emoción, hacé esa expresión mirando a la cámara y tocá «Capturar» (3 segundos). Repetí al menos 3 veces por emoción, cambiando un poco la pose. Empezá por «Neutro»: es tu cara en reposo.",
  voz: "Elegí una emoción, tocá «Grabar», decí una frase de 3 a 6 segundos con ese tono y tocá «Detener». Repetí al menos 3 veces por emoción.",
};
const CONSEJO = {
  cara: {
    NEUTRO: "Cara relajada, como cuando leés la pantalla. Sin levantar las cejas.",
    ALEGRIA: "Sonrisa de verdad, de las que levantan las mejillas.",
    TRISTEZA: "Comisuras hacia abajo y cejas en ángulo (las puntas de adentro hacia arriba).",
    ENOJO: "Ceño bien fruncido y labios apretados.",
    ANSIEDAD: "Ojos bien abiertos, cejas levantadas y boca tensa, como preocupado o asustado.",
  },
  voz: {
    NEUTRO: "Como leyendo un dato cualquiera, sin emoción.",
    ALEGRIA: "Con energía y entusiasmo, como contando una buena noticia.",
    TRISTEZA: "Bajo, lento y apagado.",
    ENOJO: "Firme, fuerte y cortante.",
    ANSIEDAD: "Rápido, entrecortado, como apurado o nervioso.",
  },
};
async function refrescarEntrenamiento() {
  st.info = await api("/api/estado");
  const datos = st.info[st.mod].datos;
  $("ent-ayuda").innerHTML = `${AYUDA[st.mod]}<br><br><strong>${NOMBRE[st.etiqueta]}:</strong> ${CONSEJO[st.mod][st.etiqueta]}`;
  $("ent-capturar").textContent = st.mod === "cara" ? "Capturar (3 s)" : (st.grabando ? "Detener" : "Grabar");
  $("ent-emociones").innerHTML = ORDEN.map((e) => `<button data-e="${e}" class="${e === st.etiqueta ? "activa" : ""}">${NOMBRE[e]}</button>`).join("");
  $("ent-tabla").innerHTML = `<tr><th>Emoción</th><th class="num">${st.mod === "cara" ? "Ráfagas" : "Frases"}</th><th class="num">Muestras</th></tr>` +
    ORDEN.map((e) => `<tr><td>${NOMBRE[e]}</td><td class="num">${datos.rafagas[e]}</td><td class="num">${datos.muestras[e]}</td></tr>`).join("");
  pintarResultado(datos.modelo);
  pintarEstadoGeneral();
}
function pintarResultado(m, error) {
  const caja = $("ent-resultado");
  if (error) { caja.innerHTML = `<p class="error">${error}</p>`; return; }
  if (!m) { caja.innerHTML = `<p class="apagado">Cuando entrenes, acá aparece qué tan bien reconoce tus emociones el modelo.</p>`; return; }
  const comp = Object.entries(m.comparacion).map(([n, a]) => `<tr><td>${n}</td><td class="num">${pct(a)}</td></tr>`).join("");
  const cab = m.clases.map((c) => `<th class="num">${NOMBRE[c].slice(0, 4)}.</th>`).join("");
  const filas = m.matriz_confusion.map((fila, i) =>
    `<tr><td>${NOMBRE[m.clases[i]]}</td>${fila.map((n, j) => `<td class="num ${i === j ? "diag" : ""}">${n}</td>`).join("")}</tr>`).join("");
  caja.innerHTML = `<p class="apagado">Acierto en muestras que el modelo no vio</p><p class="grande">${pct(m.precision_validacion)}</p>
    <p>Modelo elegido: <strong>${m.modelo}</strong></p>
    <table class="tabla"><tr><th>Modelo comparado</th><th class="num">Acierto</th></tr>${comp}</table>
    <table class="tabla matriz"><tr><th>Real \\ Predicho</th>${cab}</tr>${filas}</table>
    <p class="apagado">${m.muestras} muestras en ${m.rafagas} capturas · ${m.validacion} · ${m.fecha}</p>`;
}
function finCaptura() {
  const cap = st.captura; st.captura = null;
  $("ent-capturar").disabled = false;
  $("ent-progreso").textContent = cap.n ? `Listo: ${cap.n} muestras de ${NOMBRE[cap.etiqueta].toLowerCase()}.` : "No se vio tu cara: no se guardó nada.";
  refrescarEntrenamiento();
}
async function capturar() {
  if (st.mod === "cara") {
    if (!disp.camara.activa) { $("ent-progreso").textContent = "Encendé la cámara para capturar."; return; }
    st.captura = { etiqueta: st.etiqueta, rafaga: Date.now().toString(36), hasta: performance.now() + 3000, n: 0 };
    $("ent-capturar").disabled = true; $("ent-progreso").textContent = "Capturando…";
    return;
  }
  if (!disp.mic.activa) { $("ent-progreso").textContent = "Encendé el micrófono para grabar."; return; }
  if (!st.grabando) { empezarGrabacion("entrenamiento"); $("ent-capturar").textContent = "Detener"; $("ent-progreso").textContent = "Grabando…"; return; }
  const g = terminarGrabacion(); $("ent-capturar").textContent = "Grabar";
  const datos = new FormData(); datos.append("etiqueta", st.etiqueta); datos.append("audio", g.wav, "muestra.wav");
  try { await api("/api/voz_muestra", { method: "POST", body: datos }); $("ent-progreso").textContent = "Frase guardada."; }
  catch (e) { $("ent-progreso").textContent = e.message; }
  refrescarEntrenamiento();
}

/* ===================== Estado general ===================== */
function pintarEstadoGeneral() {
  const i = st.info; if (!i) return;
  pintarPerfil();
  const coach = $("chip-coach"), tres = $("chip-3d");
  coach.className = "chip fijo " + (i.llm.disponible ? "ok" : "aviso");
  coach.querySelector(".txt").textContent = i.llm.disponible ? `Coach: Gemini (${i.llm.modelo})`
    : i.llm.estado === "conectando" ? "Coach: conectando con Gemini…" : `Coach: respaldo local (${i.llm.estado})`;
  // Cuota gratuita de Gemini: cada turno gasta 2 pedidos; los modelos "flash" admiten 20 por día
  const uso = i.llm.uso?.modelos || {}, cadena = i.llm.candidatos || [];
  const total = Object.values(uso).reduce((n, m) => n + m.pedidos, 0);
  const libres = cadena.filter((m) => !uso[m]?.agotado && (m.includes("lite") || (uso[m]?.pedidos || 0) < i.llm.limite_flash)).length;
  const chipUso = $("chip-uso");
  chipUso.className = "chip fijo " + (!cadena.length ? "" : libres >= 3 ? "ok" : libres >= 1 ? "aviso" : "mal");
  chipUso.querySelector(".txt").textContent = cadena.length ? `Gemini hoy: ${total} pedidos · ${libres} de ${cadena.length} modelos con cuota` : "Gemini hoy: sin clave";
  chipUso.title = "Pedidos por modelo en el día de cuota (se reinicia a la medianoche del Pacífico, cerca de las 5 de la mañana de Argentina):\n" +
    cadena.map((m) => `${m}: ${uso[m]?.pedidos || 0}${m.includes("lite") ? "" : " de " + i.llm.limite_flash}${uso[m]?.agotado ? " (agotado)" : ""}`).join("\n");
  tres.className = "chip fijo " + (i.cuerpo.disponible ? "ok" : "mal");
  tres.querySelector(".txt").textContent = i.cuerpo.disponible ? "3D: activo" : "3D: no disponible";
  $("aviso-sim").hidden = i.cuerpo.disponible; $("sim").hidden = !i.cuerpo.disponible;
  if (!i.llm.disponible && i.llm.estado !== "conectando") $("pie-chat").textContent = "Sin conexión con Gemini: Ottoman responde con frases de respaldo y no puede transcribir audio. " +
    "No es un profesional de la salud y no da consejo médico.";
}

/* Cámara del simulador: arrastrar mueve, clic derecho gira, la rueda acerca o aleja.
   Con el simulador enfocado (un clic sobre él), R reinicia la vista y C hace un primer plano
   de la cara. Las teclas solo actúan ahí: escribir en el chat no las dispara. */
function prepararCamaraSim() {
  const caja = $("sim-caja");
  const enviar = (accion, dx = 0, dy = 0) => fetch("/api/camara", { method: "POST", headers: { "Content-Type": "application/json" },
                                                                    body: JSON.stringify({ accion, dx, dy }) }).catch(() => {});
  let arrastre = null, dx = 0, dy = 0, ultimo = 0;
  const volcar = () => { if (arrastre && (dx || dy)) { enviar(arrastre.accion, dx, dy); dx = dy = 0; ultimo = performance.now(); } };
  caja.addEventListener("mousedown", (ev) => {
    if (ev.button !== 0 && ev.button !== 2) return;
    ev.preventDefault(); caja.focus();
    arrastre = { accion: ev.button === 2 ? "rotar" : "mover", x: ev.clientX, y: ev.clientY };
    caja.classList.add("arrastrando");
  });
  window.addEventListener("mousemove", (ev) => {
    if (!arrastre) return;
    dx += ev.clientX - arrastre.x; dy += ev.clientY - arrastre.y; arrastre.x = ev.clientX; arrastre.y = ev.clientY;
    if (performance.now() - ultimo > 40) volcar();
  });
  window.addEventListener("mouseup", () => { volcar(); arrastre = null; caja.classList.remove("arrastrando"); });
  caja.addEventListener("contextmenu", (ev) => ev.preventDefault());
  caja.addEventListener("wheel", (ev) => { ev.preventDefault(); enviar("zoom", 0, ev.deltaY); }, { passive: false });
  caja.addEventListener("keydown", (ev) => {
    if (ev.ctrlKey || ev.altKey || ev.metaKey) return;
    const tecla = ev.key.toLowerCase();
    if (tecla === "r") { ev.preventDefault(); enviar("reset"); }
    if (tecla === "c") { ev.preventDefault(); enviar("cara"); }
  });
}

async function iniciar() {
  // Los ojos de la cabeza 3D muestran el mismo diseño que el panel de la cara
  Ojos.alCambiar = (diseno) => fetch("/api/ojos", { method: "POST", headers: { "Content-Type": "application/json" },
                                                    body: JSON.stringify({ diseno }) }).catch(() => {});
  Ojos.iniciar($("ojos"));
  prepararCamaraSim();
  try { $("chk-caricatura").checked = localStorage.getItem("voz_caricatura") !== "no"; } catch (e) {}
  $("chk-caricatura").onchange = (ev) => { try { localStorage.setItem("voz_caricatura", ev.target.checked ? "si" : "no"); } catch (e) {} };
  try {
    st.info = await api("/api/estado");
    pintarEstadoGeneral();
    pintarReglas(st.info.reglas.reglas, st.info.reglas.modificadores);
    if (st.info.cuerpo.disponible) $("sim").src = "/sim/stream";
  } catch (e) { avisar("No pude conectar con el servidor local: " + e.message); }
  pintarChips();
  await buscarDispositivos();
  setInterval(cicloCamara, 220);
  setInterval(async () => { try { st.info = await api("/api/estado"); pintarEstadoGeneral(); } catch (e) {} }, 7000);
  navigator.mediaDevices?.addEventListener?.("devicechange", buscarDispositivos);

  $("btn-buscar").onclick = buscarDispositivos;
  $("btn-camara-reintentar").onclick = () => { disp.camara.deseada = true; buscarDispositivos(); };
  $("chip-camara").onclick = () => alternar("camara");
  $("chip-mic").onclick = () => alternar("mic");
  $("btn-hablar").onclick = () => (st.grabando ? terminarGrabacion() : empezarGrabacion("chat"));
  $("form-chat").onsubmit = (ev) => {
    ev.preventDefault();
    const t = $("texto").value.trim(); if (!t || st.ocupado) return;
    $("texto").value = ""; contexto(); enviarTurno({ texto: t });
  };
  let escribiendo = false;
  $("texto").oninput = () => {
    const hay = $("texto").value.length > 0;
    if (hay && !escribiendo && !st.ocupado && !st.hablando) { escribiendo = true; Ojos.fijar("escuchando"); cuerpo("escuchando"); $("estado-robot").textContent = "Te lee"; }
    if (!hay && escribiendo) { escribiendo = false; volverAReposo(); }
  };
  $("form-chat").addEventListener("submit", () => (escribiendo = false));
  $("btn-nueva").onclick = async () => {
    detenerVoz(); await api("/api/reiniciar", { method: "POST" });
    $("mensajes").innerHTML = ""; avisar("Conversación nueva. Ottoman no recuerda lo anterior.");
    st.actitudOjos = "reposo"; volverAReposo();
  };

  const dlg = $("dlg-entrenar");
  $("chip-perfil").onclick = () => { dlg.showModal(); refrescarEntrenamiento(); };
  $("ent-perfil").onchange = async (ev) => {
    await api("/api/perfiles", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id: ev.target.value }) });
    st.persona = "reconocido"; refrescarEntrenamiento();
  };
  $("ent-perfil-nuevo").onclick = () => crearPerfil(false);
  $("btn-perfil-nuevo-aviso").onclick = () => crearPerfil(true);
  $("btn-soy-yo").onclick = async () => { await api("/api/perfiles/soy_yo", { method: "POST" }); st.persona = "reconocido"; $("aviso-persona").hidden = true; refrescarPerfiles(); };
  $("chk-reconocer").onchange = (ev) => api("/api/perfiles/reconocer", { method: "POST", headers: { "Content-Type": "application/json" },
                                                                         body: JSON.stringify({ activo: ev.target.checked }) });
  $("btn-entrenar").onclick = () => { dlg.showModal(); refrescarEntrenamiento(); };
  $("dlg-cerrar").onclick = () => dlg.close();
  document.querySelectorAll(".pestana").forEach((p) => (p.onclick = () => {
    if (st.grabando || st.captura) return;
    document.querySelectorAll(".pestana").forEach((x) => x.classList.toggle("activa", x === p));
    st.mod = p.dataset.mod; $("ent-progreso").textContent = ""; refrescarEntrenamiento();
  }));
  $("ent-emociones").onclick = (ev) => {
    const e = ev.target.dataset?.e; if (!e || st.grabando || st.captura) return;
    st.etiqueta = e; refrescarEntrenamiento();
  };
  $("ent-capturar").onclick = capturar;
  $("ent-entrenar").onclick = async () => {
    $("ent-entrenar").disabled = true; $("ent-entrenar").textContent = "Entrenando…";
    try { pintarResultado(await api(`/api/entrenar/${st.mod}`, { method: "POST" })); st.info = await api("/api/estado"); }
    catch (e) { pintarResultado(null, e.message); }
    $("ent-entrenar").disabled = false; $("ent-entrenar").textContent = "Entrenar modelo";
  };
  $("ent-borrar").onclick = async () => {
    if (!confirm(`¿Borrar todas tus muestras de ${st.mod} y el modelo entrenado? No se puede deshacer.`)) return;
    await api(`/api/borrar/${st.mod}`, { method: "POST" }); $("ent-progreso").textContent = "Datos borrados."; refrescarEntrenamiento();
  };
}
iniciar();
