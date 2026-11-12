/* Módulo de cara: los ojos de Ottoman.
   Diez diseños en un solo celeste. El diseño cambia con lo que Ottoman va diciendo (uno
   por frase, atado al mismo gesto que usa el cuerpo) y la actitud del coach decide cuáles
   están permitidos. El paso de un diseño a otro es un parpadeo: cierra, cambia, abre.
   No laten ni cambian de tamaño; el único salto de tamaño es al enfatizar. */
(function () {
  const COLOR = "#38bdf8";
  const FONDO = "#03060d";
  const NOMBRES = {
    capsula: "Cápsula", arco: "Arco feliz", estrella: "Estrella", exclamacion: "!", pregunta: "?",
    calma: "Suave calma", entrecerrado: "Entrecerrado", circular: "Circular", empatia: "Empatía",
  };

  // Qué diseño acompaña a cada gesto de frase
  const POR_GESTO = {
    apertura: "capsula", validar: "empatia", preguntar: "pregunta", enfatizar: "exclamacion",
    explicar: "capsula", calmar: "calma", celebrar: "arco", pensar: "entrecerrado",
    ofrecer: "capsula", yo: "capsula", limite: "entrecerrado",
  };
  // Diseño de base de cada estado o actitud
  const BASE = {
    reposo: "capsula", escuchando: "capsula", pensando: "entrecerrado",
    calma: "calma", calida: "empatia", alegre: "arco", seria: "entrecerrado", curiosa: "capsula",
  };
  // Lo que cada actitud no permite, y con qué se reemplaza
  const RESTRICCION = {
    seria: { arco: "capsula", estrella: "capsula", circular: "capsula" },
    calma: { estrella: "capsula", exclamacion: "capsula", circular: "capsula" },
    calida: { estrella: "arco", exclamacion: "capsula" },
    cuidado: { arco: "capsula", estrella: "capsula", exclamacion: "capsula", pregunta: "capsula",
               circular: "capsula", entrecerrado: "capsula" },
  };

  let ctx, W, H, U;                      // U: unidad de dibujo (altura de la cápsula)
  let actitud = "reposo", cuidado = false, expresividad = 0.6;
  let base = "capsula";                  // diseño al que vuelve cuando no hay gesto
  let visible = "capsula";               // diseño que se está dibujando
  let pendiente = null;                  // diseño al que se cambia cuando el párpado está cerrado
  let cierre = 0, sentido = 0;           // parpadeo: 0 abierto, 1 cerrado
  let salto = 0;                         // énfasis: salto de tamaño que se apaga solo
  let mirada = { x: 0, y: 0 }, miradaObj = { x: 0, y: 0 };
  let tParpadeo = 3, tMirada = 2, tGesto = 0, despues = null, previo = 0;

  const azar = (a, b) => a + Math.random() * (b - a);

  function permitido(d) {
    const r = RESTRICCION[cuidado ? "cuidado" : actitud] || {};
    return r[d] || d;
  }
  function disenoDe(gesto) {
    let d = POR_GESTO[gesto] || "capsula";
    if (gesto === "apertura" && actitud === "alegre") d = "circular";
    if (gesto === "celebrar" && actitud === "alegre" && expresividad > 0.8) d = "estrella";
    if (gesto === "explicar" && actitud === "seria") d = "entrecerrado";
    if (gesto === "ofrecer" && actitud === "calida") d = "empatia";
    return permitido(d);
  }
  function cambiar(d) {
    if (d === visible && !pendiente) return;
    if (d === (pendiente || visible)) return;
    pendiente = d; sentido = 1;          // cierra, cambia, abre
  }
  function parpadear() { if (!sentido && !pendiente) sentido = 1; }

  /* ---------- dibujo de cada diseño (centrado en 0,0) ---------- */
  function capsula(w, h) { ctx.beginPath(); ctx.roundRect(-w / 2, -h / 2, w, h, Math.min(w, h) / 2); ctx.fill(); }
  const DIBUJO = {
    capsula() { capsula(U * 0.66, U); },
    entrecerrado() { capsula(U * 0.78, U * 0.28); },
    circular() { ctx.beginPath(); ctx.arc(0, 0, U * 0.6, 0, Math.PI * 2); ctx.fill(); },
    arco() {
      ctx.lineWidth = U * 0.13; ctx.lineCap = "round";
      ctx.beginPath(); ctx.arc(0, U * 0.2, U * 0.4, Math.PI * 1.04, Math.PI * 1.96); ctx.stroke();
    },
    calma(lado) {                        // línea suave que sube hacia el centro de la cara
      ctx.lineWidth = U * 0.1; ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(-U * 0.4 * lado, U * 0.11); ctx.lineTo(U * 0.4 * lado, -U * 0.11); ctx.stroke();
    },
    empatia() {                          // media cápsula: el párpado de abajo sube, como al sonreír con los ojos
      const w = U * 0.66, h = U * 0.46;
      ctx.beginPath(); ctx.moveTo(-w / 2, h / 2);
      ctx.lineTo(-w / 2, 0); ctx.bezierCurveTo(-w / 2, -h * 0.75, w / 2, -h * 0.75, w / 2, 0);
      ctx.lineTo(w / 2, h / 2); ctx.closePath(); ctx.fill();
    },
    estrella() {
      const a = U * 0.6, b = U * 0.48, c = U * 0.13;
      ctx.beginPath(); ctx.moveTo(0, -a); ctx.lineTo(c, -c); ctx.lineTo(b, 0); ctx.lineTo(c, c);
      ctx.lineTo(0, a); ctx.lineTo(-c, c); ctx.lineTo(-b, 0); ctx.lineTo(-c, -c); ctx.closePath(); ctx.fill();
    },
  };
  function glifo(texto) {
    ctx.font = `800 ${U * 1.25}px "Segoe UI", system-ui, sans-serif`;
    ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(texto, 0, U * 0.06);
  }

  function cuadro(ahora) {
    const dt = Math.min(0.3, (ahora - previo) / 1000); previo = ahora;
    // Parpadeo y cambio de diseño
    if (sentido === 1) { cierre += dt / 0.07; if (cierre >= 1) { cierre = 1; if (pendiente) { visible = pendiente; pendiente = null; if (window.Ojos.alCambiar) window.Ojos.alCambiar(visible); } sentido = -1; } }
    else if (sentido === -1) { cierre -= dt / 0.10; if (cierre <= 0) { cierre = 0; sentido = 0; } }
    tParpadeo -= dt; if (tParpadeo <= 0) { parpadear(); tParpadeo = azar(2.8, 6); }
    // Mirada: se desplaza apenas, sin cambiar de tamaño
    tMirada -= dt;
    if (tMirada <= 0) {
      miradaObj = actitud === "pensando" ? { x: -0.3 + azar(-0.04, 0.04), y: -0.2 } : { x: azar(-0.05, 0.05), y: azar(-0.03, 0.03) };
      tMirada = azar(1.2, 3.2);
    }
    const k = 1 - Math.exp(-dt * 6);
    mirada.x += (miradaObj.x - mirada.x) * k; mirada.y += (miradaObj.y - mirada.y) * k;
    // Fin del gesto de la frase: vuelve al diseño que corresponde
    if (tGesto > 0) { tGesto -= dt; if (tGesto <= 0 && despues) { cambiar(despues); despues = null; } }
    salto = Math.max(0, salto - dt / 0.35);

    ctx.fillStyle = FONDO; ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = COLOR; ctx.strokeStyle = COLOR; ctx.shadowColor = COLOR; ctx.shadowBlur = U * 0.22;
    const escala = 1 + 0.16 * salto, sy = Math.max(0.07, 1 - cierre);
    const cx = W / 2 + mirada.x * U, cy = H / 2 + mirada.y * U;
    if (visible === "exclamacion" || visible === "pregunta") {
      ctx.save(); ctx.translate(cx, cy); ctx.scale(escala, escala * sy); glifo(visible === "pregunta" ? "?" : "!"); ctx.restore();
    } else {
      for (const lado of [1, -1]) {      // 1: ojo de la izquierda de la pantalla
        ctx.save(); ctx.translate(cx - lado * U * 0.86, cy); ctx.scale(escala, escala * sy);
        DIBUJO[visible](lado); ctx.restore();
      }
    }
    ctx.shadowBlur = 0;
    requestAnimationFrame(cuadro);
  }

  window.Ojos = {
    iniciar(canvas) {
      ctx = canvas.getContext("2d"); W = canvas.width; H = canvas.height; U = H * 0.44;
      previo = performance.now(); requestAnimationFrame(cuadro);
    },
    /* Estado o actitud: reposo | escuchando | pensando | calma | calida | alegre | seria | curiosa */
    fijar(nombre, opciones) {
      if (!(nombre in BASE)) nombre = "reposo";
      actitud = nombre; cuidado = !!(opciones && opciones.cuidado);
      if (opciones && opciones.expresividad !== undefined) expresividad = opciones.expresividad;
      base = permitido(BASE[nombre]); tGesto = 0; despues = null; tMirada = 0;
      cambiar(base);
    },
    /* Gesto de una frase: muestra su diseño mientras dura la frase. Los símbolos ! y ? son
       breves y después vuelven los ojos. */
    gesto(nombre, segundos) {
      if (!nombre) { tGesto = 0; despues = null; cambiar(base); return; }
      const d = disenoDe(nombre);
      const esGlifo = d === "exclamacion" || d === "pregunta";
      cambiar(d);
      if (nombre === "enfatizar") salto = 1;
      tGesto = esGlifo ? Math.min(1.1, segundos || 1.1) : (segundos || 2);
      despues = esGlifo ? "capsula" : null;
      tMirada = 0;
    },
    disenoDe, nombre: (d) => NOMBRES[d] || d,
    hablar() {}, energia() {},           // se conservan por compatibilidad: los ojos ya no laten con la voz
    disenos: Object.keys(NOMBRES),
    mostrar(d) { cambiar(d); },          // para la página de catálogo
  };
})();
