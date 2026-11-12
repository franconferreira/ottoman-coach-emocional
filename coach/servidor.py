"""Servidor local del dashboard: une percepción, reglas, coach y expresión."""
from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

RAIZ = Path(__file__).resolve().parent.parent
load_dotenv(RAIZ / ".env")

from . import entrenamiento, reglas, tts  # noqa: E402
from .cara import AnalizadorCara  # noqa: E402
from .cuerpo import GESTOS, AnimadorCuerpo, gesto_permitido  # noqa: E402
from .emociones import EMOCIONES, NOMBRES, PESOS, fusionar  # noqa: E402
from .llm import LIMITE_DIARIO_FLASH, CoachLLM, gestos_por_heuristica, uso_de_hoy  # noqa: E402
from .voz import AnalizadorVoz  # noqa: E402

app = FastAPI(title="Ottoman Coach Emocional")
cara = AnalizadorCara()
voz = AnalizadorVoz()
llm = CoachLLM()
try:
    cuerpo: AnimadorCuerpo | None = AnimadorCuerpo()
    cuerpo.iniciar()
    error_cuerpo = None
except Exception as e:  # sin 3D el resto del dashboard sigue funcionando
    cuerpo, error_cuerpo = None, f"{type(e).__name__}: {e}"

sesion = {"historial": [], "emociones": []}


def _fiabilidad(modalidad: str) -> float:
    """Acierto del modelo personal en validación; sin modelo, las reglas base valen 0,6."""
    modelo = entrenamiento.resumen(modalidad)["modelo"]
    return float(modelo["precision_validacion"]) if modelo else 0.6


def _estado() -> dict:
    return {
        "llm": {"disponible": llm.disponible, "estado": llm.estado, "modelo": llm.modelo or None,
                "ultimo_error": llm.ultimo_error, "uso": uso_de_hoy(), "candidatos": llm.candidatos,
                "limite_flash": LIMITE_DIARIO_FLASH},
        "cuerpo": {"disponible": cuerpo is not None and cuerpo.error is None,
                   "error": error_cuerpo or (cuerpo.error if cuerpo else None)},
        "cara": {"modelo_personal": cara.modelo.disponible(), "datos": entrenamiento.resumen("cara")},
        "voz": {"modelo_personal": voz.modelo.disponible(), "datos": entrenamiento.resumen("voz")},
        "emociones": [{"clave": e, "nombre": NOMBRES[e]} for e in EMOCIONES],
        "pesos": PESOS,
        "reglas": reglas.tabla_reglas(),
        "turnos": len(sesion["historial"]),
    }


@app.get("/api/estado")
def api_estado():
    return _estado()


# ------------------------------------------------------------------ cara
@app.post("/api/cara")
async def api_cara(request: Request, etiqueta: str | None = None, rafaga: str | None = None):
    """Analiza un cuadro de la cámara. Con `etiqueta` además lo guarda como muestra de entrenamiento."""
    jpeg = await request.body()
    r = await asyncio.to_thread(cara.analizar, jpeg, etiqueta is None)
    guardada = False
    if etiqueta and r.get("detectada"):
        entrenamiento.agregar("cara", etiqueta, rafaga or "r0", r["columnas"], r["valores"])
        guardada = True
    r.pop("columnas", None)
    r.pop("valores", None)
    r["guardada"] = guardada
    return r


# ------------------------------------------------------------------ turno completo
@app.post("/api/turno")
async def api_turno(texto: str | None = Form(None), audio: UploadFile | None = File(None),
                    usar_camara: bool = Form(False), ventana: float = Form(8.0),
                    voz_caricatura: bool = Form(True)):
    t0 = time.time()
    wav = await audio.read() if audio is not None else None
    if not wav and not (texto and texto.strip()):
        raise HTTPException(400, "Mandá un texto o un audio.")
    if cuerpo:
        cuerpo.fijar(modo="pensando")

    # 1) Percepción: palabras (LLM), voz (señal) y cara (ventana de los últimos segundos)
    tarea_analisis = asyncio.to_thread(llm.analizar, texto, wav)
    tarea_voz = asyncio.to_thread(voz.analizar, wav) if wav else None
    analisis = await tarea_analisis
    r_voz = await tarea_voz if tarea_voz else None
    dicho = analisis["transcripcion"]
    if not dicho:
        if cuerpo:
            cuerpo.fijar(modo="reposo")
        aviso = (f"Gemini no pudo procesar el audio ({llm.ultimo_error}). Probá de nuevo o escribilo."
                 if llm.disponible and llm.ultimo_error else
                 "No llegué a entender el audio. ¿Podés repetirlo o escribirlo?" if llm.disponible else
                 "Sin conexión con Gemini no puedo transcribir audio. Escribime el mensaje en el chat.")
        return {"entendido": False, "aviso": aviso}
    r_cara = cara.resumen_ventana(ventana) if usar_camara else None
    # Una señal poco confiable pesa menos: su peso se multiplica por el acierto de su modelo
    if r_cara:
        r_cara["calidad"] *= _fiabilidad("cara")
    if r_voz:
        r_voz["calidad"] *= _fiabilidad("voz")

    # 2) Veredicto: fusión de las señales disponibles
    veredicto = fusionar(
        {"texto": {"probs": analisis["probs"], "calidad": 1.0}, "cara": r_cara, "voz": r_voz},
        intensidad_texto=analisis["intensidad"],
        activacion_voz=r_voz["activacion"] if r_voz else None,
        afirma_estar_bien=bool(analisis.get("dice_estar_bien")),
    )

    # 3) Reglas: actitud del coach
    decision = reglas.decidir(veredicto, dicho, analisis, sesion["emociones"])

    # Ottoman no interroga: solo puede preguntar si no preguntó en sus dos turnos anteriores
    recientes = ["?" in h["robot"] for h in sesion["historial"][-2:]]
    if veredicto["emocion"] in ("TRISTEZA", "ENOJO", "ANSIEDAD"):
        analisis["puede_preguntar"] = not any(recientes)
    else:
        analisis["puede_preguntar"] = not (recientes and recientes[-1])
    # Cada tanto puede comentar, como amigo, lo que nota en la cara o en la voz
    analisis["observacion"] = None
    if len(sesion["historial"]) % 3 == 1:
        como = {"TRISTEZA": "apagada", "ANSIEDAD": "acelerada", "ENOJO": "tensa", "ALEGRIA": "contenta"}
        for modalidad, parte in (("voz", "la voz"), ("cara", "la cara")):
            m = veredicto["por_modalidad"].get(modalidad)
            if m and m["emocion"] in como and m["prob"] >= 0.6 and m["emocion"] == veredicto["emocion"]:
                analisis["observacion"] = f"le notás {parte} {como[m['emocion']]}"
                break

    # 4) Respuesta: mensaje fijo en protocolo de cuidado; si no, la redacta el LLM
    if decision["actitud"] == "CUIDADO":
        respuesta, origen = reglas.MENSAJE_CUIDADO, "mensaje fijo de seguridad"
        frases = gestos_por_heuristica(respuesta)
    else:
        respuesta, origen, frases = await asyncio.to_thread(
            llm.responder, dicho, sesion["historial"], veredicto, decision, analisis)
    # La actitud filtra los gestos: una actitud seria no festeja, una calma no enfatiza
    for f in frases:
        f["gesto"] = gesto_permitido(f["gesto"], decision["actitud"])
        f["nombre"] = GESTOS[f["gesto"]]["nombre"]

    # 5) Voz de Ottoman
    archivo = await tts.sintetizar(respuesta, decision["voz"], voz_caricatura)

    sesion["historial"].append({"usuario": dicho, "robot": respuesta})
    sesion["emociones"].append(veredicto["emocion"])
    if cuerpo:
        cuerpo.fijar(modo="reposo", actitud=decision["actitud"], expresividad=decision["expresividad"])

    return {
        "entendido": True,
        "id": uuid.uuid4().hex[:8],
        "usuario": dicho,
        "respuesta": respuesta,
        "frases": frases,
        "caricatura": voz_caricatura,
        "origen_respuesta": origen,
        "audio": f"/audio/{archivo}" if archivo else None,
        "veredicto": veredicto,
        "decision": decision,
        "texto": {"origen": analisis["origen"], "registro": analisis.get("registro"), "trato": analisis.get("trato")},
        "voz_usuario": ({k: r_voz[k] for k in ("rasgos", "activacion", "fuente")} if r_voz else None),
        "cara_usuario": ({"cuadros": r_cara["cuadros"]} if r_cara else None),
        "llm_error": llm.ultimo_error,
        "segundos": round(time.time() - t0, 2),
    }


@app.post("/api/reiniciar")
def api_reiniciar():
    sesion["historial"].clear()
    sesion["emociones"].clear()
    voz.reiniciar()
    if cuerpo:
        cuerpo.fijar(modo="reposo", actitud="CURIOSA", expresividad=0.6)
    return {"ok": True}


# ------------------------------------------------------------------ cuerpo 3D
@app.post("/api/cuerpo")
async def api_cuerpo(request: Request):
    d = await request.json()
    if cuerpo:
        cuerpo.fijar(modo=d.get("modo"), actitud=d.get("actitud"), expresividad=d.get("expresividad"),
                     guion=d.get("guion"))
        return cuerpo.estado()
    return {"modo": None}


@app.get("/sim/stream")
async def sim_stream():
    if not cuerpo:
        raise HTTPException(503, "Simulador 3D no disponible")

    async def cuadros():
        ultimo = None
        while True:
            j = cuerpo.jpeg()
            if j and j is not ultimo:
                ultimo = j
                yield b"--cuadro\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n"
            await asyncio.sleep(1 / 30)

    return StreamingResponse(cuadros(), media_type="multipart/x-mixed-replace; boundary=cuadro")


@app.get("/sim/cuadro.jpg")
def sim_cuadro():
    if not cuerpo or not cuerpo.jpeg():
        raise HTTPException(503, "Simulador 3D no disponible")
    return Response(cuerpo.jpeg(), media_type="image/jpeg")


# ------------------------------------------------------------------ datos propios
@app.post("/api/voz_muestra")
async def api_voz_muestra(etiqueta: str = Form(...), audio: UploadFile = File(...)):
    r = await asyncio.to_thread(voz.analizar, await audio.read())
    if not r:
        raise HTTPException(400, "El audio es muy corto o no se escucha voz.")
    entrenamiento.agregar("voz", etiqueta, uuid.uuid4().hex[:8], r["columnas"], r["valores"])
    return {"ok": True, "datos": entrenamiento.resumen("voz")}


@app.post("/api/entrenar/{modalidad}")
async def api_entrenar(modalidad: str):
    if modalidad not in ("cara", "voz"):
        raise HTTPException(404)
    try:
        return await asyncio.to_thread(entrenamiento.entrenar, modalidad)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/borrar/{modalidad}")
def api_borrar(modalidad: str):
    if modalidad not in ("cara", "voz"):
        raise HTTPException(404)
    entrenamiento.borrar(modalidad)
    return {"ok": True}


# ------------------------------------------------------------------ archivos
@app.get("/audio/{nombre}")
def audio(nombre: str):
    ruta = tts.DIR_AUDIO / Path(nombre).name
    if not ruta.exists():
        raise HTTPException(404)
    return FileResponse(ruta, media_type="audio/mpeg")


@app.get("/")
def inicio():
    return FileResponse(RAIZ / "web" / "index.html", headers={"Cache-Control": "no-store"})


@app.exception_handler(Exception)
async def error_general(request: Request, exc: Exception):
    return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"}, status_code=500)


app.mount("/web", StaticFiles(directory=RAIZ / "web"), name="web")
