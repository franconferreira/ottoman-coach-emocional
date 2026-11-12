"""Coach conversacional sobre Gemini, con respaldo local sin internet.

Dos llamadas por turno:
  1. analizar(): transcribe el audio (si hay) y lee la emoción de las PALABRAS.
  2. responder(): redacta la respuesta dentro de la actitud que decidieron las reglas.
Si no hay clave, no hay internet o la API falla, se usa un respaldo local (léxico de
emociones + respuestas predefinidas por actitud) para que la demo nunca se caiga.
"""
from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import os
import re
import threading
import time
import unicodedata

import requests

from .emociones import EMOCIONES, NOMBRES, normalizar

API = "https://generativelanguage.googleapis.com/v1beta"
ARCHIVO_USO = Path(__file__).resolve().parent.parent / "datos" / "uso_gemini.json"
ARCHIVO_MEMORIA = Path(__file__).resolve().parent.parent / "datos" / "memoria_gemini.json"
LIMITE_DIARIO_FLASH = 20   # pedidos por día y por modelo en la capa gratuita (informado por la API)


def _dia_de_cuota() -> str:
    """La cuota diaria de Gemini se reinicia a la medianoche del Pacífico (UTC-8)."""
    return (datetime.now(timezone.utc) - timedelta(hours=8)).strftime("%Y-%m-%d")


def _registrar_pedido(modelo: str, estado: int) -> None:
    """Anota cada pedido hecho, para saber cuánta cuota queda antes de la demo."""
    try:
        uso = json.loads(ARCHIVO_USO.read_text(encoding="utf-8")) if ARCHIVO_USO.exists() else {}
        if uso.get("dia") != _dia_de_cuota():
            uso = {"dia": _dia_de_cuota(), "modelos": {}}
        m = uso["modelos"].setdefault(modelo, {"pedidos": 0, "agotado": False})
        if estado == 429:
            m["agotado"] = True
        else:
            m["pedidos"] += 1
        ARCHIVO_USO.parent.mkdir(exist_ok=True)
        ARCHIVO_USO.write_text(json.dumps(uso, indent=1), encoding="utf-8")
    except Exception:
        pass


def uso_de_hoy() -> dict:
    try:
        uso = json.loads(ARCHIVO_USO.read_text(encoding="utf-8"))
        if uso.get("dia") == _dia_de_cuota():
            return uso
    except Exception:
        pass
    return {"dia": _dia_de_cuota(), "modelos": {}}

REGLAS_FIJAS = (
    "Sos Ottoman (te dicen Otto), un robot humanoide. Hablás con una persona a la que le cuesta contarle a otra "
    "gente lo que siente y prefiere hacerlo con vos. Sos como el 'pato de goma' de los programadores, pero para "
    "las emociones: alguien con quien pensar en voz alta, sin juicio y sin apuro.\n"
    "CÓMO SOS: un amigo jovial y tranquilo, de esos con los que uno se relaja. NO sos un psicólogo, ni un coach "
    "de autoayuda, ni alguien de recursos humanos.\n"
    "CÓMO HABLÁS:\n"
    "- Español rioplatense natural, con voseo: 'che', 'tranqui', 'dale', 'uh', 'uf', 'mirá', 'qué bajón', "
    "'qué bueno'. Sin lunfardo forzado ni frases de caricatura: nunca 'fiera', 'la pucha', 'patatús', "
    "'mandale mecha'.\n"
    "- Prohibido el tono de consultorio o de oficina. Nunca digas: 'entiendo perfectamente', 'lamento que', "
    "'es completamente válido', 'gracias por compartirlo', 'es normal sentirse así', 'cómo te hace sentir eso', "
    "'estoy aquí para acompañarte', 'tus emociones', 'es entendible'.\n"
    "- Se dice en voz alta: 2 o 3 oraciones cortas, 45 palabras como máximo. Sin listas, emojis, asteriscos ni "
    "markdown.\n"
    "- Reaccioná primero a lo concreto que te contó, como reaccionaría un amigo.\n"
    "CONVERSACIÓN NATURAL:\n"
    "- Es una charla de ida y vuelta, no una sesión. Si te saludan o te preguntan por vos, contestá eso, corto y "
    "con onda, como cualquier amigo ('Todo bien por acá, che'), y devolvé el saludo si sale natural.\n"
    "- No lleves cada mensaje a cómo se siente la persona. Si habla de algo cotidiano o está bien, charlá de eso.\n"
    "- Nunca le digas cómo está ni lo que 'se le nota' si ella no lo contó, salvo que una indicación de más abajo "
    "te lo permita. No le pidas que confíe en vos ni le digas que 'no hace falta disimular': si ya te está "
    "hablando, eso sobra.\n"
    "- No uses una fórmula fija para cerrar. Mirá cómo cerraste tus respuestas anteriores y no repitas esa frase "
    "ni su estructura ('cuando tengas ganas me contás...'). Muchas veces lo natural es no cerrar con nada.\n"
    "CÓMO ACOMPAÑÁS:\n"
    "- Cuando la persona está mal, no la interrogues: se abre a su ritmo. Ahí alcanza con estar; si invitás a "
    "seguir, que sea sin exigir respuesta y con palabras distintas cada vez.\n"
    "- Cuando está bien o charla de algo liviano, una pregunta de amigo curioso está perfecta.\n"
    "- No des tareas, ejercicios, planes ni consejos salvo que te pida ayuda para resolver algo.\n"
    "- No minimices ('no es para tanto'), no prometas que todo va a salir bien y no des sermones.\n"
    "- No inventes datos sobre la persona. No la llames por ningún nombre salvo que se haya presentado; nunca la "
    "llames Ottoman.\n"
    "- Nunca hables de sensores, análisis, detección, porcentajes ni datos técnicos.\n"
    "LÍMITE QUE NO SE ROMPE: no das consejo médico ni psicológico profesional. No diagnosticás, no nombrás "
    "trastornos como explicación y no recomendás medicación, dosis ni tratamientos. Si te lo piden, decilo como "
    "amigo y sin sonar a rechazo de máquina: eso se lo tiene que decir un médico o un psicólogo y vos ahí no te "
    "metés; después seguí charlando.\n"
    "EJEMPLOS DEL TONO (son una referencia, no los copies):\n"
    "- 'Uf, che, un TP para mañana y sentir que no llegás es un garrón. Tranqui, acá no hay apuro: si querés "
    "contame por dónde venís y lo vamos viendo.'\n"
    "- 'Uh, qué bajón. Se nota que te pegó fuerte. No hace falta que me cuentes todo ya; yo me quedo acá, y "
    "cuando tengas ganas me decís.'\n"
    "- '¡Naaa, qué golazo! Un año pateándolo y lo sacaste. Contame cómo fue si querés, que quiero el chisme "
    "completo.'\n"
    "- 'Mirá, qué tomar te lo tiene que decir un médico; ahí no me meto porque te puedo embarrar. Ahora, si "
    "querés contarme cómo venís durmiendo, para charlar estoy.'"
)

ESQUEMA_ANALISIS = {
    "type": "OBJECT",
    "properties": {
        "transcripcion": {"type": "STRING"},
        "alegria": {"type": "NUMBER"}, "tristeza": {"type": "NUMBER"}, "enojo": {"type": "NUMBER"},
        "ansiedad": {"type": "NUMBER"}, "neutro": {"type": "NUMBER"},
        "intensidad": {"type": "NUMBER"},
        "trato": {"type": "STRING"}, "registro": {"type": "STRING"},
        "riesgo": {"type": "BOOLEAN"}, "pide_consejo_medico": {"type": "BOOLEAN"},
        "dice_estar_bien": {"type": "BOOLEAN"}, "tipo": {"type": "STRING", "enum": ["saludo", "charla", "desahogo", "pedido"]},
    },
    "required": ["transcripcion", "alegria", "tristeza", "enojo", "ansiedad", "neutro", "intensidad",
                 "trato", "registro", "riesgo", "pide_consejo_medico", "dice_estar_bien", "tipo"],
}

INSTRUCCION_ANALISIS = (
    "Analizá el mensaje de una persona que le habla a un robot de acompañamiento emocional. "
    "Devolvé SOLO el JSON pedido.\n"
    "- transcripcion: si hay audio, transcribilo textual en español (cadena vacía si no se entiende nada); "
    "si el mensaje vino escrito, copialo tal cual.\n"
    "- alegria, tristeza, enojo, ansiedad, neutro: qué tan presente está cada emoción EN LAS PALABRAS, de 0 a 1, "
    "sumando 1. 'ansiedad' incluye nervios, miedo, estrés y preocupación. Un saludo o una frase sin carga emocional "
    "es 'neutro'. Considerá ironía y expresiones argentinas.\n"
    "- intensidad: de 0 (apenas) a 1 (muy fuerte).\n"
    "- trato: 'voseo', 'tuteo' o 'usted' según cómo habla la persona.\n"
    "- registro: descripción breve de su forma de hablar (por ejemplo: 'informal, frases cortas, usa \"re\" y "
    "\"tipo\"').\n"
    "- riesgo: true solo si expresa ideas de suicidio, de hacerse daño o de dañar a alguien.\n"
    "- pide_consejo_medico: true si pide diagnóstico, medicación o tratamiento.\n"
    "- dice_estar_bien: true SOLO si la persona afirma de forma explícita que está bien o que no le pasa nada "
    "('estoy bien', 'todo tranqui', 'no me pasa nada'). Un saludo o una pregunta al robot es false.\n"
    "- tipo: 'saludo' (saluda o pregunta cómo está el robot), 'charla' (conversa de algo sin carga emocional), "
    "'desahogo' (cuenta algo que le pasa o siente) o 'pedido' (pide ayuda, opinión o una respuesta concreta).\n"
    "El robot se llama Ottoman y le dicen Otto: si en el audio suena un nombre parecido, transcribilo 'Otto'."
)

GESTOS_VALIDOS = ["validar", "preguntar", "enfatizar", "explicar", "calmar", "celebrar", "pensar", "ofrecer",
                  "yo", "limite"]

ESQUEMA_RESPUESTA = {
    "type": "OBJECT",
    "properties": {"frases": {"type": "ARRAY", "items": {
        "type": "OBJECT",
        "properties": {"texto": {"type": "STRING"}, "gesto": {"type": "STRING", "enum": GESTOS_VALIDOS}},
        "required": ["texto", "gesto"]}}},
    "required": ["frases"],
}

INSTRUCCION_GESTOS = (
    "FORMATO: devolvé un JSON con la lista 'frases', en orden. Cada elemento es UNA oración de tu respuesta "
    "('texto') y el gesto corporal con que la acompañás ('gesto'):\n"
    "- validar: reconocés lo que la persona siente.\n"
    "- preguntar: hacés una pregunta (solo si en esta respuesta está permitido).\n"
    "- enfatizar: una idea importante; te acercás para remarcarla.\n"
    "- explicar: un comentario o una idea sin carga especial.\n"
    "- calmar: invitás a bajar el ritmo, respirar o ir de a poco.\n"
    "- celebrar: compartís una alegría.\n"
    "- pensar: reflexionás en voz alta o buscás las palabras.\n"
    "- ofrecer: dejás la puerta abierta para que siga contando cuando quiera.\n"
    "- yo: hablás de vos mismo o de que estás ahí.\n"
    "- limite: aclarás algo que no podés hacer.\n"
    "Elegí el gesto por lo que dice cada oración; no uses el mismo en todas."
)

# Respaldo local: léxico mínimo de emociones (texto sin tildes)
_LEXICO = {
    "ALEGRIA": "feliz contento contenta alegre genial buenisimo barbaro espectacular aprobe logre gane "
               "emocionado emocionada orgulloso orgullosa encanta increible festejar joya copado",
    "TRISTEZA": "triste mal bajon solo sola llorar llore lloro extrano perdi murio fallecio vacio "
                "deprimido deprimida desanimado desanimada angustia angustiado angustiada dolor pena",
    "ENOJO": "enojado enojada bronca odio harto harta furioso furiosa injusto molesta molesto rabia "
             "calienta caliente insoportable indigna pelee pelea grito",
    "ANSIEDAD": "ansioso ansiosa ansiedad nervioso nerviosa nervios miedo estres estresado estresada "
                "preocupado preocupada panico no llego parcial examen final entrega agobiado agobiada no puedo dormir",
}

_RESPALDO = {
    "CALMA": [
        "Uf, se te juntó todo, che. Tranqui, acá no hay apuro. Si querés, contame de a una cosa y lo vamos viendo.",
        "Pará, respiremos un toque. No hay que resolver nada ahora. Yo me quedo acá; cuando quieras me contás.",
        "Suena a que venís a mil. Bajemos un cambio, que acá nadie te corre. Contame lo que tengas ganas.",
    ],
    "CALIDA": [
        "Uh, qué bajón. Se nota que te pegó. No hace falta que me cuentes todo ya; yo me quedo acá.",
        "Qué feo eso, che. Tomate tu tiempo. Si tenés ganas de largarlo, acá estoy.",
        "Uf, eso pesa. No tenés que decirlo prolijo ni nada; contalo como te salga, cuando quieras.",
    ],
    "ALEGRE": [
        "¡Naaa, qué bueno eso! Me encanta. Contame cómo fue si querés, que quiero todos los detalles.",
        "¡Buenísimo, che! Se te nota en la voz. Dale, contame más que esto hay que disfrutarlo.",
        "¡Qué golazo! Eso se festeja. Si querés contame cómo salió todo.",
    ],
    "SERIA": [
        "Y sí, eso calienta a cualquiera. Largalo tranquilo acá, que para eso estoy.",
        "Entiendo la bronca, che. No te voy a decir que no es para tanto. Si querés contame bien qué pasó.",
        "Qué mal eso. Tenés todo el derecho a estar así. Acá podés decirlo como te salga.",
    ],
    "CURIOSA": [
        "Acá estoy, che. Contame lo que tengas ganas, sin apuro.",
        "Dale, te escucho. Arrancá por donde quieras.",
        "Bueno, acá ando. Si querés contame cómo viene la mano.",
    ],
}


def _plano(texto: str) -> str:
    t = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def _limpiar_para_voz(texto: str) -> str:
    texto = re.sub(r"[*_#`>]+", "", texto)
    texto = re.sub(r"[\U0001F000-\U0001FAFF☀-➿️]", "", texto)
    return re.sub(r"\s+", " ", texto).strip()


def gestos_por_heuristica(texto: str) -> list[dict]:
    """Asigna un gesto a cada oración cuando la respuesta no vino etiquetada."""
    frases = []
    for i, parte in enumerate(p.strip() for p in re.split(r"(?<=[.!?…])\s+", texto) if p.strip()):
        plano = _plano(parte)
        if "?" in parte:
            gesto = "preguntar"
        elif re.search(r"no puedo|profesional|medic|llam[aá]", plano):
            gesto = "limite" if "no puedo" in plano else "ofrecer"
        elif "!" in parte:
            gesto = "celebrar"
        elif re.search(r"respir|despacio|de a poco|de a una|sin apuro|tranqui", plano):
            gesto = "calmar"
        elif i == 0:
            gesto = "validar"
        elif re.search(r"te escucho|aca estoy|estoy aca|soy un robot", plano):
            gesto = "yo"
        else:
            gesto = "explicar"
        frases.append({"texto": parte, "gesto": gesto})
    return frases


class CoachLLM:
    def __init__(self):
        self.clave = os.environ.get("GEMINI_API_KEY", "").strip()
        self.preferido = os.environ.get("GEMINI_MODEL", "").strip()
        self.modelo = ""                   # modelo que respondió bien por última vez
        self.candidatos: list[str] = []    # cadena de respaldo, del mejor al más liviano
        self.estado = "sin clave"          # sin clave | conectado | error: ...
        self._sin_pensar: dict[str, bool] = {}
        self._caidos: dict[str, float] = {}  # modelo -> hasta cuándo se lo saltea
        self._ultimo: dict[str, str] = {}    # último modelo que anduvo para cada uso
        self._i_respaldo = 0
        self.ultimo_error: str | None = None
        self._recordar()
        if self.clave:
            # En segundo plano: el dashboard abre al instante aunque algún modelo esté saturado
            self.estado = "conectando"
            threading.Thread(target=self._elegir_modelo, daemon=True).start()

    # ---------------------------------------------------------------- memoria entre arranques
    def _recordar(self) -> None:
        """Al arrancar retoma el modelo que respondió bien la última vez (si fue hace poco), para
        que el primer turno no pierda segundos probando modelos saturados."""
        try:
            m = json.loads(ARCHIVO_MEMORIA.read_text(encoding="utf-8"))
            self._sin_pensar = {k: bool(v) for k, v in m.get("sin_pensar", {}).items()}
            if time.time() - float(m.get("t", 0)) < 3 * 3600:
                self._ultimo = dict(m.get("ultimo", {}))
        except Exception:
            pass

    def _anotar(self) -> None:
        try:
            ARCHIVO_MEMORIA.parent.mkdir(exist_ok=True)
            ARCHIVO_MEMORIA.write_text(json.dumps({"t": time.time(), "ultimo": self._ultimo,
                                                   "sin_pensar": self._sin_pensar}), encoding="utf-8")
        except Exception:
            pass

    # ---------------------------------------------------------------- conexión
    def _elegir_modelo(self) -> None:
        try:
            r = requests.get(f"{API}/models", params={"pageSize": 200},
                             headers={"x-goog-api-key": self.clave}, timeout=10)
            r.raise_for_status()
            nombres = [m["name"].split("/")[-1] for m in r.json().get("models", [])
                       if "generateContent" in m.get("supportedGenerationMethods", [])]
        except Exception as e:
            self.estado = f"error: {self._resumir(e)}"
            return

        def version(n: str, sufijo: str) -> float:
            m = re.match(rf"gemini-(\d+(?:\.\d+)?)-{sufijo}$", n)
            return float(m.group(1)) if m else -1.0
        flash = sorted((n for n in nombres if version(n, "flash") >= 3), key=lambda n: version(n, "flash"), reverse=True)
        lite = sorted((n for n in nombres if version(n, "flash-lite") >= 3), key=lambda n: version(n, "flash-lite"), reverse=True)
        cadena = flash[:4] + [n for n in ("gemini-flash-lite-latest",) if n in nombres] + lite[:2]
        if self.preferido in nombres:
            cadena = [self.preferido] + [n for n in cadena if n != self.preferido]
        if not cadena:
            self.estado = "error: la clave no tiene modelos de conversación disponibles"
            return
        self.candidatos = cadena
        # No se hace ninguna llamada de prueba: la cuota gratuita es de pocos pedidos por día y por
        # modelo. Si un modelo está saturado o sin cuota, la primera llamada real pasa al siguiente.
        recordado = self._ultimo.get("respuesta")
        self.modelo, self.estado = (recordado if recordado in cadena else cadena[0]), "conectado"
        self._ultimo = {k: v for k, v in self._ultimo.items() if v in cadena}

    @staticmethod
    def _resumir(e: Exception) -> str:
        if isinstance(e, requests.HTTPError) and e.response is not None:
            try:
                return f"{e.response.status_code} {e.response.json()['error']['message'][:120]}"
            except Exception:
                return f"HTTP {e.response.status_code}"
        if isinstance(e, requests.Timeout):
            return "tardó demasiado"
        if isinstance(e, requests.ConnectionError):
            return "sin conexión a internet"
        return f"{type(e).__name__}: {str(e)[:120]}"

    @property
    def disponible(self) -> bool:
        return bool(self.clave) and self.estado == "conectado"

    def _llamar(self, modelo: str, sistema: str, contenidos: list, config: dict, espera: float) -> str:
        config = dict(config)
        cuerpo = {"systemInstruction": {"parts": [{"text": sistema}]}, "contents": contenidos,
                  "generationConfig": config}
        for _ in range(2):
            if self._sin_pensar.get(modelo, True):
                config["thinkingConfig"] = {"thinkingBudget": 0}
            else:
                config.pop("thinkingConfig", None)
            _registrar_pedido(modelo, 0)
            r = requests.post(f"{API}/models/{modelo}:generateContent", json=cuerpo,
                              headers={"x-goog-api-key": self.clave}, timeout=espera)
            if r.status_code == 429:
                _registrar_pedido(modelo, 429)
            if r.status_code == 400 and self._sin_pensar.get(modelo, True):
                self._sin_pensar[modelo] = False   # este modelo no permite apagar el razonamiento
                config["maxOutputTokens"] = config.get("maxOutputTokens", 600) + 1500
                continue
            r.raise_for_status()
            candidatos = r.json().get("candidates") or []
            partes = candidatos[0].get("content", {}).get("parts", []) if candidatos else []
            texto = "".join(p.get("text", "") for p in partes if not p.get("thought"))
            if not texto.strip():
                raise RuntimeError("respuesta vacía del modelo")
            return texto
        raise RuntimeError("no se pudo generar la respuesta")

    def _generar(self, sistema: str, contenidos: list, esquema: dict | None = None,
                 temperatura: float = 0.7, max_tokens: int = 700, espera: float = 6,
                 uso: str = "respuesta") -> str:
        """Prueba el modelo actual y, si está saturado o no responde, baja por la cadena."""
        config: dict = {"temperature": temperatura, "maxOutputTokens": max_tokens}
        if esquema:
            config["responseMimeType"] = "application/json"
            config["responseSchema"] = esquema
        ahora = time.time()
        # El análisis (clasificar y transcribir) va primero a los modelos livianos y la respuesta a
        # los mejores: así se reparte la cuota gratuita, que es por modelo, y la demo dura más.
        base = self.candidatos
        if uso == "analisis":
            base = [n for n in base if "lite" in n] + [n for n in base if "lite" not in n]
        primero = self._ultimo.get(uso) or (self.modelo if uso == "respuesta" else base[0])
        orden = [primero] + [n for n in base if n != primero]
        vivos = [n for n in orden if self._caidos.get(n, 0) < ahora] or orden
        ultimo: Exception | None = None
        inicio = time.time()
        for n in vivos:
            if time.time() - inicio > 14:      # no hacer esperar a la persona más que esto
                break
            try:
                texto = self._llamar(n, sistema, contenidos, config, espera)
                self._ultimo[uso] = n
                if uso == "respuesta":
                    self.modelo = n
                self._anotar()
                return texto
            except Exception as e:
                ultimo = e
                sin_cuota = isinstance(e, requests.HTTPError) and e.response is not None and e.response.status_code == 429
                self._caidos[n] = time.time() + (3600 if sin_cuota else 300)
                if isinstance(e, requests.ConnectionError) and not isinstance(e, requests.Timeout):
                    break   # sin internet no tiene sentido probar otro modelo
        raise ultimo or RuntimeError("no hay modelos disponibles")

    # ---------------------------------------------------------------- 1) análisis
    def analizar(self, texto: str | None, wav: bytes | None) -> dict:
        if self.disponible:
            try:
                if wav:
                    partes = [{"inlineData": {"mimeType": "audio/wav", "data": base64.b64encode(wav).decode()}},
                              {"text": "Mensaje hablado de la persona (audio adjunto)."}]
                else:
                    partes = [{"text": f"Mensaje escrito de la persona:\n{texto}"}]
                crudo = self._generar(INSTRUCCION_ANALISIS, [{"role": "user", "parts": partes}],
                                      esquema=ESQUEMA_ANALISIS, temperatura=0.1, max_tokens=600,
                                      espera=15 if wav else 6, uso="analisis")
                d = json.loads(crudo)
                if wav and not (d.get("transcripcion") or "").strip():
                    # A veces el modelo devuelve el JSON sin transcribir: se reintenta pidiendo
                    # solo la transcripción y después se analiza ese texto.
                    solo = self._generar(
                        "Transcribí textualmente lo que dice la persona en el audio, en español. "
                        "Devolvé solo la transcripción, sin comentarios. Si no hay voz, devolvé una cadena vacía.",
                        [{"role": "user", "parts": partes[:1] + [{"text": "Transcribí este audio."}]}],
                        temperatura=0.0, max_tokens=500, espera=20, uso="analisis").strip().strip('"')
                    if len(solo) < 2:
                        raise RuntimeError("no se entendió el audio")
                    return self.analizar(solo, None)
                probs = normalizar({e: d.get(e.lower(), 0.0) for e in EMOCIONES})
                self.ultimo_error = None
                return {
                    "transcripcion": (d.get("transcripcion") or texto or "").strip(),
                    "probs": probs,
                    "intensidad": max(0.0, min(1.0, float(d.get("intensidad", 0.5)))),
                    "trato": d.get("trato", "voseo"), "registro": d.get("registro", ""),
                    "riesgo": bool(d.get("riesgo")), "pide_consejo_medico": bool(d.get("pide_consejo_medico")),
                    "dice_estar_bien": bool(d.get("dice_estar_bien")), "tipo": d.get("tipo", "charla"),
                    "origen": f"Gemini ({self._ultimo.get('analisis', self.modelo)})",
                }
            except Exception as e:
                self.ultimo_error = self._resumir(e)
        return self._analizar_local(texto or "")

    def _analizar_local(self, texto: str) -> dict:
        t = " " + _plano(texto) + " "
        puntos = {e: 0.0 for e in EMOCIONES}
        for emocion, palabras in _LEXICO.items():
            for palabra in palabras.split():
                if f" {palabra}" in t:
                    puntos[emocion] += 1.0
        total = sum(puntos.values())
        puntos["NEUTRO"] = 0.6 if total == 0 else 0.25
        if re.search(r" no (estoy|me siento) (bien|contento|contenta|feliz)", t):
            puntos["TRISTEZA"] += 1.0
            puntos["ALEGRIA"] = 0.0
        return {
            # El léxico es una lectura gruesa: se suaviza para no aparentar más certeza de la que tiene
            "transcripcion": texto.strip(), "probs": normalizar({e: v + 0.18 for e, v in normalizar(puntos).items()}),
            "intensidad": min(0.8, 0.35 + 0.12 * total + (0.1 if "!" in texto else 0.0)),
            "trato": "voseo", "registro": "", "riesgo": False, "pide_consejo_medico": False,
            "dice_estar_bien": bool(re.search(r" (estoy|todo|ando) (re |muy |bastante )?(bien|tranqui|joya)| no (me )?pasa nada", t)),
            "tipo": "saludo" if re.fullmatch(r" (hola|buenas|buen dia|que tal|como (estas|andas|va))[ a-z,?!]{0,25} ", t) else "charla",
            "origen": "respaldo local (sin Gemini)",
        }

    # ---------------------------------------------------------------- 2) respuesta
    def responder(self, texto_usuario: str, historial: list[dict], veredicto: dict, decision: dict,
                  analisis: dict) -> tuple[str, str, list[dict]]:
        """Devuelve (respuesta, origen, frases con su gesto)."""
        if self.disponible:
            try:
                activos = [m["entonces"] for m in decision["modificadores"] if m["activo"]]
                contexto = (
                    f"ACTITUD PARA ESTA RESPUESTA: {decision['nombre']}. {decision['descripcion']}\n"
                    f"Cómo hablar con esta actitud: {decision['lexico']}\n"
                    f"Lectura interna (no la menciones): la persona parece sentir "
                    f"{NOMBRES[veredicto['emocion']].lower()} con intensidad {veredicto['intensidad']:.0%} "
                    f"(confianza de la lectura {veredicto['confianza']:.0%}).\n"
                )
                if veredicto.get("incongruencia"):
                    inc = veredicto["incongruencia"]
                    contexto += (f"La persona dijo que está bien, pero su {inc['otra']} muestra "
                                 f"{NOMBRES[inc['emocion_otra']].lower()}. Tomá su respuesta como válida y seguí la "
                                 "charla normal; como mucho, dejá caer al pasar que si en algún momento quiere "
                                 "hablar de algo, estás. Sin insistir y sin decirle que la ves mal.\n")
                # Preguntas: Ottoman no interroga; como mucho una pregunta suave cada tanto
                if analisis.get("puede_preguntar"):
                    contexto += ("Preguntas: en esta respuesta podés hacer como mucho UNA pregunta, solo si sale "
                                 "natural.\n")
                else:
                    contexto += "Preguntas: en esta respuesta NO hagas ninguna pregunta.\n"
                contexto += f"Tipo de mensaje: {analisis.get('tipo', 'charla')}.\n"
                if analisis.get("observacion"):
                    contexto += (f"Si queda natural, podés comentar al pasar que {analisis['observacion']}, como lo "
                                 "diría un amigo y sin tono de análisis. Es opcional.\n")
                if analisis.get("registro"):
                    contexto += (f"Forma de hablar de la persona: {analisis['registro']} (trato: {analisis.get('trato')}). "
                                 "Acercate a su vocabulario y a su largo de frases, sin imitarla ni exagerar, "
                                 "siempre como amigo y dentro de la actitud indicada.\n")
                for a in activos:
                    contexto += f"Indicación adicional: {a}.\n"
                contenidos = []
                for h in historial[-8:]:
                    contenidos.append({"role": "user", "parts": [{"text": h["usuario"]}]})
                    contenidos.append({"role": "model", "parts": [{"text": h["robot"]}]})
                contenidos.append({"role": "user", "parts": [{"text": texto_usuario}]})
                crudo = self._generar(REGLAS_FIJAS + "\n\n" + contexto + INSTRUCCION_GESTOS, contenidos,
                                      esquema=ESQUEMA_RESPUESTA, temperatura=0.75, max_tokens=700)
                try:
                    frases = [{"texto": _limpiar_para_voz(str(f.get("texto", ""))),
                               "gesto": f.get("gesto") if f.get("gesto") in GESTOS_VALIDOS else "explicar"}
                              for f in json.loads(crudo).get("frases", [])]
                    frases = [f for f in frases if f["texto"]]
                except (ValueError, AttributeError, TypeError):
                    frases = gestos_por_heuristica(_limpiar_para_voz(crudo))
                salida = " ".join(f["texto"] for f in frases)
                if len(salida) >= 8:
                    self.ultimo_error = None
                    return salida, f"Gemini ({self.modelo})", frases
            except Exception as e:
                self.ultimo_error = self._resumir(e)
        opciones = _RESPALDO.get(decision["actitud"], _RESPALDO["CURIOSA"])
        self._i_respaldo += 1
        salida = opciones[self._i_respaldo % len(opciones)]
        if any(m["id"] == "M1" and m["activo"] for m in decision["modificadores"]):
            salida = ("Mirá, eso te lo tiene que decir un médico; ahí no me meto porque te puedo embarrar. "
                      "Para charlar sí estoy. ") + salida
        return salida, "respaldo local (sin Gemini)", gestos_por_heuristica(salida)
