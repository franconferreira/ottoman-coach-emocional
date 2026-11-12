"""Sistema de reglas que decide la ACTITUD del coach.

La actitud no la elige el LLM: la decide este motor de reglas, que es explicable y se
muestra en el dashboard. El LLM solo redacta la respuesta dentro de la actitud elegida.
La misma decisión gobierna los cuatro módulos de expresión: cuerpo, cara, léxico y voz.
"""
from __future__ import annotations

import re
import unicodedata

from .emociones import NEGATIVAS

# Voz: la identidad de la voz queda fija; la actitud solo la mueve un poco (ritmo, tono y
# "color"), para que Ottoman suene siempre como el mismo personaje.
ACTITUDES = {
    "CALMA": {
        "nombre": "Calma",
        "descripcion": "Baja un cambio y transmite que no hay apuro. Frases cortas y tranquilas.",
        "lexico": "Bajá un cambio: frases cortas y tranquilas, sin exclamaciones. Que se note que no hay apuro. "
                  "Podés invitar a respirar o a ir de a una cosa, dicho como amigo y no como un ejercicio.",
        "voz": {"velocidad": -6, "tono": -3, "volumen": -4},
        "timbre": "calido",
        "ojos": "calma",
    },
    "CALIDA": {
        "nombre": "Cálida",
        "descripcion": "Se queda al lado de la persona con lo que le pasa, sin apurarla ni buscarle soluciones.",
        "lexico": "Cercano y afectuoso, sin solemnidad. Quedate con lo que le pasa: sin apurar soluciones ni "
                  "buscarle el lado bueno.",
        "voz": {"velocidad": -4, "tono": -2, "volumen": -2},
        "timbre": "calido",
        "ojos": "calida",
    },
    "ALEGRE": {
        "nombre": "Alegre",
        "descripcion": "Festeja con ganas y con humor liviano lo que la persona cuenta.",
        "lexico": "Con entusiasmo y humor liviano: festejá con ganas lo que cuenta. Podés usar exclamaciones.",
        "voz": {"velocidad": 4, "tono": 3, "volumen": 3},
        "timbre": "brillante",
        "ojos": "alegre",
    },
    "SERIA": {
        "nombre": "Seria",
        "descripcion": "Se toma el enojo en serio: sin chistes, sin darle ni quitarle la razón.",
        "lexico": "Sin chistes ni entusiasmo, pero igual de amigo: bancá su enojo sin darle ni quitarle la razón "
                  "y sin sermonear. Frases directas.",
        "voz": {"velocidad": -2, "tono": -4, "volumen": 0},
        "timbre": "neutro",
        "ojos": "seria",
    },
    "CURIOSA": {
        "nombre": "Curiosa",
        "descripcion": "Interés genuino: deja la puerta abierta para que la persona cuente lo que quiera.",
        "lexico": "Liviano e interesado, como quien se sienta a charlar. No asumas cómo está: dejá la puerta "
                  "abierta para que cuente lo que quiera, a su ritmo.",
        "voz": {"velocidad": 0, "tono": 2, "volumen": 0},
        "timbre": "neutro",
        "ojos": "curiosa",
    },
    "CUIDADO": {
        "nombre": "Cuidado",
        "descripcion": "Protocolo de seguridad: deja el rol de coach y orienta hacia ayuda humana y profesional.",
        "lexico": "Mensaje fijo de contención y derivación. No lo redacta el LLM.",
        "voz": {"velocidad": -6, "tono": -3, "volumen": -2},
        "timbre": "calido",
        "ojos": "calida",
    },
}

# Reglas principales (se evalúan en orden; gana la primera que se cumple)
REGLAS = [
    {"id": "R1", "si": "Hay señales de riesgo para la persona", "entonces": "CUIDADO",
     "porque": "La seguridad está por encima de la conversación: se deriva a ayuda profesional."},
    {"id": "R2", "si": "Dice que está bien, pero la cara o la voz muestran otra cosa", "entonces": "CURIOSA",
     "porque": "Si las señales no coinciden, no se asume: se deja la puerta abierta para que cuente."},
    {"id": "R3", "si": "La confianza del veredicto es baja (menos de 40%)", "entonces": "CURIOSA",
     "porque": "Sin una lectura clara, lo responsable es no asumir cómo está la persona."},
    {"id": "R4", "si": "Emoción dominante: ansiedad", "entonces": "CALMA",
     "porque": "A una persona acelerada se la acompaña bajando el ritmo."},
    {"id": "R5", "si": "Emoción dominante: tristeza", "entonces": "CALIDA",
     "porque": "La tristeza pide contención y validación, no soluciones rápidas."},
    {"id": "R6", "si": "Emoción dominante: enojo", "entonces": "SERIA",
     "porque": "El enojo necesita ser tomado en serio; el humor o el entusiasmo lo empeoran."},
    {"id": "R7", "si": "Emoción dominante: alegría", "entonces": "ALEGRE",
     "porque": "Compartir la alegría la refuerza."},
    {"id": "R8", "si": "Emoción dominante: neutro", "entonces": "CURIOSA",
     "porque": "Sin una emoción marcada, se invita a contar más."},
]

# Reglas complementarias (se suman a la principal)
MODIFICADORES = [
    {"id": "M1", "si": "La persona pide diagnóstico, medicación o tratamiento",
     "entonces": "No dar consejo médico y sugerir consultar a un profesional"},
    {"id": "M2", "si": "Enojo o ansiedad muy intensos (más de 70%)",
     "entonces": "Menos gestos y voz más lenta: el robot baja la intensidad en vez de imitarla"},
    {"id": "M3", "si": "Tres turnos seguidos con una emoción negativa",
     "entonces": "Sugerir hablarlo también con alguien de confianza o un profesional"},
]

_PATRON_RIESGO = re.compile(
    r"suicid|matarme|me quiero matar|me voy a matar|quitarme la vida|sacarme la vida|no quiero vivir|"
    r"no quiero seguir viviendo|mejor muert|quiero morir|me quiero morir|hacerme dano|lastimarme|"
    r"autolesion|cortarme|terminar con todo|desaparecer para siempre|no vale la pena vivir"
)
_PATRON_MEDICO = re.compile(
    r"diagnostic|medicacion|medicamento|pastilla|antidepresiv|ansiolitic|clonazepam|rivotril|"
    r"alplax|dosis|que tomo|que puedo tomar|receta|tratamiento|tengo depresion\?|sera que tengo"
)


def _plano(texto: str) -> str:
    t = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def hay_riesgo(texto: str) -> bool:
    return bool(_PATRON_RIESGO.search(_plano(texto)))


def pide_consejo_medico(texto: str) -> bool:
    return bool(_PATRON_MEDICO.search(_plano(texto)))


MENSAJE_CUIDADO = (
    "Gracias por contarme esto, de verdad. Lo que estás sintiendo es importante y no tenés que cargarlo en soledad. "
    "Yo soy un robot y no puedo darte la ayuda que merecés en un momento así, pero hay personas que sí. "
    "Si estás en Argentina podés llamar al 135 desde Capital y Gran Buenos Aires, o al 0800 345 1435 desde todo el país: "
    "es el Centro de Asistencia al Suicida, atienden gratis. Si estás en peligro ahora, llamá al 911. "
    "¿Hay alguien de confianza a quien puedas avisarle o que pueda estar con vos en este momento?"
)


def decidir(veredicto: dict, texto: str, senales_llm: dict | None = None,
            historial_emociones: list[str] | None = None) -> dict:
    """Aplica las reglas y devuelve la actitud, la expresividad y la traza de la decisión."""
    senales_llm = senales_llm or {}
    emocion = veredicto["emocion"]
    confianza = veredicto["confianza"]
    intensidad = veredicto["intensidad"]
    riesgo = hay_riesgo(texto) or bool(senales_llm.get("riesgo"))

    condiciones = {
        "R1": riesgo,
        "R2": veredicto.get("incongruencia") is not None,
        "R3": confianza < 0.40,
        "R4": emocion == "ANSIEDAD",
        "R5": emocion == "TRISTEZA",
        "R6": emocion == "ENOJO",
        "R7": emocion == "ALEGRIA",
        "R8": emocion == "NEUTRO",
    }
    elegida = next(r for r in REGLAS if condiciones[r["id"]])
    actitud = elegida["entonces"]

    previas = (historial_emociones or [])[-2:]
    mods = {
        "M1": pide_consejo_medico(texto) or bool(senales_llm.get("pide_consejo_medico")),
        "M2": emocion in ("ENOJO", "ANSIEDAD") and intensidad > 0.70,
        "M3": emocion in NEGATIVAS and len(previas) == 2 and all(e in NEGATIVAS for e in previas),
    }

    # Expresividad: cuánto se mueve y cuánto varía la voz. Con alegría el robot acompaña
    # la intensidad de la persona; con enojo o ansiedad hace lo contrario (desescala).
    if actitud == "ALEGRE":
        expresividad = 0.55 + 0.45 * intensidad
    elif actitud in ("CALMA", "SERIA"):
        expresividad = 0.60 - 0.30 * intensidad
    elif actitud == "CUIDADO":
        expresividad = 0.30
    else:
        expresividad = 0.45 + 0.25 * intensidad
    if mods["M2"]:
        expresividad = min(expresividad, 0.30)
    expresividad = round(max(0.15, min(1.0, expresividad)), 2)

    perfil = ACTITUDES[actitud]
    escala = 0.6 + 0.8 * expresividad if actitud == "ALEGRE" else 1.0 + (0.4 if mods["M2"] else 0.0)
    voz = {k: round(v * escala) for k, v in perfil["voz"].items()}

    return {
        "actitud": actitud,
        "nombre": perfil["nombre"],
        "descripcion": perfil["descripcion"],
        "regla": elegida["id"],
        "porque": elegida["porque"],
        "expresividad": expresividad,
        "voz": voz,
        "ojos": perfil["ojos"],
        "timbre": perfil["timbre"],
        "lexico": perfil["lexico"],
        "riesgo": riesgo,
        "reglas": [{**r, "cumple": condiciones[r["id"]], "elegida": r["id"] == elegida["id"],
                    "actitud_nombre": ACTITUDES[r["entonces"]]["nombre"]} for r in REGLAS],
        "modificadores": [{**m, "activo": mods[m["id"]]} for m in MODIFICADORES],
    }


def tabla_reglas() -> dict:
    """Tabla estática para mostrar en el dashboard antes del primer turno."""
    return {
        "reglas": [{**r, "cumple": False, "elegida": False,
                    "actitud_nombre": ACTITUDES[r["entonces"]]["nombre"]} for r in REGLAS],
        "modificadores": [{**m, "activo": False} for m in MODIFICADORES],
        "actitudes": {k: {"nombre": v["nombre"], "descripcion": v["descripcion"], "voz": v["voz"]}
                      for k, v in ACTITUDES.items()},
    }
