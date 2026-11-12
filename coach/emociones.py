"""Vocabulario común de emociones y fusión multimodal del veredicto."""
from __future__ import annotations

EMOCIONES = ["ALEGRIA", "TRISTEZA", "ENOJO", "ANSIEDAD", "NEUTRO"]
NEGATIVAS = {"TRISTEZA", "ENOJO", "ANSIEDAD"}

NOMBRES = {
    "ALEGRIA": "Alegría",
    "TRISTEZA": "Tristeza",
    "ENOJO": "Enojo",
    "ANSIEDAD": "Ansiedad",
    "NEUTRO": "Neutro",
}

# Peso de cada señal en el veredicto. El texto pesa más porque es lo que la persona
# decide contar; la cara aporta lo que no se dice; la voz sola distingue bien la
# activación (agitado / apagado) pero no si la emoción es agradable o no.
PESOS = {"texto": 0.50, "cara": 0.30, "voz": 0.20}


def normalizar(probs: dict[str, float]) -> dict[str, float]:
    limpio = {e: max(0.0, float(probs.get(e, 0.0))) for e in EMOCIONES}
    total = sum(limpio.values())
    if total <= 0:
        return {e: 1.0 / len(EMOCIONES) for e in EMOCIONES}
    return {e: v / total for e, v in limpio.items()}


def dominante(probs: dict[str, float]) -> tuple[str, float]:
    e = max(EMOCIONES, key=lambda k: probs.get(k, 0.0))
    return e, float(probs.get(e, 0.0))


def fusionar(senales: dict[str, dict | None], intensidad_texto: float | None = None,
             activacion_voz: float | None = None, afirma_estar_bien: bool = False) -> dict:
    """Combina las señales disponibles en un veredicto único.

    `senales` mapea modalidad -> {"probs": {...}, "calidad": 0..1} o None si no está.
    Si falta cámara o micrófono, los pesos se reparten entre lo que queda.
    """
    usadas, pesos = {}, {}
    for modalidad, s in senales.items():
        if not s or not s.get("probs"):
            continue
        calidad = float(s.get("calidad", 1.0))
        if calidad <= 0.05:
            continue
        usadas[modalidad] = normalizar(s["probs"])
        pesos[modalidad] = PESOS[modalidad] * calidad
    if not usadas:
        probs = {e: (1.0 if e == "NEUTRO" else 0.0) for e in EMOCIONES}
        return {"emocion": "NEUTRO", "confianza": 0.0, "intensidad": 0.3, "probs": probs,
                "pesos": {}, "por_modalidad": {}, "incongruencia": None}
    total = sum(pesos.values())
    pesos = {m: p / total for m, p in pesos.items()}
    probs = {e: sum(pesos[m] * usadas[m][e] for m in usadas) for e in EMOCIONES}
    emocion, p_top = dominante(probs)
    segunda = sorted(probs.values(), reverse=True)[1]
    # Confianza: qué tan clara es la emoción ganadora frente a la segunda
    confianza = max(0.0, min(1.0, 0.55 * p_top + 0.45 * min(1.0, (p_top - segunda) * 2.5)))

    por_modalidad = {}
    for m, p in usadas.items():
        e, v = dominante(p)
        por_modalidad[m] = {"emocion": e, "prob": round(v, 3), "peso": round(pesos[m], 3),
                            "probs": {k: round(x, 3) for k, x in p.items()}}

    # Incongruencia ("enmascaramiento"): la persona AFIRMA que está bien, pero la cara o la voz
    # muestran con claridad otra cosa. Un saludo o una pregunta no dicen nada de cómo está:
    # ahí no hay nada que contradecir, y una lectura dudosa de la cara no alcanza.
    incongruencia = None
    if "texto" in usadas and afirma_estar_bien:
        e_txt, p_txt = dominante(usadas["texto"])
        for otra in ("cara", "voz"):
            if otra not in usadas:
                continue
            e_o, p_o = dominante(usadas[otra])
            if e_txt not in NEGATIVAS and e_o in NEGATIVAS and p_o >= 0.70:
                incongruencia = {"texto": e_txt, "otra": otra, "emocion_otra": e_o}
                break

    intensidades = [p_top]
    if intensidad_texto is not None:
        intensidades.append(float(intensidad_texto))
    if activacion_voz is not None and emocion != "NEUTRO":
        intensidades.append(0.5 + 0.5 * abs(float(activacion_voz)))
    intensidad = sum(intensidades) / len(intensidades)
    if emocion == "NEUTRO":
        intensidad = min(intensidad, 0.4)

    return {
        "emocion": emocion,
        "confianza": round(confianza, 3),
        "intensidad": round(max(0.0, min(1.0, intensidad)), 3),
        "probs": {e: round(v, 3) for e, v in probs.items()},
        "pesos": {m: round(p, 3) for m, p in pesos.items()},
        "por_modalidad": por_modalidad,
        "incongruencia": incongruencia,
    }
