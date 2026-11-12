"""Lectura de la expresión facial del usuario.

MediaPipe Face Landmarker (red neuronal preentrenada) devuelve 52 "blendshapes": cuánto
está activo cada movimiento de la cara (sonrisa, ceño, cejas, párpados...). Sobre eso:
  1. Reglas base: funcionan sin entrenar nada.
  2. Modelo personal: clasificador entrenado con las expresiones del propio usuario.
La imagen se procesa en memoria y se descarta; nunca se guarda.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from .emociones import EMOCIONES, normalizar
from . import entrenamiento
from .entrenamiento import ModeloPersonal, cargar

RAIZ = Path(__file__).resolve().parent.parent
MODELO_MP = RAIZ / "modelos" / "face_landmarker.task"

# Rasgos legibles que se muestran al lado de la cámara
RASGOS = {
    "sonrisa": ("mouthSmileLeft", "mouthSmileRight"),
    "mejillas": ("cheekSquintLeft", "cheekSquintRight"),
    "ceño": ("browDownLeft", "browDownRight"),
    "cejas_arriba": ("browOuterUpLeft", "browOuterUpRight"),
    "cejas_internas": ("browInnerUp",),
    "boca_abajo": ("mouthFrownLeft", "mouthFrownRight"),
    "ojos_abiertos": ("eyeWideLeft", "eyeWideRight"),
    "ojos_entrecerrados": ("eyeSquintLeft", "eyeSquintRight"),
    "boca_abierta": ("jawOpen",),
    "labios_apretados": ("mouthPressLeft", "mouthPressRight"),
    "nariz_arrugada": ("noseSneerLeft", "noseSneerRight"),
    "boca_tensa": ("mouthStretchLeft", "mouthStretchRight"),
    "menton_arriba": ("mouthShrugLower",),
}
ETIQUETAS_RASGOS = {
    "sonrisa": "Sonrisa", "mejillas": "Mejillas elevadas", "ceño": "Ceño fruncido",
    "cejas_arriba": "Cejas levantadas", "cejas_internas": "Cejas en ángulo (pena)",
    "boca_abajo": "Comisuras hacia abajo", "ojos_abiertos": "Ojos muy abiertos",
    "ojos_entrecerrados": "Ojos entrecerrados", "boca_abierta": "Boca abierta",
    "labios_apretados": "Labios apretados", "nariz_arrugada": "Nariz arrugada",
    "boca_tensa": "Boca tensa", "menton_arriba": "Mentón apretado",
}


# Puntos del rostro que casi no se mueven al gesticular (contorno de ojos, nariz, frente,
# mentón, pómulos y costados). Sus distancias relativas son la "firma" de una cara.
PUNTOS_FIRMA = (33, 263, 133, 362, 1, 168, 6, 98, 327, 152, 10, 234, 454, 93, 323)
UMBRAL_MISMA_PERSONA = 0.026     # diferencia relativa media por debajo de la cual es la misma cara
UMBRAL_OTRA_PERSONA = 0.038      # por encima, se considera otra persona


def firma_de(puntos, ancho: int, alto: int) -> list[float] | None:
    """Distancias entre puntos fijos del rostro, en 3D y relativas a la separación de los ojos."""
    p = np.array([[puntos[i].x * ancho, puntos[i].y * alto, puntos[i].z * ancho] for i in PUNTOS_FIRMA])
    entre_ojos = float(np.linalg.norm(p[0] - p[1]))
    if entre_ojos < 20:              # cara demasiado chica en la imagen para medirla bien
        return None
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    i, j = np.triu_indices(len(PUNTOS_FIRMA), k=1)
    return [float(x) for x in d[i, j] / entre_ojos]


class AnalizadorCara:
    def __init__(self):
        self._det = None
        self._lock = threading.Lock()
        self._columnas: list[str] | None = None
        self.modelo = ModeloPersonal("cara")
        self.error: str | None = None
        self._ventana: deque = deque(maxlen=400)   # (t, probs, detectada)
        self._neutro: dict[str, float] | None = None
        self._neutro_t = 0.0
        # Reconocimiento de la persona
        self.reconocer = True
        self._firmas: deque = deque(maxlen=24)     # (t, firma) de los últimos cuadros de frente
        self._persona = {"estado": "sin_datos"}
        self._t_persona = 0.0
        self._racha = ("", 0)

    def _detector(self):
        if self._det is None:
            import mediapipe as mp
            from mediapipe.tasks import python as mpp
            from mediapipe.tasks.python import vision
            self._mp = mp
            opciones = vision.FaceLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=str(MODELO_MP)),
                output_face_blendshapes=True, num_faces=1,
                running_mode=vision.RunningMode.IMAGE)
            self._det = vision.FaceLandmarker.create_from_options(opciones)
        return self._det

    # ---------------------------------------------------------------- neutro personal
    def _linea_base(self) -> dict[str, float]:
        """Cara en reposo del usuario (promedio de sus muestras NEUTRO), si las capturó."""
        if self._neutro is None or time.time() - self._neutro_t > 10:
            self._neutro_t = time.time()
            self._neutro = {}
            etiquetas, _, X, columnas = cargar("cara")
            idx = [i for i, e in enumerate(etiquetas) if e == "NEUTRO"]
            if len(idx) >= 5:
                media = X[idx].mean(axis=0)
                self._neutro = {c: float(v) for c, v in zip(columnas, media)}
        return self._neutro

    # ---------------------------------------------------------------- análisis
    def analizar(self, jpeg: bytes, registrar: bool = True) -> dict:
        import cv2
        arr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            return {"detectada": False, "motivo": "imagen inválida"}
        rgb = np.ascontiguousarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
        try:
            with self._lock:
                det = self._detector()
                res = det.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            return {"detectada": False, "motivo": self.error}

        if not res.face_blendshapes:
            if registrar:
                self._ventana.append((time.time(), None, False))
            return {"detectada": False, "motivo": "no se ve una cara"}

        bs = {c.category_name: float(c.score) for c in res.face_blendshapes[0] if c.category_name != "_neutral"}
        if self._columnas is None:
            self._columnas = sorted(bs)
        valores = [bs.get(c, 0.0) for c in self._columnas]

        # Encuadre: caja de la cara e inclinación de la cabeza, para ayudar a posicionarse
        pts = res.face_landmarks[0]
        xs, ys = [p.x for p in pts], [p.y for p in pts]
        caja = [min(xs), min(ys), max(xs), max(ys)]
        ojo_i, ojo_d = pts[33], pts[263]
        inclinacion = math.degrees(math.atan2(ojo_d.y - ojo_i.y, ojo_d.x - ojo_i.x))
        alto = caja[3] - caja[1]
        if alto < 0.22:
            encuadre = "Acercate un poco"
        elif alto > 0.85:
            encuadre = "Alejate un poco"
        elif caja[0] < 0.03 or caja[2] > 0.97 or caja[1] < 0.02 or caja[3] > 0.98:
            encuadre = "Centrá la cara"
        else:
            encuadre = "Bien encuadrado"

        base = self._linea_base()
        rasgos = {}
        for nombre, claves in RASGOS.items():
            bruto = sum(bs.get(k, 0.0) for k in claves) / len(claves)
            reposo = sum(base.get(k, 0.0) for k in claves) / len(claves) if base else 0.0
            rasgos[nombre] = max(0.0, min(1.0, bruto - reposo))

        probs_reglas = self._reglas(rasgos)
        firma = firma_de(pts, arr.shape[1], arr.shape[0]) if abs(inclinacion) < 12 else None
        if firma:
            self._firmas.append((time.time(), firma))
        # Si la cara no es la del perfil activo, su modelo personal no aplica: quedan las reglas base
        desconocida = self._persona.get("estado") == "desconocido"
        probs_modelo = None if desconocida else self.modelo.predecir(valores)
        if probs_modelo:
            probs = normalizar({e: 0.75 * probs_modelo[e] + 0.25 * probs_reglas[e] for e in EMOCIONES})
            fuente = "modelo personal + reglas"
        else:
            probs, fuente = probs_reglas, "reglas base"
        if registrar:
            self._ventana.append((time.time(), probs, True))

        return {
            "detectada": True,
            "probs": {e: round(v, 3) for e, v in probs.items()},
            "rasgos": [{"clave": k, "nombre": ETIQUETAS_RASGOS[k], "valor": round(v, 3)} for k, v in rasgos.items()],
            "fuente": fuente,
            "caja": [round(v, 3) for v in caja],
            "inclinacion": round(inclinacion, 1),
            "encuadre": encuadre,
            "columnas": self._columnas,
            "valores": valores,
            "firma": firma,
        }

    @staticmethod
    def _reglas(r: dict[str, float]) -> dict[str, float]:
        puntajes = {
            "ALEGRIA": 1.7 * r["sonrisa"] + 0.5 * r["mejillas"],
            "TRISTEZA": 1.5 * r["boca_abajo"] + 1.0 * r["cejas_internas"] * (1 - r["sonrisa"])
                        + 0.5 * r["menton_arriba"] - 0.4 * r["ojos_abiertos"],
            "ENOJO": 1.6 * r["ceño"] + 0.8 * r["nariz_arrugada"] + 0.5 * r["labios_apretados"]
                     - 0.6 * r["cejas_internas"] - 0.5 * r["sonrisa"],
            "ANSIEDAD": 1.3 * r["ojos_abiertos"] + 0.7 * r["cejas_arriba"] + 0.7 * r["boca_tensa"]
                        + 0.3 * r["cejas_internas"] - 0.5 * r["sonrisa"],
            "NEUTRO": 0.42,
        }
        exps = {e: math.exp(4.5 * v) for e, v in puntajes.items()}
        return normalizar(exps)

    # ---------------------------------------------------------------- quién está frente a la cámara
    def firmas_recientes(self, segundos: float = 6.0) -> list:
        limite = time.time() - segundos
        return [f for t, f in self._firmas if t >= limite]

    def olvidar_persona(self) -> None:
        self._persona, self._racha, self._t_persona = {"estado": "sin_datos"}, ("", 0), 0.0

    def identificar(self) -> dict:
        """Compara la cara de los últimos segundos con la firma de cada perfil.

        Estados: reconocido | cambio (se pasó solo a otro perfil conocido) | desconocido |
        sin_registro (el perfil activo todavía no registró su cara) | sin_datos.
        """
        ahora = time.time()
        if ahora - self._t_persona < 1.5:
            return self._persona
        self._t_persona = ahora
        if self._persona.get("estado") == "cambio":
            self._persona = {"estado": "reconocido"}
        recientes = self.firmas_recientes()
        conocidas = entrenamiento.firmas()
        activo = entrenamiento.perfil_activo()
        if not self.reconocer or len(recientes) < 8:
            if not self.reconocer:
                self._persona = {"estado": "sin_datos"}
            return self._persona
        if activo not in conocidas and not conocidas:
            self._persona = {"estado": "sin_registro"}
            return self._persona
        actual = np.median(np.array(recientes, dtype=np.float32), axis=0)
        dist = {pid: float(np.mean(np.abs(actual - mu) / np.maximum(mu, 1e-3)))
                for pid, mu in conocidas.items() if len(mu) == len(actual)}
        mejor = min(dist, key=dist.get) if dist else None
        d_activo = dist.get(activo)
        if d_activo is not None and d_activo <= UMBRAL_MISMA_PERSONA:
            self._racha = ("", 0)
            self._persona = {"estado": "reconocido", "distancia": round(d_activo, 4)}
        elif mejor and mejor != activo and dist[mejor] <= UMBRAL_MISMA_PERSONA and \
                (d_activo is None or dist[mejor] < d_activo - 0.006):
            n = self._racha[1] + 1 if self._racha[0] == mejor else 1
            self._racha = (mejor, n)
            if n >= 2:                                # dos lecturas seguidas: es otro perfil conocido
                entrenamiento.activar_perfil(mejor)
                self._racha = ("", 0)
                self._persona = {"estado": "cambio", "distancia": round(dist[mejor], 4)}
        elif d_activo is not None and d_activo >= UMBRAL_OTRA_PERSONA:
            n = self._racha[1] + 1 if self._racha[0] == "?" else 1
            self._racha = ("?", n)
            if n >= 3:                                # tres lecturas seguidas: no es nadie conocido
                self._persona = {"estado": "desconocido", "distancia": round(d_activo, 4)}
        elif d_activo is None:
            self._persona = {"estado": "sin_registro"}
        return self._persona

    # ---------------------------------------------------------------- ventana temporal
    def resumen_ventana(self, segundos: float = 8.0) -> dict | None:
        """Promedio de la cara durante los últimos segundos (mientras la persona hablaba)."""
        limite = time.time() - segundos
        cuadros = [c for c in self._ventana if c[0] >= limite]
        con_cara = [c[1] for c in cuadros if c[2]]
        if len(con_cara) < 3:
            return None
        probs = {e: sum(p[e] for p in con_cara) / len(con_cara) for e in EMOCIONES}
        return {"probs": normalizar(probs), "calidad": len(con_cara) / max(1, len(cuadros)),
                "cuadros": len(con_cara)}
