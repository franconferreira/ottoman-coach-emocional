"""Módulo de cuerpo: lenguaje corporal del G1 en MuJoCo, gobernado por la actitud del coach.

El robot se anima de forma cinemática (sin física) sobre el modelo G1 del simulador de la
cátedra. El lenguaje corporal tiene cuatro capas:

  1. Postura de la actitud: cómo se para (inclinado, erguido, ladeado).
  2. Gesto de la frase: cada frase de la respuesta trae un gesto con significado
     (validar, preguntar, enfatizar...). Usa todo el cuerpo: brazos, torso, cadera,
     rodillas y tobillos, no solo los brazos.
  3. Ritmo: acentos al hablar, quietud en las pausas.
  4. Vida: respiración y cambio de peso, siempre activos.

El G1 no tiene articulación en el cuello: la "mirada" se expresa girando e inclinando el
torso (y con los ojos del módulo de cara).
"""
from __future__ import annotations

import math
import random
import threading
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
ESCENA_G1 = RAIZ / "entorno" / "sim" / "unitree_mujoco" / "unitree_robots" / "g1" / "scene_29dof.xml"

# --------------------------------------------------------------------------------------
# Brazos. Se describen para el lado IZQUIERDO como
# (hombro_pitch, hombro_roll, hombro_yaw, codo, muñeca_roll, muñeca_pitch, muñeca_yaw);
# el derecho se obtiene espejando roll y yaw.
#   hombro_pitch < 0 levanta el brazo hacia adelante | hombro_roll > 0 lo abre hacia afuera
#   codo = 0 antebrazo horizontal | codo > 0 baja | codo < 0 sube
# --------------------------------------------------------------------------------------
BRAZO_REPOSO = (0.28, 0.16, 0.05, 1.05, 0.25, 0.0, 0.0)
BRAZO_ESCUCHA = (0.10, 0.10, -0.42, 0.62, 0.35, 0.0, 0.0)       # mano al frente, baja
BRAZO_ABIERTO = (-0.10, 0.42, 0.55, 0.42, -1.25, 0.0, 0.0)      # brazo abierto, palma arriba
BRAZO_PECHO = (-0.42, 0.02, -1.05, -0.35, 0.3, 0.0, 0.0)        # mano al pecho
BRAZO_CALMA_ALTO = (-0.30, 0.20, 0.05, 0.10, 1.35, -0.25, 0.0)  # palma abajo, arriba
BRAZO_ARRIBA = (-1.30, 0.45, 0.15, -0.35, -0.3, 0.0, 0.0)       # brazo en alto (festejo)
BRAZO_ENFASIS = (-0.45, 0.28, 0.30, -0.10, -0.6, 0.0, 0.0)      # antebrazo arriba, mano abierta
BRAZO_PREGUNTA = (-0.30, 0.30, 0.75, -0.25, -1.3, 0.0, 0.0)     # palma arriba, al costado
BRAZO_MARCA = (-0.25, 0.10, -0.05, 0.25, 0.0, 0.0, 0.0)         # mano al frente que "marca" ideas
BRAZO_PENSAR = (-0.50, 0.05, -0.80, -0.62, 0.2, 0.0, 0.0)       # mano hacia el mentón
BRAZO_ALTO = (-0.40, 0.22, 0.10, -0.50, 0.0, -0.3, 0.0)         # palma al frente ("hasta acá")
BRAZO_SALUDO = (-1.9, 0.35, 0.0, -0.55, 0.0, 0.0, 0.0)

_NOMBRES_BRAZO = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                  "wrist_roll", "wrist_pitch", "wrist_yaw")
_ESPEJO = (1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
_DEDOS = ("index", "middle", "ring", "pinky")

# Canales de cuerpo entero (no son articulaciones: se traducen a piernas y pelvis)
#   pelvis_pitch: inclina todo el tronco desde la cadera (+ hacia adelante)
#   pelvis_roll:  ladea la pelvis
#   cadera:       lleva el peso hacia una pierna (las piernas se inclinan, los pies no se mueven)
#   agachar:      flexiona rodillas y tobillos, bajando el cuerpo
CANALES = ("pelvis_pitch", "pelvis_roll", "cadera", "agachar")
AMPLITUD_CUERPO = 1.6
_PIERNA_BASE = {"hip_pitch": -0.10, "knee": 0.22, "ankle_pitch": -0.12}


def _brazo(lado: str, valores: tuple) -> dict[str, float]:
    signo = (1.0,) * 7 if lado == "left" else _ESPEJO
    return {f"{lado}_{n}": v * s for n, v, s in zip(_NOMBRES_BRAZO, valores, signo)}


def _pose(izq: tuple, der: tuple, dedos: float = 0.3, **cuerpo: float) -> dict[str, float]:
    p = {"waist_pitch": 0.0, "waist_roll": 0.0, "waist_yaw": 0.0, **{c: 0.0 for c in CANALES}}
    p.update(_brazo("left", izq))
    p.update(_brazo("right", der))
    for lado in ("left", "right"):
        for d in _DEDOS:
            p[f"{lado}_hand_{d}"] = dedos
    # El cuerpo (torso, cadera, rodillas) se exagera un poco: en un robot los gestos chicos no se leen
    p.update({k: v * AMPLITUD_CUERPO for k, v in cuerpo.items()})
    return p


# --------------------------------------------------------------------------------------
# Gestos con significado. Ottoman es diestro: los gestos de una mano los hace con la
# derecha mientras la izquierda descansa. "anim" agrega un movimiento propio del gesto.
# --------------------------------------------------------------------------------------
GESTOS = {
    "apertura": {"pose": _pose(BRAZO_REPOSO, BRAZO_MARCA, waist_pitch=0.03, agachar=0.07), "anim": "asentir",
                 "nombre": "Arranca"},
    "validar": {"pose": _pose(BRAZO_REPOSO, BRAZO_PECHO, 0.15, waist_pitch=0.08, pelvis_pitch=0.05,
                              waist_roll=0.04, cadera=0.035, agachar=0.05), "anim": "asentir", "nombre": "Valida"},
    "preguntar": {"pose": _pose(BRAZO_REPOSO, BRAZO_PREGUNTA, 0.05, waist_roll=-0.09, waist_pitch=0.06,
                                pelvis_pitch=0.05, cadera=-0.05, waist_yaw=0.06, agachar=0.04), "anim": None,
                  "nombre": "Pregunta"},
    "enfatizar": {"pose": _pose(BRAZO_MARCA, BRAZO_MARCA, 0.0, waist_pitch=0.16, pelvis_pitch=0.11, agachar=0.26),
                  "anim": "golpe", "nombre": "Enfatiza"},
    "explicar": {"pose": _pose(BRAZO_REPOSO, BRAZO_ENFASIS, 0.05, waist_yaw=-0.07, waist_pitch=0.02, cadera=0.03),
                 "anim": None, "nombre": "Explica"},
    "calmar": {"pose": _pose(BRAZO_CALMA_ALTO, BRAZO_CALMA_ALTO, 0.05, waist_pitch=0.04, agachar=0.05),
               "anim": "bajar", "nombre": "Calma"},
    "celebrar": {"pose": _pose(BRAZO_ENFASIS, BRAZO_ARRIBA, 0.0, waist_pitch=-0.05, pelvis_pitch=-0.03,
                               waist_roll=-0.04, cadera=0.04), "anim": "rebote", "nombre": "Celebra"},
    "pensar": {"pose": _pose(BRAZO_REPOSO, BRAZO_PENSAR, 0.35, waist_yaw=-0.22, waist_pitch=-0.09,
                             pelvis_pitch=-0.03, waist_roll=0.05, cadera=0.06), "anim": None, "nombre": "Piensa"},
    "ofrecer": {"pose": _pose(BRAZO_ABIERTO, BRAZO_ABIERTO, 0.05, waist_pitch=0.10, pelvis_pitch=0.07, agachar=0.10),
                "anim": None, "nombre": "Ofrece"},
    "yo": {"pose": _pose(BRAZO_REPOSO, BRAZO_PECHO, 0.15, waist_pitch=0.01, cadera=-0.03), "anim": None,
           "nombre": "Habla de sí"},
    "limite": {"pose": _pose(BRAZO_ALTO, BRAZO_ALTO, 0.0, waist_pitch=-0.02, pelvis_pitch=-0.02), "anim": "negar",
               "nombre": "Pone un límite"},
}

# Cómo se para Ottoman en cada actitud (se suma a cualquier gesto)
POSTURA = {
    "CALMA":   {"waist_pitch": 0.02, "agachar": 0.03},
    "CALIDA":  {"waist_pitch": 0.05, "pelvis_pitch": 0.04, "cadera": 0.03},
    "ALEGRE":  {"waist_pitch": -0.03, "pelvis_pitch": -0.02},
    "SERIA":   {},
    "CURIOSA": {"waist_roll": -0.05, "cadera": -0.03, "waist_pitch": 0.03},
    "CUIDADO": {"waist_pitch": 0.07, "pelvis_pitch": 0.05, "agachar": 0.05},
}

# Gestos que una actitud no permite, y con cuál se reemplazan (igual que los ojos: una
# actitud seria no festeja)
SUSTITUCION = {
    "CALMA":   {"celebrar": "validar", "enfatizar": "calmar", "limite": "explicar"},
    "CALIDA":  {"celebrar": "validar", "enfatizar": "ofrecer"},
    "ALEGRE":  {"calmar": "explicar"},
    "SERIA":   {"celebrar": "explicar"},
    "CURIOSA": {"celebrar": "explicar"},
    "CUIDADO": {"celebrar": "validar", "enfatizar": "calmar", "pensar": "validar", "limite": "ofrecer"},
}

# Si la respuesta no trae gestos por frase, cada actitud tiene un repertorio propio
REPERTORIO = {
    "CALMA":   ["calmar", "validar", "explicar", "ofrecer"],
    "CALIDA":  ["validar", "ofrecer", "yo", "preguntar"],
    "ALEGRE":  ["celebrar", "explicar", "enfatizar", "ofrecer"],
    "SERIA":   ["explicar", "validar", "limite", "enfatizar"],
    "CURIOSA": ["preguntar", "pensar", "explicar", "ofrecer"],
    "CUIDADO": ["validar", "calmar", "ofrecer"],
}

# Dinámica por actitud: rapidez del movimiento (resorte), amplitud del acento al hablar
DINAMICA = {
    "CALMA":   dict(dur=3.6, vel=2.8, beat=0.020),
    "CALIDA":  dict(dur=3.0, vel=3.6, beat=0.035),
    "ALEGRE":  dict(dur=1.9, vel=7.0, beat=0.085),
    "SERIA":   dict(dur=3.2, vel=4.4, beat=0.028),
    "CURIOSA": dict(dur=2.6, vel=4.8, beat=0.045),
    "CUIDADO": dict(dur=4.0, vel=2.5, beat=0.015),
}

POSE_REPOSO = _pose(BRAZO_REPOSO, BRAZO_REPOSO, cadera=0.025, pelvis_roll=0.01)
# Escuchar: peso en una pierna, ladeado e inclinado hacia la persona (gesto propio de Ottoman)
POSE_ESCUCHA = _pose(BRAZO_ESCUCHA, BRAZO_ESCUCHA, 0.25, waist_pitch=0.08, pelvis_pitch=0.05,
                     waist_roll=-0.06, cadera=-0.06, pelvis_roll=-0.02, agachar=0.05)
POSE_PENSANDO = GESTOS["pensar"]["pose"]
POSE_SALUDO = _pose(BRAZO_REPOSO, BRAZO_SALUDO, 0.0, waist_roll=-0.03, cadera=0.04)


def gesto_permitido(gesto: str, actitud: str) -> str:
    """Devuelve el gesto que corresponde hacer en esa actitud."""
    if gesto not in GESTOS:
        gesto = "explicar"
    return SUSTITUCION.get(actitud, {}).get(gesto, gesto)


class AnimadorCuerpo:
    """Calcula la pose del robot en cada instante y la renderiza a JPEG."""

    def __init__(self, ancho: int = 640, alto: int = 560, fps: int = 30):
        import mujoco

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(ESCENA_G1))
        self.data = mujoco.MjData(self.model)
        self.ancho, self.alto, self.fps = ancho, alto, fps
        # El framebuffer fuera de pantalla del modelo es de 640x480 por defecto
        self.model.vis.global_.offwidth = max(int(self.model.vis.global_.offwidth), ancho)
        self.model.vis.global_.offheight = max(int(self.model.vis.global_.offheight), alto)

        self.jmap: dict[str, int] = {}
        for i in range(self.model.njnt):
            nombre = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if nombre and nombre.endswith("_joint"):
                self.jmap[nombre[:-6]] = int(self.model.jnt_qposadr[i])
        self._tobillos = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{lado}_ankle_roll_link")
                          for lado in ("left", "right")]

        # Estado expresivo (lo fija el servidor según la decisión del coach)
        self._lock = threading.Lock()
        self.modo = "reposo"            # reposo | escuchando | pensando | hablando | saludo
        self.actitud = "CURIOSA"
        self.expresividad = 0.6
        self._t_modo = time.time()
        self._guion: list[tuple[float, str]] = []   # (segundo desde que empieza a hablar, gesto)
        self._rng = random.Random(7)

        # Filtro de segundo orden por canal (movimiento con aceleración suave)
        self._pos: dict[str, float] = dict(POSE_REPOSO)
        self._vel: dict[str, float] = {k: 0.0 for k in self._pos}
        self._t_prev = time.time()
        self._aplicar(dict(POSE_REPOSO), fijar_referencia=True)

        self._jpeg: bytes = b""
        self._hilo: threading.Thread | None = None
        self._activo = False
        self.error: str | None = None

    # ------------------------------------------------------------------ estado
    def fijar(self, modo: str | None = None, actitud: str | None = None,
              expresividad: float | None = None, guion: list | None = None) -> None:
        with self._lock:
            if actitud and actitud in POSTURA:
                self.actitud = actitud
            if expresividad is not None:
                self.expresividad = float(np.clip(expresividad, 0.15, 1.0))
            if guion is not None:
                self._guion = sorted((float(g["t"]), gesto_permitido(str(g["gesto"]), self.actitud)) for g in guion)
            if modo and (modo != self.modo or guion is not None):
                self.modo = modo
                self._t_modo = time.time()
                if modo != "hablando":
                    self._guion = []

    def estado(self) -> dict:
        return {"modo": self.modo, "actitud": self.actitud, "expresividad": round(self.expresividad, 2)}

    def _guion_automatico(self, actitud: str, expr: float) -> list[tuple[float, str]]:
        dur = DINAMICA[actitud]["dur"] * (1.25 - 0.45 * expr)
        rep = REPERTORIO[actitud]
        return [(0.0, "apertura")] + [(0.45 + i * dur, rep[i % len(rep)]) for i in range(40)]

    # ------------------------------------------------------------------ pose objetivo
    def _objetivo(self, t: float) -> tuple[dict[str, float], float]:
        """Pose objetivo y rapidez del resorte para el instante t."""
        with self._lock:
            modo, actitud, expr = self.modo, self.actitud, self.expresividad
            t_modo = t - self._t_modo
            guion = list(self._guion)
        din = DINAMICA[actitud]
        postura = POSTURA[actitud]
        if modo == "hablando":
            if not guion:
                guion = self._guion_automatico(actitud, expr)
            i = max((k for k, (tg, _) in enumerate(guion) if tg <= t_modo), default=0)
            t_gesto = t_modo - guion[i][0]
            gesto = GESTOS[guion[i][1]]
            # La expresividad escala cuánto se aleja el gesto de la postura de reposo
            k = 0.5 + 0.5 * expr
            obj = {j: POSE_REPOSO[j] + k * (gesto["pose"][j] - POSE_REPOSO[j]) for j in POSE_REPOSO}
            for j, v in postura.items():
                obj[j] += v
            entrada = min(1.0, t_gesto / 0.35)            # quietud breve al cambiar de frase
            anim = gesto["anim"]
            if anim == "asentir" and t_gesto < 1.3:        # asiente con el torso
                s = math.sin(2 * math.pi * 1.5 * t_gesto) * (1 - t_gesto / 1.3)
                obj["waist_pitch"] += 0.07 * s
                obj["pelvis_pitch"] += 0.025 * s
            elif anim == "golpe":                           # se acerca y marca la idea con el cuerpo
                s = max(0.0, math.sin(2 * math.pi * 0.9 * t_gesto)) * k
                obj["agachar"] += 0.08 * s
                obj["waist_pitch"] += 0.04 * s
                obj["left_elbow"] += 0.30 * s
                obj["right_elbow"] += 0.30 * s
            elif anim == "bajar":                           # las manos bajan despacio, como una exhalación
                s = 0.5 - 0.5 * math.cos(2 * math.pi * 0.30 * t_gesto)
                obj["left_elbow"] += 0.45 * s
                obj["right_elbow"] += 0.45 * s
                obj["left_shoulder_pitch"] += 0.22 * s
                obj["right_shoulder_pitch"] += 0.22 * s
                obj["agachar"] += 0.05 * s
            elif anim == "rebote":                          # rebota sobre las rodillas
                s = 0.5 + 0.5 * math.sin(2 * math.pi * 1.9 * t_gesto)
                obj["agachar"] += 0.09 * s * k
                obj["right_shoulder_roll"] -= 0.12 * math.sin(2 * math.pi * 1.9 * t_gesto)
            elif anim == "negar" and t_gesto < 1.8:         # "no" con todo el torso
                obj["waist_yaw"] += 0.11 * math.sin(2 * math.pi * 1.2 * t_gesto) * (1 - t_gesto / 1.8)
            # Acento al ritmo del habla: antebrazo dominante, torso y rodillas
            f = (1.3 + 1.6 * expr) if actitud == "ALEGRE" else (0.7 + 0.9 * expr)
            b = din["beat"] * (0.5 + expr) * entrada
            s1, s2 = math.sin(2 * math.pi * f * t), math.sin(2 * math.pi * f * t + 1.1)
            obj["right_elbow"] += b * 2.4 * s1
            obj["right_shoulder_pitch"] -= b * s1
            obj["left_elbow"] += b * 1.0 * s2
            obj["waist_pitch"] += b * 0.6 * math.sin(2 * math.pi * f * 0.5 * t)
            obj["agachar"] += b * 0.5 * (0.5 + 0.5 * s1)
            obj["cadera"] += b * 0.35 * math.sin(2 * math.pi * f * 0.25 * t)
            vel = din["vel"] * (0.7 + 0.5 * expr)
        elif modo == "escuchando":
            obj = dict(POSE_ESCUCHA)
            fase = (t_modo % 4.2) / 4.2                     # asiente cada tanto mientras escucha
            if fase > 0.80:
                s = math.sin((fase - 0.80) / 0.20 * 2 * math.pi)
                obj["waist_pitch"] += 0.05 * s
                obj["agachar"] += 0.02 * abs(s)
            vel = 3.2
        elif modo == "pensando":
            obj = dict(POSE_PENSANDO)                       # "mira" arriba a su derecha
            obj["waist_yaw"] += 0.03 * math.sin(2 * math.pi * 0.3 * t)
            vel = 3.4
        elif modo == "saludo":
            obj = dict(POSE_SALUDO)
            obj["right_shoulder_roll"] += 0.28 * math.sin(2 * math.pi * 1.5 * t)
            vel = 7.0
        else:
            obj = dict(POSE_REPOSO)
            for j, v in postura.items():
                obj[j] += 0.5 * v
            vel = 2.6

        # Vida: respiración y cambio de peso (el robot nunca queda "congelado")
        resp = math.sin(2 * math.pi * 0.23 * t)
        obj["waist_pitch"] += 0.012 * resp
        obj["left_shoulder_pitch"] += 0.02 * resp
        obj["right_shoulder_pitch"] += 0.02 * resp
        peso = math.sin(2 * math.pi * 0.06 * t)
        obj["cadera"] += 0.022 * peso
        obj["pelvis_roll"] += 0.012 * peso
        obj["waist_roll"] -= 0.012 * peso
        obj["waist_yaw"] += 0.02 * math.sin(2 * math.pi * 0.05 * t + 1.0)
        return obj, vel

    # ------------------------------------------------------------------ pose -> modelo
    def _aplicar(self, pose: dict[str, float], fijar_referencia: bool = False) -> None:
        """Traduce la pose (articulaciones + canales de cuerpo entero) al modelo."""
        mj = self._mj
        q = self.data.qpos
        q[:] = self.model.qpos0
        pp, pr = pose["pelvis_pitch"], pose["pelvis_roll"]
        cad, ag = pose["cadera"], max(0.0, pose["agachar"])
        for j, x in pose.items():
            adr = self.jmap.get(j)
            if adr is not None:
                q[adr] = x
        for lado in ("left", "right"):
            # Las piernas compensan la inclinación de la pelvis: los pies quedan planos y quietos
            q[self.jmap[f"{lado}_hip_pitch"]] = _PIERNA_BASE["hip_pitch"] - ag / 2 - pp
            q[self.jmap[f"{lado}_knee"]] = _PIERNA_BASE["knee"] + ag
            q[self.jmap[f"{lado}_ankle_pitch"]] = _PIERNA_BASE["ankle_pitch"] - ag / 2
            q[self.jmap[f"{lado}_hip_roll"]] = cad - pr
            q[self.jmap[f"{lado}_ankle_roll"]] = -cad
        c1, s1, c2, s2 = math.cos(pp / 2), math.sin(pp / 2), math.cos(pr / 2), math.sin(pr / 2)
        q[3:7] = [c1 * c2, c1 * s2, s1 * c2, -s1 * s2]
        mj.mj_forward(self.model, self.data)
        pies = np.mean([self.data.xpos[b] for b in self._tobillos], axis=0)
        if fijar_referencia:
            self._ref_pies = pies.copy()
            return
        q[0:3] += self._ref_pies - pies       # los pies no se despegan ni se deslizan
        mj.mj_forward(self.model, self.data)

    def avanzar(self, t: float | None = None) -> None:
        """Avanza el filtro hacia la pose objetivo y actualiza el modelo."""
        t = time.time() if t is None else t
        dt = float(np.clip(t - self._t_prev, 1e-3, 0.1))
        self._t_prev = t
        obj, w = self._objetivo(t)
        for j, x_obj in obj.items():
            x, v = self._pos.get(j, x_obj), self._vel.get(j, 0.0)
            a = w * w * (x_obj - x) - 2.0 * w * v        # resorte críticamente amortiguado
            v += a * dt
            x += v * dt
            self._pos[j], self._vel[j] = x, v
        self._aplicar(self._pos)

    # ------------------------------------------------------------------ render
    def _crear_render(self):
        mj = self._mj
        self._renderer = mj.Renderer(self.model, self.alto, self.ancho)
        cam = mj.MjvCamera()
        cam.type = mj.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.0, 0.0, 0.86]
        cam.distance = 2.25
        cam.azimuth = 168
        cam.elevation = -9
        self._cam = cam

    def renderizar(self) -> np.ndarray:
        if not hasattr(self, "_renderer"):
            self._crear_render()
        self._renderer.update_scene(self.data, self._cam)
        return self._renderer.render()

    def _bucle(self) -> None:
        import cv2

        try:
            self._crear_render()
            periodo = 1.0 / self.fps
            while self._activo:
                t0 = time.time()
                self.avanzar(t0)
                rgb = self.renderizar()
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    self._jpeg = buf.tobytes()
                time.sleep(max(0.0, periodo - (time.time() - t0)))
        except Exception as e:  # el dashboard sigue funcionando sin 3D
            self.error = f"{type(e).__name__}: {e}"
            self._activo = False

    def iniciar(self) -> None:
        if self._hilo and self._hilo.is_alive():
            return
        self._activo = True
        self._hilo = threading.Thread(target=self._bucle, name="cuerpo-3d", daemon=True)
        self._hilo.start()

    def detener(self) -> None:
        self._activo = False

    def jpeg(self) -> bytes:
        return self._jpeg
