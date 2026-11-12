"""Lectura del sonido de la voz (no de las palabras): tono, volumen, ritmo y pausas.

La voz sola indica bien la ACTIVACIÓN (agitada o apagada) y poco si la emoción es
agradable o no; por eso pesa menos que el texto y la cara en el veredicto. Con muestras
propias del usuario se usa además un modelo personal.
"""
from __future__ import annotations

import io
import math
import wave

import numpy as np

from .emociones import EMOCIONES, normalizar
from .entrenamiento import ModeloPersonal

COLUMNAS = ["tono_hz", "tono_variacion", "tono_rango", "volumen_db", "volumen_variacion",
            "ritmo", "pausas", "brillo_hz", "zcr"]
ETIQUETAS = {
    "tono_hz": ("Tono medio", "Hz"), "tono_variacion": ("Variación del tono", "Hz"),
    "volumen_db": ("Volumen", "dB"), "ritmo": ("Ritmo", "síl/s"), "pausas": ("Pausas", "%"),
}
FS = 16000
# El volumen absoluto depende del micrófono y de su ganancia, no de la emoción: no entra al modelo
N_MEL = 20
COLUMNAS_TIMBRE = ([f"timbre_media_{i}" for i in range(N_MEL)] + [f"timbre_desvio_{i}" for i in range(N_MEL)]
                   + [f"timbre_cambio_{i}" for i in range(N_MEL)])
COLUMNAS_MODELO = [c for c in COLUMNAS if c != "volumen_db"] + COLUMNAS_TIMBRE


def _timbre(x: np.ndarray) -> list[float]:
    """Forma del espectro en 20 bandas mel (media, variación y cambio en el tiempo).

    Describe el "color" de la voz (tensa, apagada, brillante), que separa emociones con
    la misma activación: por ejemplo alegría de enojo. Se le quita el nivel de cada cuadro
    para que no dependa del volumen del micrófono.
    """
    largo, paso = 400, 160
    cuadros = np.stack([x[i:i + largo] for i in range(0, len(x) - largo, paso)]) * np.hanning(largo)
    rms = np.sqrt((cuadros ** 2).mean(axis=1))
    potencia = np.abs(np.fft.rfft(cuadros[rms > 0.25 * np.percentile(rms, 90)], 512, axis=1)) ** 2
    f = np.fft.rfftfreq(512, 1 / FS)
    a_mel = lambda hz: 2595 * np.log10(1 + hz / 700)
    bordes = np.interp(np.linspace(a_mel(60), a_mel(7600), N_MEL + 2), a_mel(f), f)
    banco = np.zeros((N_MEL, len(f)))
    for i in range(N_MEL):
        a, b, c = bordes[i:i + 3]
        banco[i] = np.clip(np.minimum((f - a) / (b - a), (c - f) / (c - b)), 0, None)
    L = np.log(potencia @ banco.T + 1e-8)
    L = L - L.mean(axis=1, keepdims=True)
    cambio = np.abs(np.diff(L, axis=0)).mean(axis=0) if len(L) > 1 else np.zeros(N_MEL)
    return [float(v) for v in np.concatenate([L.mean(axis=0), L.std(axis=0), cambio])]


def leer_wav(datos: bytes) -> np.ndarray:
    """WAV PCM 16 bits -> señal mono float32 a 16 kHz."""
    with wave.open(io.BytesIO(datos), "rb") as w:
        fs, canales, ancho = w.getframerate(), w.getnchannels(), w.getsampwidth()
        crudo = w.readframes(w.getnframes())
    if ancho != 2:
        raise ValueError("Se esperaba audio PCM de 16 bits")
    x = np.frombuffer(crudo, dtype=np.int16).astype(np.float32) / 32768.0
    if canales > 1:
        x = x.reshape(-1, canales).mean(axis=1)
    if fs != FS and len(x) > 1:
        n = int(len(x) * FS / fs)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return x


def _tono(cuadro: np.ndarray) -> float:
    """Frecuencia fundamental por autocorrelación (0 si el cuadro no es sonoro)."""
    c = cuadro - cuadro.mean()
    ac = np.correlate(c, c, mode="full")[len(c) - 1:]
    if ac[0] <= 1e-8:
        return 0.0
    lo, hi = FS // 400, FS // 70
    tramo = ac[lo:hi]
    k = int(np.argmax(tramo))
    return FS / (k + lo) if tramo[k] / ac[0] > 0.35 else 0.0


def extraer(x: np.ndarray) -> dict | None:
    if len(x) < FS * 0.6:
        return None
    # Se iguala el nivel de la grabación para que un micrófono bajo no cambie las medidas
    pico = float(np.percentile(np.abs(x), 99.9))
    if pico < 1e-4:
        return None
    x = np.clip(x / pico * 0.9, -1.0, 1.0)
    largo, paso = int(0.04 * FS), int(0.02 * FS)
    n = 1 + (len(x) - largo) // paso
    cuadros = np.stack([x[i * paso: i * paso + largo] for i in range(n)])
    rms = np.sqrt((cuadros ** 2).mean(axis=1) + 1e-10)
    umbral = max(0.008, 0.25 * float(np.percentile(rms, 90)))
    habla = rms > umbral
    if habla.sum() < 8:
        return None
    f0 = np.array([_tono(c) if h else 0.0 for c, h in zip(cuadros, habla)])
    sonoros = f0[f0 > 0]
    if len(sonoros) < 5:
        sonoros = np.array([0.0])
    # Ritmo: picos de la envolvente de energía (aproxima sílabas por segundo)
    env = np.convolve(rms, np.ones(5) / 5, mode="same")
    picos = (env[1:-1] > env[:-2]) & (env[1:-1] >= env[2:]) & (env[1:-1] > umbral * 1.3)
    seg_habla = habla.sum() * paso / FS
    espectro = np.abs(np.fft.rfft(cuadros[habla] * np.hanning(largo), axis=1))
    frec = np.fft.rfftfreq(largo, 1 / FS)
    brillo = float((espectro * frec).sum() / (espectro.sum() + 1e-9))
    zcr = float((np.abs(np.diff(np.sign(cuadros[habla]), axis=1)) > 0).mean())
    timbre = dict(zip(COLUMNAS_TIMBRE, _timbre(x)))
    return {
        **timbre,
        "tono_hz": float(np.median(sonoros)),
        "tono_variacion": float(np.std(sonoros)),
        "tono_rango": float(np.percentile(sonoros, 90) - np.percentile(sonoros, 10)),
        "volumen_db": float(20 * math.log10(float(rms[habla].mean()) + 1e-9)),
        "volumen_variacion": float(np.std(20 * np.log10(rms[habla] + 1e-9))),
        "ritmo": float(picos.sum() / max(0.3, seg_habla)),
        "pausas": float(1.0 - habla.mean()),
        "brillo_hz": brillo,
        "zcr": zcr,
        "duracion": float(len(x) / FS),
    }


class AnalizadorVoz:
    def __init__(self):
        self.modelo = ModeloPersonal("voz")
        self._historial: list[dict] = []   # turnos previos: línea base de ESTA persona

    def reiniciar(self) -> None:
        self._historial.clear()

    def _base(self) -> dict:
        if len(self._historial) >= 2:
            return {k: float(np.mean([h[k] for h in self._historial])) for k in COLUMNAS}
        # Valores típicos de habla conversacional hasta conocer a la persona
        return {"tono_hz": 150.0, "tono_variacion": 28.0, "tono_rango": 70.0, "volumen_db": -27.0,
                "volumen_variacion": 6.0, "ritmo": 4.2, "pausas": 0.30, "brillo_hz": 1500.0, "zcr": 0.10}

    def analizar(self, wav: bytes) -> dict | None:
        try:
            r = extraer(leer_wav(wav))
        except Exception:
            return None
        if r is None:
            return None
        base = self._base()
        personalizada = len(self._historial) >= 2
        # Activación: más fuerte, más agudo, más rápido y más variado que lo habitual
        d_vol = (r["volumen_db"] - base["volumen_db"]) / 8.0
        d_tono = (r["tono_hz"] - base["tono_hz"]) / max(25.0, 0.2 * base["tono_hz"]) if personalizada else 0.0
        d_var = (r["tono_variacion"] - base["tono_variacion"]) / 25.0
        d_ritmo = (r["ritmo"] - base["ritmo"]) / 1.5
        d_pausa = (base["pausas"] - r["pausas"]) / 0.25
        activacion = float(np.tanh(0.35 * d_vol + 0.25 * d_tono + 0.2 * d_var + 0.3 * d_ritmo + 0.2 * d_pausa))
        arriba, abajo = max(0.0, activacion), max(0.0, -activacion)
        probs = normalizar({
            "NEUTRO": 1.0 - abs(activacion) + 0.15,
            "TRISTEZA": 1.3 * abajo + 0.05,
            "ENOJO": arriba * (0.45 + 0.25 * max(0.0, np.tanh(d_vol))) + 0.05,
            "ANSIEDAD": arriba * (0.45 + 0.25 * max(0.0, np.tanh(d_ritmo))) + 0.05,
            "ALEGRIA": arriba * (0.35 + 0.25 * max(0.0, np.tanh(d_var))) + 0.05,
        })
        fuente = "reglas base" + (" (respecto de tu voz habitual)" if personalizada else "")
        valores = [r[c] for c in COLUMNAS_MODELO]
        p_modelo = self.modelo.predecir(valores)
        if p_modelo:
            probs = normalizar({e: 0.7 * p_modelo[e] + 0.3 * probs[e] for e in EMOCIONES})
            fuente = "modelo personal + reglas"
        self._historial.append(r)
        self._historial[:] = self._historial[-12:]
        return {
            "probs": {e: round(v, 3) for e, v in probs.items()},
            "activacion": round(activacion, 3),
            "calidad": 1.0 if r["duracion"] >= 1.5 else 0.6,
            "fuente": fuente,
            "rasgos": [
                {"nombre": "Tono medio", "valor": f"{r['tono_hz']:.0f} Hz"},
                {"nombre": "Variación del tono", "valor": f"{r['tono_variacion']:.0f} Hz"},
                {"nombre": "Volumen", "valor": f"{r['volumen_db']:.0f} dB"},
                {"nombre": "Ritmo", "valor": f"{r['ritmo']:.1f} síl/s"},
                {"nombre": "Pausas", "valor": f"{100 * r['pausas']:.0f} %"},
            ],
            "columnas": COLUMNAS_MODELO,
            "valores": valores,
        }
