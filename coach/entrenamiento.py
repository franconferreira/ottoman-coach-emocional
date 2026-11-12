"""Datos propios del usuario y modelos personales (cara y voz).

Cada muestra es un vector numérico de rasgos (nunca la foto ni el audio) con su etiqueta
de emoción y el número de ráfaga en que se capturó. La validación se hace por ráfagas
completas: los cuadros de una misma ráfaga son casi iguales entre sí y mezclarlos entre
entrenamiento y prueba inflaría la precisión.
"""
from __future__ import annotations

import csv
import json
import threading
import time
from pathlib import Path

import numpy as np

from .emociones import EMOCIONES

RAIZ = Path(__file__).resolve().parent.parent
DIR_DATOS = RAIZ / "datos"
DIR_MODELOS = RAIZ / "modelos"
_lock = threading.Lock()


# --------------------------------------------------------------------------------------
# Perfiles: cada persona tiene sus propias muestras, sus modelos y la "firma" de su cara
# (proporciones entre puntos fijos del rostro), que sirve para reconocerla por la cámara.
# --------------------------------------------------------------------------------------
ARCHIVO_PERFILES = DIR_DATOS / "perfiles.json"
_perfiles: dict | None = None


def _guardar_perfiles() -> None:
    DIR_DATOS.mkdir(exist_ok=True)
    ARCHIVO_PERFILES.write_text(json.dumps(_perfiles, ensure_ascii=False, indent=1), encoding="utf-8")


def _cargar_perfiles() -> dict:
    global _perfiles
    if _perfiles is None:
        if ARCHIVO_PERFILES.exists():
            _perfiles = json.loads(ARCHIVO_PERFILES.read_text(encoding="utf-8"))
        else:
            # Primera vez: los datos que ya existían pasan a ser el primer perfil
            _perfiles = {"activo": "p1", "perfiles": {"p1": {"nombre": "Franco", "firma": None}}}
            for modalidad in ("cara", "voz"):
                movimientos = [(DIR_DATOS / f"muestras_{modalidad}.csv", DIR_DATOS / "perfiles" / "p1")]
                movimientos += [(DIR_MODELOS / f"personal_{modalidad}{ext}", DIR_MODELOS / "perfiles" / "p1")
                                for ext in (".joblib", ".json")]
                for origen, carpeta in movimientos:
                    if origen.exists():
                        carpeta.mkdir(parents=True, exist_ok=True)
                        origen.replace(carpeta / origen.name)
            _guardar_perfiles()
    return _perfiles


def perfil_activo() -> str:
    return _cargar_perfiles()["activo"]


def nombre_perfil(pid: str | None = None) -> str:
    p = _cargar_perfiles()
    return p["perfiles"][pid or p["activo"]]["nombre"]


def perfiles() -> dict:
    p = _cargar_perfiles()
    return {"activo": p["activo"],
            "lista": [{"id": pid, "nombre": d["nombre"], "tiene_firma": bool(d.get("firma"))}
                      for pid, d in p["perfiles"].items()]}


def crear_perfil(nombre: str) -> str:
    nombre = " ".join(nombre.split())[:40]
    if not nombre:
        raise ValueError("El perfil necesita un nombre.")
    p = _cargar_perfiles()
    with _lock:
        if any(d["nombre"].lower() == nombre.lower() for d in p["perfiles"].values()):
            raise ValueError("Ya hay un perfil con ese nombre.")
        pid = f"p{1 + max(int(k[1:]) for k in p['perfiles'])}"
        p["perfiles"][pid] = {"nombre": nombre, "firma": None}
        p["activo"] = pid
        _guardar_perfiles()
    return pid


def activar_perfil(pid: str) -> None:
    p = _cargar_perfiles()
    if pid not in p["perfiles"]:
        raise ValueError("Ese perfil no existe.")
    with _lock:
        p["activo"] = pid
        _guardar_perfiles()


def firmas() -> dict[str, np.ndarray]:
    """Firma facial media de cada perfil que ya registró su cara."""
    return {pid: np.array(d["firma"]["media"], dtype=np.float32)
            for pid, d in _cargar_perfiles()["perfiles"].items() if d.get("firma") and d["firma"]["n"] >= 8}


def acumular_firma(vectores: list) -> None:
    """Suma mediciones de la cara a la firma del perfil activo (promedio acumulado)."""
    if not vectores:
        return
    p = _cargar_perfiles()
    with _lock:
        d = p["perfiles"][p["activo"]]
        f = d.get("firma") or {"n": 0, "media": [0.0] * len(vectores[0])}
        media, n = np.array(f["media"], dtype=np.float64), int(f["n"])
        for v in vectores:
            if len(v) != len(media):
                continue
            n_eff = min(n, 400)                    # tras 400 mediciones se vuelve una media móvil
            media = (media * n_eff + np.array(v, dtype=np.float64)) / (n_eff + 1)
            n += 1
        d["firma"] = {"n": n, "media": [round(float(x), 5) for x in media]}
        _guardar_perfiles()


def _csv(modalidad: str) -> Path:
    return DIR_DATOS / "perfiles" / perfil_activo() / f"muestras_{modalidad}.csv"


def _modelo(modalidad: str) -> Path:
    return DIR_MODELOS / "perfiles" / perfil_activo() / f"personal_{modalidad}.joblib"


def agregar(modalidad: str, etiqueta: str, rafaga: str, columnas: list[str], valores: list[float]) -> None:
    if etiqueta not in EMOCIONES:
        raise ValueError(f"Etiqueta desconocida: {etiqueta}")
    ruta = _csv(modalidad)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        nuevo = not ruta.exists()
        with ruta.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if nuevo:
                w.writerow(["etiqueta", "rafaga"] + columnas)
            w.writerow([etiqueta, rafaga] + [f"{v:.5f}" for v in valores])


def cargar(modalidad: str):
    ruta = _csv(modalidad)
    if not ruta.exists():
        return [], [], np.zeros((0, 0)), []
    with ruta.open(encoding="utf-8") as f:
        filas = list(csv.reader(f))
    if len(filas) < 2:
        return [], [], np.zeros((0, 0)), filas[0][2:] if filas else []
    columnas = filas[0][2:]
    etiquetas = [r[0] for r in filas[1:]]
    rafagas = [r[1] for r in filas[1:]]
    X = np.array([[float(v) for v in r[2:]] for r in filas[1:]], dtype=np.float32)
    return etiquetas, rafagas, X, columnas


def resumen(modalidad: str) -> dict:
    etiquetas, rafagas, _, _ = cargar(modalidad)
    conteo = {e: 0 for e in EMOCIONES}
    raf = {e: set() for e in EMOCIONES}
    for e, r in zip(etiquetas, rafagas):
        if e in conteo:
            conteo[e] += 1
            raf[e].add(r)
    meta = {}
    ruta_meta = _modelo(modalidad).with_suffix(".json")
    if ruta_meta.exists():
        meta = json.loads(ruta_meta.read_text(encoding="utf-8"))
    return {"muestras": conteo, "rafagas": {e: len(s) for e, s in raf.items()},
            "total": len(etiquetas), "modelo": meta or None}


def borrar(modalidad: str) -> None:
    with _lock:
        for ruta in (_csv(modalidad), _modelo(modalidad), _modelo(modalidad).with_suffix(".json")):
            if ruta.exists():
                ruta.unlink()


def entrenar(modalidad: str) -> dict:
    """Entrena y compara dos modelos; guarda el mejor. Devuelve las métricas."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib

    etiquetas, rafagas, X, columnas = cargar(modalidad)
    clases = sorted(set(etiquetas), key=EMOCIONES.index)
    if len(clases) < 2:
        raise ValueError("Hacen falta muestras de al menos 2 emociones distintas.")
    y = np.array(etiquetas)
    grupos = np.array(rafagas)
    n_grupos = len(set(rafagas))
    min_por_clase = min(len({r for e, r in zip(etiquetas, rafagas) if e == c}) for c in clases)
    if min_por_clase < 2:
        raise ValueError("Hacen falta al menos 2 ráfagas por emoción para poder validar el modelo.")

    candidatos = {
        "Regresión logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
        "Red neuronal (MLP 32-16)": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1500, alpha=1e-2, random_state=0)),
    }
    cv = GroupKFold(n_splits=min(5, n_grupos, max(2, min_por_clase)))
    resultados, mejor, mejor_acc, mejor_pred = {}, None, -1.0, None
    for nombre, modelo in candidatos.items():
        pred = cross_val_predict(modelo, X, y, cv=cv, groups=grupos)
        acc = float((pred == y).mean())
        resultados[nombre] = round(acc, 3)
        if acc > mejor_acc:
            mejor, mejor_acc, mejor_pred = nombre, acc, pred
    final = candidatos[mejor].fit(X, y)
    _modelo(modalidad).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modelo": final, "clases": list(final.classes_), "columnas": columnas}, _modelo(modalidad))
    matriz = confusion_matrix(y, mejor_pred, labels=clases).tolist()
    meta = {
        "modalidad": modalidad, "modelo": mejor, "precision_validacion": round(mejor_acc, 3),
        "comparacion": resultados, "clases": clases, "matriz_confusion": matriz,
        "muestras": int(len(y)), "rafagas": int(n_grupos),
        "validacion": f"validación cruzada por ráfagas ({cv.get_n_splits()} particiones)",
        "fecha": time.strftime("%Y-%m-%d %H:%M"),
    }
    _modelo(modalidad).with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


class ModeloPersonal:
    """Carga perezosa del modelo personal de una modalidad, con recarga tras entrenar."""

    def __init__(self, modalidad: str):
        self.modalidad = modalidad
        self._paquete = None
        self._mtime = None

    def _cargar(self):
        ruta = _modelo(self.modalidad)
        if not ruta.exists():
            self._paquete = None
            return None
        m = (str(ruta), ruta.stat().st_mtime)      # cambia al entrenar o al cambiar de perfil
        if self._paquete is None or m != self._mtime:
            import joblib
            self._paquete = joblib.load(ruta)
            self._mtime = m
        return self._paquete

    def disponible(self) -> bool:
        return self._cargar() is not None

    def predecir(self, valores: list[float]) -> dict[str, float] | None:
        paquete = self._cargar()
        if paquete is None or len(valores) != len(paquete["columnas"]):
            return None
        p = paquete["modelo"].predict_proba(np.array([valores], dtype=np.float32))[0]
        probs = {e: 0.0 for e in EMOCIONES}
        for clase, v in zip(paquete["clases"], p):
            probs[clase] = float(v)
        return probs
