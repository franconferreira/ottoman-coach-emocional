"""Importa grabaciones de voz ya etiquetadas (una carpeta por emoción) como muestras de entrenamiento.

Uso:  python -m coach.importar_voz "C:\\ruta\\a\\la\\carpeta"

Cada subcarpeta se toma como etiqueta. Solo se guardan las medidas numéricas de cada
audio, igual que con las frases grabadas desde el dashboard.
"""
from __future__ import annotations

import sys
from pathlib import Path

from . import entrenamiento
from .voz import COLUMNAS_MODELO, extraer, leer_wav

# Etiquetas de grabaciones anteriores -> emociones del proyecto. "motivado" y "cansancio"
# son aproximaciones: comparten la activación (alta / baja) con alegría y tristeza.
MAPA = {
    "alegria": "ALEGRIA", "motivado": "ALEGRIA", "tristeza": "TRISTEZA", "cansancio": "TRISTEZA",
    "enojo": "ENOJO", "ansiedad": "ANSIEDAD", "neutro": "NEUTRO",
}


def importar(carpeta: Path, prefijo: str = "imp") -> dict[str, int]:
    conteo: dict[str, int] = {}
    for sub in sorted(p for p in carpeta.iterdir() if p.is_dir()):
        etiqueta = MAPA.get(sub.name.lower())
        if not etiqueta:
            print(f"  (se saltea la carpeta '{sub.name}': etiqueta desconocida)")
            continue
        for wav in sorted(sub.glob("*.wav")):
            try:
                rasgos = extraer(leer_wav(wav.read_bytes()))
            except Exception:
                rasgos = None
            if not rasgos:
                continue
            entrenamiento.agregar("voz", etiqueta, f"{prefijo}-{wav.stem}", COLUMNAS_MODELO,
                                  [rasgos[c] for c in COLUMNAS_MODELO])
            conteo[etiqueta] = conteo.get(etiqueta, 0) + 1
    return conteo


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    print(importar(Path(sys.argv[1])))
    print(entrenamiento.entrenar("voz"))
