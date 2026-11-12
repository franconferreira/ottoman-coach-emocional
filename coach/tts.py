"""Módulo de voz de Ottoman: síntesis con velocidad, tono y volumen según la actitud.

Usa la voz neuronal es-AR de Edge (requiere internet). Si falla, el dashboard habla con
la voz del navegador usando los mismos parámetros.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DIR_AUDIO = RAIZ / "datos" / "audio_tmp"
VOZ = "es-AR-TomasNeural"
# Voz de caricatura: más aguda y ágil que la voz natural (los efectos de megáfono, saturación
# y eco se aplican en el navegador al reproducirla)
BASE_CARICATURA = {"velocidad": 14, "tono": 20, "volumen": 0}


def _limpiar_viejos(max_seg: float = 900) -> None:
    ahora = time.time()
    for f in DIR_AUDIO.glob("*.mp3"):
        try:
            if ahora - f.stat().st_mtime > max_seg:
                f.unlink()
        except OSError:
            pass


async def sintetizar(texto: str, voz: dict, caricatura: bool = True) -> str | None:
    """Genera el mp3 y devuelve su nombre de archivo, o None si no se pudo."""
    try:
        import edge_tts
        DIR_AUDIO.mkdir(parents=True, exist_ok=True)
        _limpiar_viejos()
        nombre = f"{uuid.uuid4().hex[:12]}.mp3"
        if caricatura:
            voz = {k: voz.get(k, 0) + BASE_CARICATURA[k] for k in BASE_CARICATURA}
        com = edge_tts.Communicate(
            texto, VOZ,
            rate=f"{voz.get('velocidad', 0):+d}%",
            pitch=f"{voz.get('tono', 0):+d}Hz",
            volume=f"{voz.get('volumen', 0):+d}%",
        )
        await asyncio.wait_for(com.save(str(DIR_AUDIO / nombre)), timeout=20)
        if (DIR_AUDIO / nombre).stat().st_size < 500:
            return None
        return nombre
    except Exception:
        return None
