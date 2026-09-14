"""Formato único de entrada: 12 derivaciones normalizadas + máscara de tramos con dato."""

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

HZ = 250
MUESTRAS = 2500
DERIVACIONES = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")

MODO_COMPLETO = "completo"
MODO_PDF = "pdf"

_INDICE_DERIVACION = {derivacion: i for i, derivacion in enumerate(DERIVACIONES)}


@dataclass(frozen=True)
class EsquemaPdf:
    """Layout 3x4 + tira de ritmo del PDF del instituto (GE 12SL). Valores nominales."""

    inicio_columna_s: tuple[float, float, float, float] = (0.0, 2.5, 5.0, 7.5)
    columnas: tuple[tuple[str, ...], ...] = (
        ("I", "II", "III"),
        ("aVR", "aVL", "aVF"),
        ("V1", "V2", "V3"),
        ("V4", "V5", "V6"),
    )
    muestras_por_tramo: int = 619
    derivacion_ritmo: str = "V1"


def mascara_esquema_pdf(esquema: EsquemaPdf = EsquemaPdf()) -> np.ndarray:
    """Máscara booleana (12, MUESTRAS) de qué tramos existen según el layout del PDF."""
    mascara = np.zeros((len(DERIVACIONES), MUESTRAS), dtype=bool)
    for indice_columna, derivaciones_columna in enumerate(esquema.columnas):
        inicio_muestra = round(esquema.inicio_columna_s[indice_columna] * HZ)
        fin_muestra = inicio_muestra + esquema.muestras_por_tramo
        for derivacion in derivaciones_columna:
            i = _INDICE_DERIVACION[derivacion]
            if derivacion == esquema.derivacion_ritmo:
                mascara[i, :] = True  # tira de ritmo: derivación completa
            else:
                mascara[i, inicio_muestra:fin_muestra] = True
    return mascara


def detectar_tramos_con_senal(
    senal: np.ndarray,
    ventana: int = 125,
    umbral_cambio: float = 1e-6,
    fraccion_minima: float = 0.05,
) -> np.ndarray:
    """Detecta por ventana si hay señal real (vs. relleno plano). Opera sobre el último eje."""
    n_muestras = senal.shape[-1]
    diff = np.diff(senal, axis=-1, prepend=senal[..., :1])
    cambia = np.abs(diff) > umbral_cambio
    n_ventanas = n_muestras // ventana
    mascara = np.zeros_like(senal, dtype=bool)
    for w in range(n_ventanas):
        ini, fin = w * ventana, (w + 1) * ventana
        tiene_senal = cambia[..., ini:fin].mean(axis=-1) >= fraccion_minima
        mascara[..., ini:fin] = tiene_senal[..., None]
    return mascara


def filtrar_pasabajos(senal: np.ndarray, corte_hz: float = 40.0, orden: int = 4) -> np.ndarray:
    """Butterworth pasabajos + filtfilt (sin desfasaje) sobre el eje temporal."""
    sos = signal.butter(orden, corte_hz, btype="low", fs=HZ, output="sos")
    filtrada = signal.sosfiltfilt(sos, senal, axis=-1)
    return filtrada.astype(np.float32)


def armar_entrada(senal: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    """Arma el tensor (24, MUESTRAS) que come la red: 0-11 señal, 12-23 máscara."""
    senal_enmascarada = np.where(mascara, senal, 0).astype(np.float32)
    mascara_num = mascara.astype(np.float32)
    return np.concatenate([senal_enmascarada, mascara_num], axis=0)
