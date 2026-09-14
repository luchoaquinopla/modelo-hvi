"""Adaptador EchoNext -> formato único (señal + máscara)."""

from pathlib import Path

import numpy as np

from modelo_hvi.formato_unico import (
    DERIVACIONES,
    MODO_COMPLETO,
    MODO_PDF,
    MUESTRAS,
    detectar_tramos_con_senal,
    filtrar_pasabajos,
    mascara_esquema_pdf,
)


class AdaptadorEchoNext:
    """Convierte ondas crudas de EchoNext (n, 1, 2500, 12) al formato único."""

    def convertir_lote(self, ondas: np.ndarray, modo: str) -> tuple[np.ndarray, np.ndarray]:
        if modo not in (MODO_COMPLETO, MODO_PDF):
            raise ValueError(f"modo inválido: {modo!r}. Debe ser {MODO_COMPLETO!r} o {MODO_PDF!r}")

        crudo = np.transpose(ondas[:, 0, :, :], (0, 2, 1)).astype(np.float32)  # (n, 12, 2500)
        tramos_con_senal = detectar_tramos_con_senal(crudo)  # sobre crudo, antes de filtrar
        filtrada = filtrar_pasabajos(crudo)

        if modo == MODO_PDF:
            mascara = tramos_con_senal & mascara_esquema_pdf()[None, :, :]
        else:
            mascara = tramos_con_senal

        senal = np.where(mascara, filtrada, 0).astype(np.float16)
        return senal, mascara

    def convertir_archivo(
        self,
        ruta_ondas: str,
        carpeta_salida: str,
        modo: str,
        tam_lote: int = 2048,
    ) -> dict:
        carpeta_salida = Path(carpeta_salida)
        carpeta_salida.mkdir(parents=True, exist_ok=True)

        ondas = np.load(ruta_ondas, mmap_mode="r")
        n = ondas.shape[0]
        forma = (n, len(DERIVACIONES), MUESTRAS)
        senal_salida = np.lib.format.open_memmap(
            carpeta_salida / "senal.npy", mode="w+", dtype=np.float16, shape=forma
        )
        mascara_salida = np.lib.format.open_memmap(
            carpeta_salida / "mascara.npy", mode="w+", dtype=bool, shape=forma
        )

        ecg_con_tramos_vacios = 0
        total_enmascarado = 0
        for ini in range(0, n, tam_lote):
            fin = min(ini + tam_lote, n)
            lote = np.asarray(ondas[ini:fin])
            senal_lote, mascara_lote = self.convertir_lote(lote, modo)
            senal_salida[ini:fin] = senal_lote
            mascara_salida[ini:fin] = mascara_lote
            ecg_con_tramos_vacios += int(np.any(~mascara_lote, axis=(1, 2)).sum())
            total_enmascarado += int(mascara_lote.sum())
        senal_salida.flush()
        mascara_salida.flush()

        total_posible = n * len(DERIVACIONES) * MUESTRAS
        return {
            "n": n,
            "ecg_con_tramos_vacios": ecg_con_tramos_vacios,
            "fraccion_enmascarada": float(total_enmascarado / total_posible) if total_posible else 0.0,
        }
