"""Verifica que la normalización publicada de EchoNext permite volver a milivoltios (experimento E-004).

Parámetros: IntroECG/7-EchoNext Minimodel/models/echonext_multilabel_minimodel/waveform_normalization_params.json
Solo emite agregados. Uso: uv run python exploracion/verificar_unidades_echonext.py
"""

import json
import os
from pathlib import Path

import numpy as np

DIR_DATOS = Path(os.environ.get("ECHONEXT_DIR", r"D:\datasets\echonext"))
SALIDA = Path(__file__).resolve().parents[1] / "resultados" / "unidades_echonext.json"
DERIVACIONES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
MEDIA = [5.438902932045533, 4.386144027595723, -0.8828521117626768, -4.901228423594343, 3.179303315626078,
         1.718962497412901, -4.3853678923766815, -1.4991789389444636, -0.2390101635046568, 3.622168549154881,
         5.396823627457744, 5.660982479475681]
DESVIO = [32.4283237955487, 31.37262367036232, 30.155474930762434, 28.126380180199195, 27.097711990189254,
          26.148743373957526, 37.739210938823206, 52.792504469040814, 56.75138860280555, 49.90223860158139,
          45.001427330308026, 39.06606103398455]
LIMITE_INFERIOR = [-140.0, -206.0, -324.0, -247.0, -153.0, -249.0, -369.0, -522.0, -554.0, -411.0, -290.0, -209.0]
LIMITE_SUPERIOR = [287.0, 297.0, 236.0, 118.5, 270.0, 245.5, 219.0, 297.0, 366.0, 450.0, 451.0, 396.0]
UV_POR_UNIDAD = 4.88  # resolución de amplitud de GE MUSE (hipótesis a verificar con los datos)
TAMANO_MUESTRA = 1000
SEMILLA = 20260912


def main() -> None:
    arreglo = np.load(DIR_DATOS / "EchoNext_val_waveforms.npy", mmap_mode="r")
    indices = np.sort(np.random.default_rng(SEMILLA).choice(arreglo.shape[0], TAMANO_MUESTRA, replace=False))
    crudo = np.asarray(arreglo[indices, 0], dtype=np.float64) * np.array(DESVIO) + np.array(MEDIA)
    pico_a_pico = np.median(crudo.max(axis=1) - crudo.min(axis=1), axis=0)

    diseno = np.column_stack([crudo[..., 0].ravel(), crudo[..., 1].ravel(), np.ones(crudo[..., 0].size)])
    coef, *_ = np.linalg.lstsq(diseno, crudo[..., 2].ravel(), rcond=None)

    resultado = {
        "muestra": {"particion": "val", "ecg": TAMANO_MUESTRA, "semilla": SEMILLA},
        "mediana_pico_a_pico_mV_si_1uV": dict(zip(DERIVACIONES, np.round(pico_a_pico / 1000, 2).tolist())),
        "mediana_pico_a_pico_mV_si_4_88uV": dict(zip(DERIVACIONES, np.round(pico_a_pico * UV_POR_UNIDAD / 1000, 2).tolist())),
        "einthoven_III_sobre_I_II": {"observado": np.round(coef[:2], 3).tolist(), "teorico": [-1.0, 1.0]},
        "recorte_mV": {
            d: [round(li * UV_POR_UNIDAD / 1000, 2), round(ls * UV_POR_UNIDAD / 1000, 2)]
            for d, li, ls in zip(DERIVACIONES, LIMITE_INFERIOR, LIMITE_SUPERIOR)
        },
    }
    SALIDA.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resultado, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
