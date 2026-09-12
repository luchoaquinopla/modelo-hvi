"""Exploración inicial de EchoNext (experimento E-003).

Solo emite resultados AGREGADOS: nunca imprime ni guarda filas de pacientes (licencia de PhysioNet).
Uso: uv run python exploracion/explorar_echonext.py
"""

import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

DIR_DATOS = Path(os.environ.get("ECHONEXT_DIR", r"D:\datasets\echonext"))
SALIDA = Path(__file__).resolve().parents[1] / "resultados" / "exploracion_echonext.json"
PARTICIONES = ["train", "val", "test"]
ETIQUETA = "lvwt_gte_13_flag"
UMBRAL_CM = 1.3
TAMANO_MUESTRA = 1000
PUNTOS_REGRESION = 200_000
SEMILLA = 20260912
ORDEN_HIPOTESIS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
# Coeficientes teóricos (sobre I y II) de las derivaciones de miembros dependientes (Einthoven/Goldberger)
TEORICOS = {"III": (-1.0, 1.0), "aVR": (-0.5, -0.5), "aVL": (1.0, -0.5), "aVF": (-0.5, 1.0)}


def conteo_etiqueta(serie: pd.Series) -> dict:
    positivos, negativos = int((serie == 1).sum()), int((serie == 0).sum())
    etiquetados = positivos + negativos
    return {
        "positivos": positivos,
        "negativos": negativos,
        "sin_etiqueta": int(serie.isna().sum()),
        "prevalencia_pct": round(100 * positivos / etiquetados, 2) if etiquetados else None,
    }


def resumen_numerico(serie: pd.Series) -> dict:
    d = serie.describe(percentiles=[0.25, 0.5, 0.75])
    return {k: (round(float(v), 3) if pd.notna(v) else None) for k, v in d.items()}


def explorar_metadata(meta: pd.DataFrame) -> dict:
    res = {"filas": len(meta), "columnas": list(meta.columns)}
    res["filas_por_particion"] = {k: int(v) for k, v in meta["split"].value_counts().items()}

    etiquetada = meta[meta["split"].isin(PARTICIONES)]
    res["etiqueta_por_particion"] = {p: conteo_etiqueta(g[ETIQUETA]) for p, g in etiquetada.groupby("split")}

    res["pacientes"] = {}
    for p, g in etiquetada.groupby("split"):
        por_paciente = g.groupby("patient_key").size()
        res["pacientes"][p] = {
            "pacientes_unicos": int(por_paciente.size),
            "ecg_por_paciente_media": round(float(por_paciente.mean()), 2),
            "ecg_por_paciente_max": int(por_paciente.max()),
        }
    conjuntos = {p: set(g["patient_key"]) for p, g in etiquetada.groupby("split")}
    res["pacientes_compartidos_entre_particiones"] = {
        f"{a}-{b}": len(conjuntos[a] & conjuntos[b]) for a, b in combinations(sorted(conjuntos), 2)
    }

    # Consistencia de la etiqueta con las mediciones del eco
    espesor = etiquetada[["ivs_measurement", "lvpw_measurement"]].max(axis=1, skipna=True)
    con_ambas = etiquetada[["ivs_measurement", "lvpw_measurement"]].notna().all(axis=1)
    con_alguna = etiquetada[["ivs_measurement", "lvpw_measurement"]].notna().any(axis=1)
    calculada = (espesor >= UMBRAL_CM).astype(float).where(con_alguna)
    con_etiqueta = etiquetada[ETIQUETA].notna()
    comparables = con_etiqueta & con_alguna
    res["mediciones_eco"] = {
        "con_septum": int(etiquetada["ivs_measurement"].notna().sum()),
        "con_pared_posterior": int(etiquetada["lvpw_measurement"].notna().sum()),
        "con_ambas": int(con_ambas.sum()),
        "con_etiqueta_sin_mediciones": int((con_etiqueta & ~con_alguna).sum()),
        "etiqueta_coincide_con_max_espesor_pct": round(
            100 * float((calculada[comparables] == etiquetada.loc[comparables, ETIQUETA]).mean()), 2
        ) if comparables.any() else None,
        "septum_cm": resumen_numerico(etiquetada["ivs_measurement"]),
        "pared_posterior_cm": resumen_numerico(etiquetada["lvpw_measurement"]),
    }
    tramos = pd.cut(espesor[con_alguna], [0, 1.0, 1.1, 1.3, 1.5, 10], right=False,
                    labels=["<1.0", "1.0-1.1", "1.1-1.3", "1.3-1.5", ">=1.5"])
    res["mediciones_eco"]["max_espesor_por_tramo_cm"] = {str(k): int(v) for k, v in tramos.value_counts().sort_index().items()}

    solo_etiquetados = etiquetada[con_etiqueta]
    res["por_sexo"] = {
        ("masculino" if s == 1 else "femenino" if s == 0 else str(s)): conteo_etiqueta(g[ETIQUETA])
        for s, g in solo_etiquetados.groupby("sex")
    }
    res["edad_por_etiqueta"] = {
        ("con_hvi" if e == 1 else "sin_hvi"): resumen_numerico(g["age_at_ecg"])
        for e, g in solo_etiquetados.groupby(ETIQUETA)
    }
    grupos_edad = pd.cut(solo_etiquetados["age_at_ecg"], [18, 40, 50, 60, 70, 80, 91], right=False)
    res["prevalencia_por_edad"] = {
        str(k): conteo_etiqueta(g[ETIQUETA]) for k, g in solo_etiquetados.groupby(grupos_edad, observed=True)
    }
    res["anio_adquisicion"] = {
        "min": int(meta["acquisition_year"].min()), "max": int(meta["acquisition_year"].max())
    }
    res["ambito"] = {k: int(v) for k, v in meta["location_setting"].value_counts(dropna=False).items()}
    return res


def r2(y: np.ndarray, x: np.ndarray) -> tuple[float, np.ndarray]:
    diseno = np.column_stack([x, np.ones(len(y))])
    coef, *_ = np.linalg.lstsq(diseno, y, rcond=None)
    residuo = y - diseno @ coef
    return 1 - residuo.var() / y.var(), coef


def explorar_senales(meta: pd.DataFrame) -> dict:
    res = {"forma_por_particion": {}}
    for p in PARTICIONES:
        arreglo = np.load(DIR_DATOS / f"EchoNext_{p}_waveforms.npy", mmap_mode="r")
        res["forma_por_particion"][p] = {
            "forma": list(arreglo.shape),
            "tipo": str(arreglo.dtype),
            "coincide_con_metadata": arreglo.shape[0] == int((meta["split"] == p).sum()),
        }

    arreglo = np.load(DIR_DATOS / "EchoNext_val_waveforms.npy", mmap_mode="r")
    rng = np.random.default_rng(SEMILLA)
    indices = np.sort(rng.choice(arreglo.shape[0], size=min(TAMANO_MUESTRA, arreglo.shape[0]), replace=False))
    muestra = np.asarray(arreglo[indices, 0], dtype=np.float64)  # (n, 2500, 12)
    res["muestra"] = {"particion": "val", "ecg": int(len(indices)), "semilla": SEMILLA}

    por_derivacion = muestra.reshape(-1, 12)
    res["estadisticas_por_columna"] = [
        {
            "columna": c,
            "media": round(float(por_derivacion[:, c].mean()), 4),
            "desvio": round(float(por_derivacion[:, c].std()), 4),
            "p01": round(float(np.percentile(por_derivacion[:, c], 1)), 3),
            "p99": round(float(np.percentile(por_derivacion[:, c], 99)), 3),
            "nan": int(np.isnan(por_derivacion[:, c]).sum()),
        }
        for c in range(12)
    ]
    res["ecg_totalmente_planos"] = int((muestra.std(axis=(1, 2)) == 0).sum())

    # Orden de derivaciones: las 6 de miembros viven en un plano (cualquiera = combinación de otras dos)
    puntos = por_derivacion[rng.choice(len(por_derivacion), size=PUNTOS_REGRESION, replace=False)]
    mejor_par = {}
    for k in range(12):
        candidatos = [(r2(puntos[:, k], puntos[:, [i, j]])[0], (i, j)) for i, j in combinations(range(12), 2) if k not in (i, j)]
        valor, par = max(candidatos)
        mejor_par[k] = {"r2_max_con_dos_columnas": round(float(valor), 4), "par": list(par)}
    res["dependencia_lineal_por_columna"] = mejor_par

    # Prueba de la hipótesis de orden estándar: columnas 2-5 como combinación de columnas 0 y 1
    res["hipotesis_orden_estandar"] = {}
    for c, nombre in enumerate(ORDEN_HIPOTESIS[2:6], start=2):
        valor, coef = r2(puntos[:, c], puntos[:, [0, 1]])
        res["hipotesis_orden_estandar"][nombre] = {
            "columna": c,
            "r2": round(float(valor), 4),
            "coef_col0": round(float(coef[0]), 3),
            "coef_col1": round(float(coef[1]), 3),
            "teorico": list(TEORICOS[nombre]),
        }
    res["correlacion_entre_columnas"] = np.round(np.corrcoef(puntos.T), 2).tolist()
    return res


def main() -> None:
    meta = pd.read_csv(DIR_DATOS / "echonext_metadata_100k.csv")
    resultado = {"metadata": explorar_metadata(meta), "senales": explorar_senales(meta)}
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resultado, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
