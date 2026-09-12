"""Inspecciona si el trazado de un PDF de ECG es imagen o vectores (experimento E-001)."""

import sys
from collections import Counter

import pymupdf


def inspeccionar(ruta: str) -> None:
    pagina = pymupdf.open(ruta)[0]
    dibujos = pagina.get_drawings()
    negros = [d for d in dibujos if d.get("color") and max(d["color"]) < 0.2]
    print("rotacion de pagina:", pagina.rotation)
    print("imagenes embebidas:", len(pagina.get_images(full=True)))
    print("objetos de dibujo:", len(dibujos))
    print("colores mas comunes:", Counter(tuple(round(c, 2) for c in (d.get("color") or ())) for d in dibujos).most_common(3))
    print("segmentos por trazo negro:", sorted(len(d["items"]) for d in negros))
    print("tipos de segmento:", Counter(item[0] for d in negros for item in d["items"]))
    # Paso entre muestras consecutivas del trazo más largo (eje temporal = y por la rotación de 90°)
    largo = max(negros, key=lambda d: len(d["items"]))["items"]
    paso = abs(largo[1][1].y - largo[0][1].y)
    mm_por_pt = 25.4 / 72
    print(f"paso entre puntos: {paso:.4f} pt -> {25 / (paso * mm_por_pt):.1f} muestras/s a 25 mm/s")


if __name__ == "__main__":
    inspeccionar(sys.argv[1])
