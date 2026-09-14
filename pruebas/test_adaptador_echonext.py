import numpy as np
import pytest

from modelo_hvi.adaptadores.echonext import AdaptadorEchoNext
from modelo_hvi.formato_unico import DERIVACIONES, HZ, MODO_COMPLETO, MODO_PDF, MUESTRAS, mascara_esquema_pdf


def _onda_sintetica(n: int, semilla: int = 0) -> np.ndarray:
    """Ondas sintéticas (n, 1, MUESTRAS, 12), sin PII, solo senoides con ruido."""
    rng = np.random.default_rng(semilla)
    t = np.arange(MUESTRAS) / HZ
    base = np.sin(2 * np.pi * 5 * t)
    ondas = np.zeros((n, 1, MUESTRAS, 12), dtype=np.float32)
    for i in range(n):
        for d in range(12):
            ondas[i, 0, :, d] = base + rng.normal(0, 0.01, MUESTRAS)
    return ondas


def test_convertir_lote_formas_y_dtypes():
    ondas = _onda_sintetica(4)
    senal, mascara = AdaptadorEchoNext().convertir_lote(ondas, MODO_COMPLETO)
    assert senal.shape == (4, 12, MUESTRAS)
    assert mascara.shape == (4, 12, MUESTRAS)
    assert senal.dtype == np.float16
    assert mascara.dtype == bool


def test_convertir_lote_modo_invalido_lanza_value_error():
    ondas = _onda_sintetica(1)
    with pytest.raises(ValueError):
        AdaptadorEchoNext().convertir_lote(ondas, "otro")


def test_convertir_lote_modo_completo_marca_relleno_plano_como_false():
    ondas = _onda_sintetica(1)
    # relleno plano en miembros (derivaciones 0-2) solo entre 5s y 10s, como el 1% de EchoNext
    indice_flat = round(5.0 * HZ)
    ondas[0, 0, indice_flat:, 0] = ondas[0, 0, indice_flat, 0]
    senal, mascara = AdaptadorEchoNext().convertir_lote(ondas, MODO_COMPLETO)
    assert not mascara[0, 0, indice_flat + 200 :].any()
    assert mascara[0, 0, : indice_flat - 200].all()


def test_convertir_lote_modo_pdf_aplica_esquema_y_tramos_vacios():
    ondas = _onda_sintetica(1)
    senal, mascara = AdaptadorEchoNext().convertir_lote(ondas, MODO_PDF)
    esquema = mascara_esquema_pdf()
    # la máscara pdf nunca puede exceder al esquema del pdf
    assert (mascara[0] <= esquema).all()


def test_convertir_lote_ceros_fuera_de_mascara():
    ondas = _onda_sintetica(2)
    senal, mascara = AdaptadorEchoNext().convertir_lote(ondas, MODO_PDF)
    assert np.all(senal[~mascara] == 0)


def test_convertir_archivo_por_lotes_igual_a_convertir_lote(tmp_path):
    n = 5
    ondas = _onda_sintetica(n, semilla=42)
    ruta_ondas = tmp_path / "ondas.npy"
    np.save(ruta_ondas, ondas)
    carpeta_salida = tmp_path / "salida"

    adaptador = AdaptadorEchoNext()
    resultado = adaptador.convertir_archivo(str(ruta_ondas), str(carpeta_salida), MODO_PDF, tam_lote=3)

    senal_esperada, mascara_esperada = adaptador.convertir_lote(ondas, MODO_PDF)
    senal_obtenida = np.load(carpeta_salida / "senal.npy")
    mascara_obtenida = np.load(carpeta_salida / "mascara.npy")

    np.testing.assert_array_equal(senal_obtenida, senal_esperada)
    np.testing.assert_array_equal(mascara_obtenida, mascara_esperada)
    assert resultado["n"] == n
    assert 0.0 <= resultado["fraccion_enmascarada"] <= 1.0
    assert isinstance(resultado["ecg_con_tramos_vacios"], int)
