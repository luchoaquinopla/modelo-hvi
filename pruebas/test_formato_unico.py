import numpy as np
import pytest

from modelo_hvi.formato_unico import (
    DERIVACIONES,
    HZ,
    MUESTRAS,
    EsquemaPdf,
    armar_entrada,
    detectar_tramos_con_senal,
    filtrar_pasabajos,
    mascara_esquema_pdf,
)


def test_mascara_esquema_pdf_forma():
    mascara = mascara_esquema_pdf()
    assert mascara.shape == (12, MUESTRAS)
    assert mascara.dtype == bool


def test_mascara_esquema_pdf_v1_es_tira_de_ritmo_completa():
    mascara = mascara_esquema_pdf()
    indice_v1 = DERIVACIONES.index("V1")
    assert mascara[indice_v1].sum() == MUESTRAS
    assert mascara[indice_v1].all()


def test_mascara_esquema_pdf_lead_i_solo_primer_tramo():
    mascara = mascara_esquema_pdf()
    indice_i = DERIVACIONES.index("I")
    esperado = np.zeros(MUESTRAS, dtype=bool)
    esperado[0:619] = True
    np.testing.assert_array_equal(mascara[indice_i], esperado)


def test_mascara_esquema_pdf_lead_v4_solo_ultimo_tramo():
    mascara = mascara_esquema_pdf()
    indice_v4 = DERIVACIONES.index("V4")
    esperado = np.zeros(MUESTRAS, dtype=bool)
    esperado[1875 : 1875 + 619] = True
    np.testing.assert_array_equal(mascara[indice_v4], esperado)


def test_mascara_esquema_pdf_total_true_esperado():
    mascara = mascara_esquema_pdf()
    # 11 derivaciones con un tramo de 619 + V1 completa (2500)
    assert mascara.sum() == 11 * 619 + MUESTRAS


def test_detectar_tramos_con_senal_senoide_todo_true():
    t = np.arange(MUESTRAS) / HZ
    senoide = np.tile(np.sin(2 * np.pi * 5 * t), (12, 1))
    mascara = detectar_tramos_con_senal(senoide)
    assert mascara.all()


def test_detectar_tramos_con_senal_bloque_plano_false():
    t = np.arange(MUESTRAS) / HZ
    senal = np.tile(np.sin(2 * np.pi * 5 * t), (12, 1))
    senal[:, 1250:2500] = senal[:, 1249:1250]  # relleno plano desde la mitad
    mascara = detectar_tramos_con_senal(senal)
    assert not mascara[:, 1250:2500].any()
    assert mascara[:, 0:1250].all()


def test_filtrar_pasabajos_deja_pasar_baja_frecuencia():
    t = np.arange(MUESTRAS) / HZ
    senal = np.tile(np.sin(2 * np.pi * 5 * t), (12, 1)).astype(np.float32)
    filtrada = filtrar_pasabajos(senal)
    assert filtrada.dtype == np.float32
    amplitud_original = senal[:, 500:2000].std()
    amplitud_filtrada = filtrada[:, 500:2000].std()
    assert amplitud_filtrada == pytest.approx(amplitud_original, rel=0.1)


def test_filtrar_pasabajos_atenua_alta_frecuencia():
    t = np.arange(MUESTRAS) / HZ
    senal = np.tile(np.sin(2 * np.pi * 80 * t), (12, 1)).astype(np.float32)
    filtrada = filtrar_pasabajos(senal)
    amplitud_original = senal.std()
    amplitud_filtrada = filtrada.std()
    assert amplitud_filtrada < amplitud_original * 0.2


def test_armar_entrada_forma_y_ceros_fuera_de_mascara():
    senal = np.ones((12, MUESTRAS), dtype=np.float32) * 3.0
    mascara = mascara_esquema_pdf()
    entrada = armar_entrada(senal, mascara)
    assert entrada.shape == (24, MUESTRAS)
    assert entrada.dtype == np.float32
    # canal 0 = I: cero fuera del tramo [0, 619)
    assert entrada[0, 619:].sum() == 0
    assert (entrada[0, 0:619] == 3.0).all()


def test_armar_entrada_canales_de_mascara():
    senal = np.zeros((12, MUESTRAS), dtype=np.float32)
    mascara = mascara_esquema_pdf()
    entrada = armar_entrada(senal, mascara)
    np.testing.assert_array_equal(entrada[12:24], mascara.astype(np.float32))


def test_esquema_pdf_es_parametrizable():
    esquema = EsquemaPdf(muestras_por_tramo=100)
    mascara = mascara_esquema_pdf(esquema)
    indice_i = DERIVACIONES.index("I")
    assert mascara[indice_i].sum() == 100
