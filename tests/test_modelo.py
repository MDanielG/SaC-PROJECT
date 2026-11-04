"""Pruebas del modelo: fronteras, residuo y Jacobiano.  Correr con:  python -m pytest"""
import numpy as np
import pytest
import canal as cn


@pytest.mark.parametrize("salida", ["cero", "neumann"])
def test_jacobiano_analitico_igual_al_numerico(salida):
    c = cn.Canal(12, 4, salida=salida)
    v = cn.condicion_inicial(c, "exponencial")
    v = v + 0.01 * np.random.default_rng(0).standard_normal(v.size)
    dif = np.max(np.abs(cn.jacobiano(c, v).toarray() - cn.jacobiano_numerico(c, v)))
    assert dif < 1e-8


def test_conteo_de_nodos_80x8():
    c = cn.Canal(80, 8)
    assert c.N == 729 and c.n == 1458
    assert c.malla.mascara_interior().sum() == 553


def test_elementos_no_cero_jacobiano():
    c = cn.Canal(80, 8)
    assert cn.jacobiano(c, cn.condicion_inicial(c)).nnz == 553 * 12 + 352


def test_condicion_inicial_cumple_fronteras():
    c = cn.Canal(20, 6)
    v0 = cn.condicion_inicial(c)
    vx, vy = c.malla.separar(v0)
    assert np.all(vx[0, 1:-1] == c.V0)
    assert np.all(vx[:, 0] == 0) and np.all(vx[:, -1] == 0) and np.all(vx[-1, :] == 0)
    assert np.all(vy[0, :] == 0) and np.all(vy[:, 0] == 0)
    F = cn.residuo(c, v0)
    Fx, Fy = c.malla.separar(F)
    assert np.all(Fx[0, :] == 0) and np.all(Fx[:, 0] == 0) and np.all(Fy[:, -1] == 0)
