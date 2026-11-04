"""Pruebas de Newton y de los métodos lineales (resultados de referencia del Avance 3)."""
import numpy as np
import pytest
import canal as cn


@pytest.fixture(scope="module")
def caso():
    c = cn.Canal(80, 8)
    v0 = cn.condicion_inicial(c, "exponencial")
    ref = cn.newton(c, v0, "directo")
    return c, v0, ref


def test_newton_directo_converge_en_4_pasos(caso):
    c, v0, ref = caso
    assert ref.convergio and ref.iteraciones == 4
    vx, _ = c.malla.separar(ref.v)
    assert vx[5, 4] == pytest.approx(0.43499, abs=1e-4)          # centro del canal
    assert np.max(np.abs(vx - vx[:, ::-1])) < 1e-12               # simetría arriba/abajo


@pytest.mark.parametrize("metodo,iter_total", [("jacobi", 1646), ("gauss-seidel", 821)])
def test_metodos_de_clase(caso, metodo, iter_total):
    c, v0, ref = caso
    r = cn.newton(c, v0, metodo)
    assert r.convergio and r.iteraciones == 4
    assert r.iter_lineales_total == iter_total
    assert np.max(np.abs(r.v - ref.v)) < 1e-10


@pytest.mark.parametrize("metodo", ["inversa", "gradiente", "gradiente-conjugado-normal"])
def test_otros_metodos_misma_solucion(caso, metodo):
    c, v0, ref = caso
    r = cn.newton(c, v0, metodo)
    assert r.convergio and np.max(np.abs(r.v - ref.v)) < 1e-10


def test_richardson_diverge(caso):
    c, v0, _ = caso
    r = cn.newton(c, v0, "richardson")
    assert not r.convergio and r.pasos[0].iter_lineales == 18


def test_radios_espectrales(caso):
    c, v0, _ = caso
    d = cn.convergencia.diagnostico(c, v0)
    assert d["filas_no_dominantes"] == 886 and not d["simetrica"]
    assert d["metodos"]["Jacobi"]["radio_espectral"] == pytest.approx(0.9616, abs=1e-3)
    assert d["metodos"]["Gauss–Seidel"]["radio_espectral"] == pytest.approx(0.9248, abs=1e-3)
    assert d["metodos"]["Richardson"]["radio_espectral"] > 1
