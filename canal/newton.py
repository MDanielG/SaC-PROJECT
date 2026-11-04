"""Newton-Raphson para F(v) = 0.

En cada paso k:  J(v_k) h_k = -F(v_k),  v_(k+1) = v_k + h_k,  hasta ||F||inf < tol.

La corrección h_k se puede obtener con:
  "inversa"  h = -J^-1 F, con la inversa explícita (Avance 2)
  "directo"  factorización LU dispersa (spsolve), solo como referencia
  cualquier método iterativo de lineales.METODOS (Avance 3)
"""
import time
from dataclasses import dataclass, field
import numpy as np
from scipy.sparse.linalg import spsolve
from .ecuaciones import residuo
from .jacobiano import jacobiano
from . import lineales

METODOS_DIRECTOS = {"inversa": "Inversa del Jacobiano", "directo": "Directo (LU disperso)"}


def nombre_metodo(metodo):
    if metodo in METODOS_DIRECTOS:
        return METODOS_DIRECTOS[metodo]
    return lineales.METODOS[metodo][1]


@dataclass
class PasoNewton:
    k: int
    v: np.ndarray
    norma_inf: float                 # ||F(v_k)||inf
    norma_2: float                   # ||F(v_k)||2
    iter_lineales: int = 0           # iteraciones del método lineal en este paso
    historial_lineal: list = field(default_factory=list)
    norma_correccion: float = None   # ||h_k||2


@dataclass
class ResultadoNewton:
    v: np.ndarray
    pasos: list
    metodo: str
    tiempo: float
    convergio: bool
    motivo: str = ""

    @property
    def iteraciones(self):
        return len(self.pasos) - 1

    @property
    def iter_lineales_total(self):
        return sum(p.iter_lineales for p in self.pasos)


def correccion(J, F, metodo, tol_lineal, max_lineal):
    """Resuelve J h = -F. Devuelve (h, iteraciones, historial, convergió)."""
    if metodo == "inversa":
        return -np.linalg.inv(J.toarray()) @ F, 0, [], True
    if metodo == "directo":
        return spsolve(J.tocsc(), -F), 0, [], True
    r = lineales.resolver(J, -F, metodo, tol=tol_lineal, maxit=max_lineal)
    return r.h, r.iteraciones, r.historial, r.convergio


def newton(canal, v0, metodo="jacobi", tol=1e-10, max_iter=30,
           tol_lineal=1e-12, max_lineal=200_000, verbose=False):
    v = v0.copy(); pasos = []; t0 = time.perf_counter()
    for k in range(max_iter + 1):
        F = residuo(canal, v)
        paso = PasoNewton(k, v.copy(), float(np.max(np.abs(F))), float(np.linalg.norm(F)))
        pasos.append(paso)
        if verbose:
            print(f"  k={k:2d}  ||F||inf = {paso.norma_inf:.3e}", end="")
        if paso.norma_inf < tol:
            if verbose: print()
            return ResultadoNewton(v, pasos, metodo, time.perf_counter() - t0, True, "tolerancia")
        if k == max_iter:
            break
        h, it, hist, ok = correccion(jacobiano(canal, v), F, metodo, tol_lineal, max_lineal)
        paso.iter_lineales, paso.historial_lineal = it, hist
        if verbose:
            print(f"   iteraciones lineales = {it}" + ("" if ok else "  (el método lineal NO converge)"))
        if not ok:
            return ResultadoNewton(v, pasos, metodo, time.perf_counter() - t0, False,
                                   "el método lineal no converge")
        paso.norma_correccion = float(np.linalg.norm(h))
        v = v + h
    return ResultadoNewton(v, pasos, metodo, time.perf_counter() - t0, False, "máximo de iteraciones")
