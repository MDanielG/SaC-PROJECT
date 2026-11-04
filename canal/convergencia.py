"""Análisis de convergencia ANTES de iterar (Avance 3).

Para h_(m+1) = h_m + Q^-1 (b - J h_m), el error se multiplica por G = I - Q^-1 J.
  - Dominancia diagonal: condición suficiente para Jacobi y Gauss-Seidel.
  - ||G|| < 1: condición suficiente.
  - Radio espectral de G < 1: condición necesaria y suficiente.
"""
import numpy as np
from .jacobiano import jacobiano


def dominancia_diagonal(J):
    """Número de filas donde |diagonal| < suma de |fuera de la diagonal|."""
    diag = np.abs(J.diagonal())
    fuera = np.asarray(abs(J).sum(axis=1)).ravel() - diag
    return int(np.sum(diag < fuera))


def matrices_Q(Jd):
    n = Jd.shape[0]
    return {"Richardson": np.eye(n),
            "Jacobi": np.diag(Jd.diagonal()),
            "Gauss–Seidel": np.tril(Jd)}


def diagnostico(canal, v, densa=True):
    """Resumen de las pruebas de convergencia del Jacobiano evaluado en v.

    densa=True calcula normas y autovalores con matrices densas (solo mallas pequeñas).
    """
    J = jacobiano(canal, v)
    res = {"incognitas": J.shape[0],
           "filas_no_dominantes": dominancia_diagonal(J),
           "simetrica": bool(abs(J - J.T).max() < 1e-14),
           "metodos": {}}
    if densa:
        Jd = J.toarray(); I = np.eye(Jd.shape[0])
        for nombre, Q in matrices_Q(Jd).items():
            G = I - np.linalg.solve(Q, Jd)
            res["metodos"][nombre] = {
                "norma_inf": float(np.max(np.sum(np.abs(G), axis=1))),
                "radio_espectral": float(np.max(np.abs(np.linalg.eigvals(G))))}
        res["min_autovalor_parte_simetrica"] = float(np.linalg.eigvalsh(0.5 * (Jd + Jd.T)).min())
    return res


def imprimir(diag):
    print(f"  Filas no diagonalmente dominantes: {diag['filas_no_dominantes']} de {diag['incognitas']}")
    print(f"  J simétrica: {'sí' if diag['simetrica'] else 'no'}")
    for nombre, d in diag["metodos"].items():
        estado = "converge" if d["radio_espectral"] < 1 else "diverge"
        print(f"  {nombre:13s} ||I - Q^-1 J||inf = {d['norma_inf']:.3f}   "
              f"radio espectral = {d['radio_espectral']:.4f}  -> {estado}")
    if "min_autovalor_parte_simetrica" in diag:
        print(f"  Menor autovalor de (J + J^T)/2: {diag['min_autovalor_parte_simetrica']:.4f}")
