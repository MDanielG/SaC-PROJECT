"""Métodos para el sistema lineal J h = b (b = -F), usados dentro de Newton (Avance 3).

Métodos de la clase: forma general  h_(m+1) = h_m + Q^-1 (b - J h_m)
  Richardson    Q = I
  Jacobi        Q = diagonal de J
  Gauss-Seidel  Q = triangular inferior de J (con la diagonal)
Métodos de gradiente:
  gradiente               descenso de gradiente (supone J simétrica definida positiva)
  gradiente-conjugado     gradiente conjugado clásico (supone J simétrica definida positiva)
  gradiente-conjugado-normal  CG sobre J^T J h = J^T b (sirve aunque J no sea simétrica)

Todos parten de h = 0 y paran cuando ||b - J h||inf < tol.
"""
from dataclasses import dataclass, field
import numpy as np
import scipy.sparse as sp
from scipy.linalg.lapack import dtbtrs

LIMITE_DIVERGENCIA = 1e12


@dataclass
class ResultadoLineal:
    h: np.ndarray
    iteraciones: int
    historial: list = field(default_factory=list)   # ||b - J h||inf en cada iteración
    convergio: bool = True


def _iterar(A, b, paso, tol, maxit):
    """Bucle común: calcula el residuo, revisa la parada y aplica h <- paso(h, r)."""
    h = np.zeros_like(b); hist = []
    for m in range(maxit + 1):
        r = b - A @ h
        hist.append(float(np.max(np.abs(r))))
        if hist[-1] < tol:
            return ResultadoLineal(h, m, hist, True)
        if not np.isfinite(hist[-1]) or hist[-1] > LIMITE_DIVERGENCIA or m == maxit:
            return ResultadoLineal(h, m, hist, False)
        h = paso(h, r)


# ------------------------------------------------------------------
# Métodos de la forma h <- h + Q^-1 r
# ------------------------------------------------------------------
def richardson(A, b, tol=1e-12, maxit=200_000):
    return _iterar(A, b, lambda h, r: h + r, tol, maxit)


def jacobi(A, b, tol=1e-12, maxit=200_000):
    d = A.diagonal()
    return _iterar(A, b, lambda h, r: h + r / d, tol, maxit)


class TriangularInferior:
    """Q = triangular inferior de J y la sustitución hacia adelante Q z = r.

    Con v = [vx ; vy], la parte triangular inferior de J es
        Q = [ Lxx   0  ]    Lxx, Lyy: triangulares inferiores de banda (vecinos W y S)
            [ Cyx  Lyy ]    Cyx: términos cruzados
    y Q z = r se resuelve como  Lxx z1 = r1,  Lyy z2 = r2 - Cyx z1.
    Las bandas se guardan para dtbtrs (LAPACK), que hace la sustitución en C.
    """
    def __init__(self, A):
        A = A.tocsr()
        self.N = N = A.shape[0] // 2
        self.Cyx = A[N:, :N].diagonal()
        self.bandas = []
        for B in (A[:N, :N], A[N:, N:]):
            L = sp.tril(B).tocoo()
            kd = int((L.row - L.col).max())
            ab = np.zeros((kd + 1, N))
            ab[L.row - L.col, L.col] = L.data           # ab[i - j, j] = L[i, j]
            self.bandas.append(ab)

    def resolver(self, r):
        N = self.N
        z1, _ = dtbtrs(self.bandas[0], r[:N, None], uplo="L")
        z2, _ = dtbtrs(self.bandas[1],(r[N:] - self.Cyx * z1[:, 0])[:, None],uplo="L")
        return np.concatenate([z1[:, 0], z2[:, 0]])


def gauss_seidel(A, b, tol=1e-12, maxit=200_000):
    Q = TriangularInferior(A)
    return _iterar(A, b, lambda h, r: h + Q.resolver(r), tol, maxit)


def gradiente(A, b, tol=1e-12, maxit=200_000):
    def paso(h, r):
        return h + (r @ r) / (r @ (A @ r)) * r          # paso óptimo en la dirección r
    return _iterar(A, b, paso, tol, maxit)


# ------------------------------------------------------------------
# Gradiente conjugado
# ------------------------------------------------------------------
def gradiente_conjugado(A, b, tol=1e-12, maxit=200_000):
    h = np.zeros_like(b); r = b.copy(); p = r.copy(); rr = r @ r; hist = []
    for m in range(maxit + 1):
        hist.append(float(np.max(np.abs(b - A @ h))))
        if hist[-1] < tol:
            return ResultadoLineal(h, m, hist, True)
        if not np.isfinite(hist[-1]) or hist[-1] > LIMITE_DIVERGENCIA or m == maxit:
            return ResultadoLineal(h, m, hist, False)
        Ap = A @ p
        alfa = rr / (p @ Ap)
        h = h + alfa * p; r = r - alfa * Ap
        rr_nuevo = r @ r
        p = r + (rr_nuevo / rr) * p; rr = rr_nuevo


def gradiente_conjugado_normal(A, b, tol=1e-12, maxit=200_000):
    """CGNR: gradiente conjugado sobre J^T J h = J^T b."""
    At = A.T.tocsr()
    h = np.zeros_like(b); r = b.copy(); z = At @ r; p = z.copy(); zz = z @ z; hist = []
    for m in range(maxit + 1):
        hist.append(float(np.max(np.abs(r))))
        if hist[-1] < tol:
            return ResultadoLineal(h, m, hist, True)
        if not np.isfinite(hist[-1]) or m == maxit:
            return ResultadoLineal(h, m, hist, False)
        Ap = A @ p
        alfa = zz / (Ap @ Ap)
        h = h + alfa * p; r = r - alfa * Ap
        z = At @ r; zz_nuevo = z @ z
        p = z + (zz_nuevo / zz) * p; zz = zz_nuevo


# nombre corto -> (función, nombre para mostrar)
METODOS = {
    "richardson": (richardson, "Richardson"),
    "jacobi": (jacobi, "Jacobi"),
    "gauss-seidel": (gauss_seidel, "Gauss–Seidel"),
    "gradiente": (gradiente, "Gradiente descendente"),
    "gradiente-conjugado": (gradiente_conjugado, "Gradiente conjugado"),
    "gradiente-conjugado-normal": (gradiente_conjugado_normal, "Gradiente conjugado (ec. normales)"),
}


def resolver(A, b, metodo="jacobi", tol=1e-12, maxit=200_000):
    if metodo not in METODOS:
        raise ValueError(f"método desconocido: {metodo}. Opciones: {list(METODOS)}")
    return METODOS[metodo][0](A, b, tol=tol, maxit=maxit)
