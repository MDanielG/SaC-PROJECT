"""Jacobiano J[m, n] = dF_m / dv_n.

Derivadas de F^x en un nodo interior:
  vx propio: 4 + 1/2 (vxE - vxW)     vxE: -1 + 1/2 vx     vxW: -1 - 1/2 vx
  vxN: -1 + 1/2 vy                   vxS: -1 - 1/2 vy     vy propio: 1/2 (vxN - vxS)
Derivadas de F^y en un nodo interior:
  vy propio: 4 + 1/2 (vyN - vyS)     vecinos: igual que F^x   vx propio: 1/2 (vyE - vyW)
Filas de frontera: 1 en la diagonal (ver fronteras.py).
"""
import numpy as np
import scipy.sparse as sp
from . import fronteras
from .ecuaciones import residuo


def jacobiano(canal, v):
    """Jacobiano analítico como matriz dispersa CSR."""
    m = canal.malla
    vx, vy = m.separar(v)
    I, Jj = m.indices_interiores()
    kx = I * m.ny + Jj
    ky = m.N + kx
    u, w = vx[I, Jj], vy[I, Jj]
    filas, cols, vals = [], [], []

    def poner(f, c, val):
        filas.append(f); cols.append(c); vals.append(val)

    cE, cW, cN, cS = -1 + 0.5 * u, -1 - 0.5 * u, -1 + 0.5 * w, -1 - 0.5 * w
    for k in (kx, ky):                               # vecinos (iguales en F^x y F^y)
        poner(k, k + m.ny, cE); poner(k, k - m.ny, cW)
        poner(k, k + 1, cN);    poner(k, k - 1, cS)
    poner(kx, kx, 4 + 0.5 * (vx[I + 1, Jj] - vx[I - 1, Jj]))   # diagonal F^x
    poner(kx, ky, 0.5 * (vx[I, Jj + 1] - vx[I, Jj - 1]))       # cruzado dF^x/dvy
    poner(ky, ky, 4 + 0.5 * (vy[I, Jj + 1] - vy[I, Jj - 1]))   # diagonal F^y
    poner(ky, kx, 0.5 * (vy[I + 1, Jj] - vy[I - 1, Jj]))       # cruzado dF^y/dvx

    fb, cb, vb = fronteras.entradas_jacobiano(canal)
    filas.append(fb); cols.append(cb); vals.append(vb)
    J = sp.coo_matrix((np.concatenate(vals), (np.concatenate(filas), np.concatenate(cols))),
                      shape=(m.n, m.n))
    return J.tocsr()


def jacobiano_denso(canal, v):
    return jacobiano(canal, v).toarray()


def jacobiano_numerico(canal, v, eps=1e-6):
    """Jacobiano por diferencias centrales (solo para verificar; es lento)."""
    n = canal.malla.n
    J = np.zeros((n, n))
    for k in range(n):
        d = np.zeros(n); d[k] = eps
        J[:, k] = (residuo(canal, v + d) - residuo(canal, v - d)) / (2 * eps)
    return J
