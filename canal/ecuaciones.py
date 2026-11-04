"""Sistema no lineal F(v) = 0 (ecuaciones 4.60 y 4.61 discretizadas, h = 1, nu = 1, presión constante).

Nodos interiores:
  F^x = 4 vx - (vxE + vxW + vxN + vxS) + 1/2 vx (vxE - vxW) + 1/2 vy (vxN - vxS)
  F^y = 4 vy - (vyE + vyW + vyN + vyS) + 1/2 vx (vyE - vyW) + 1/2 vy (vyN - vyS)
Nodos de frontera: ver fronteras.py
"""
import numpy as np
from . import fronteras

# vistas del nodo central y sus vecinos sobre los nodos interiores
_C = (slice(1, -1), slice(1, -1))
_E = (slice(2, None), slice(1, -1))
_W = (slice(None, -2), slice(1, -1))
_N = (slice(1, -1), slice(2, None))
_S = (slice(1, -1), slice(None, -2))


def residuo(canal, v):
    """Vector F(v) con las 2N ecuaciones."""
    vx, vy = canal.malla.separar(v)
    Fx = np.zeros_like(vx)
    Fy = np.zeros_like(vy)
    for F, f in ((Fx, vx), (Fy, vy)):
        F[_C] = (4 * f[_C] - (f[_E] + f[_W] + f[_N] + f[_S])
                 + 0.5 * vx[_C] * (f[_E] - f[_W])
                 + 0.5 * vy[_C] * (f[_N] - f[_S]))
    fronteras.residuo(canal, vx, vy, Fx, Fy)
    return canal.malla.unir(Fx, Fy)
