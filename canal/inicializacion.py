"""Valores iniciales de las matrices vx y vy.

Criterio de la profesora: vx cerca de V0 a la entrada, cerca de 0 a la salida y
decreciendo en el medio; vy pequeña.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from . import fronteras

PERFILES = ("exponencial", "difusion", "constante")


def condicion_inicial(canal, perfil="exponencial", vy0=0.01):
    """Vector v0 con las fronteras ya impuestas."""
    m = canal.malla
    I, Jj = np.meshgrid(np.arange(m.nx), np.arange(m.ny), indexing="ij")
    if perfil == "exponencial":
        # gradiente en x: decaimiento exp(-pi x / H), asignado directamente
        vx = canal.V0 * np.exp(-np.pi * I / m.Ny)
    elif perfil == "difusion":
        vx = solucion_difusion(canal)
    elif perfil == "constante":
        vx = np.full((m.nx, m.ny), 0.5 * canal.V0)
    else:
        raise ValueError(f"perfil desconocido: {perfil}. Opciones: {PERFILES}")
    vy = np.full((m.nx, m.ny), 0.01 * canal.V0)
    vx, vy = fronteras.aplicar(canal, vx, vy)
    return m.unir(vx, vy)


def solucion_difusion(canal):
    """Laplace(vx) = 0 con las mismas fronteras (problema de solo difusión)."""
    m = canal.malla
    A = sp.lil_matrix((m.N, m.N)); b = np.zeros(m.N)
    for i in range(m.nx):
        for j in range(m.ny):
            k = m.kx(i, j); t = m.tipo(i, j)
            A[k, k] = 1.0
            if t == "inlet":
                b[k] = canal.V0
            elif t == "outlet" and canal.salida == "neumann":
                A[k, m.kx(i - 1, j)] = -1.0
            elif t == "interior":
                A[k, k] = 4.0
                for a, c in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                    A[k, m.kx(a, c)] = -1.0
    return spsolve(A.tocsc(), b).reshape(m.nx, m.ny)
