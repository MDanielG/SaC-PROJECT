"""Condiciones de frontera (corregidas en el Avance 3).

  Inlet      i = 0,  1 <= j <= Ny-1 : vx = V0, vy = 0
  Pared inf. j = 0   (todo i)       : vx = 0,  vy = 0
  Pared sup. j = Ny  (todo i)       : vx = 0,  vy = 0
  Outlet     i = Nx, 1 <= j <= Ny-1 : vx = 0,  vy = 0           (salida = "cero")
                                      vx(Nx,j) = vx(Nx-1,j), vy = 0 (salida = "neumann")
  Esquinas                          : vx = 0,  vy = 0  (son pared)

Cada frontera aporta una ecuación simple F = (valor) - (valor impuesto).
En el Jacobiano esas filas tienen un 1 en la diagonal.
"""
import numpy as np


def aplicar(canal, vx, vy):
    """Devuelve copias de vx, vy con los valores de frontera impuestos."""
    vx, vy = vx.copy(), vy.copy()
    vx[0, :] = canal.V0
    if canal.salida == "cero":
        vx[-1, :] = 0.0
    else:
        vx[-1, 1:-1] = vx[-2, 1:-1]
    vx[:, 0] = 0.0
    vx[:, -1] = 0.0
    vy[0, :] = vy[-1, :] = 0.0
    vy[:, 0] = vy[:, -1] = 0.0
    return vx, vy


def residuo(canal, vx, vy, Fx, Fy):
    """Llena (en su lugar) las filas de frontera de Fx y Fy."""
    Fx[0, 1:-1] = vx[0, 1:-1] - canal.V0
    if canal.salida == "cero":
        Fx[-1, 1:-1] = vx[-1, 1:-1]
    else:
        Fx[-1, 1:-1] = vx[-1, 1:-1] - vx[-2, 1:-1]
    Fx[:, 0] = vx[:, 0]
    Fx[:, -1] = vx[:, -1]
    Fy[0, :] = vy[0, :]
    Fy[-1, :] = vy[-1, :]
    Fy[:, 0] = vy[:, 0]
    Fy[:, -1] = vy[:, -1]


def entradas_jacobiano(canal):
    """Entradas (filas, columnas, valores) de las filas de frontera del Jacobiano."""
    m = canal.malla
    kb = np.flatnonzero(~m.mascara_interior().ravel())       # nodos de frontera
    filas = [kb, m.N + kb]
    cols = [kb, m.N + kb]
    vals = [np.ones(kb.size), np.ones(kb.size)]
    if canal.salida == "neumann":
        jo = np.arange(1, m.Ny)
        ko = m.Nx * m.ny + jo
        filas.append(ko); cols.append(ko - m.ny); vals.append(-np.ones(jo.size))
    return np.concatenate(filas), np.concatenate(cols), np.concatenate(vals)
