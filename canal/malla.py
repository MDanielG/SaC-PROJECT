"""Malla del canal: numeración de nodos e incógnitas.

Nodos (i, j) con i = 0..Nx (x = i*h) y j = 0..Ny (y = j*h), h = 1.
Vector de incógnitas v = [vx de todos los nodos ; vy de todos los nodos]
con k = i*(Ny+1) + j para vx y N + k para vy.
"""
import numpy as np


class Malla:
    def __init__(self, Nx, Ny):
        self.Nx, self.Ny = Nx, Ny
        self.nx, self.ny = Nx + 1, Ny + 1      # puntos por dirección
        self.N = self.nx * self.ny             # número de nodos
        self.n = 2 * self.N                    # número de incógnitas

    # ---- numeración ----
    def kx(self, i, j):
        return i * self.ny + j

    def ky(self, i, j):
        return self.N + i * self.ny + j

    # ---- vector <-> matrices ----
    def separar(self, v):
        """Vector v -> matrices vx, vy de tamaño (Nx+1) x (Ny+1)."""
        return v[:self.N].reshape(self.nx, self.ny), v[self.N:].reshape(self.nx, self.ny)

    def unir(self, vx, vy):
        return np.concatenate([vx.ravel(), vy.ravel()])

    # ---- tipos de nodo ----
    def tipo(self, i, j):
        if j == 0 or j == self.Ny:
            return "pared"          # incluye las cuatro esquinas
        if i == 0:
            return "inlet"
        if i == self.Nx:
            return "outlet"
        return "interior"

    def mascara_interior(self):
        m = np.zeros((self.nx, self.ny), bool)
        m[1:-1, 1:-1] = True
        return m

    def indices_interiores(self):
        """Arreglos (I, J) con todos los nodos interiores."""
        I, J = np.meshgrid(np.arange(1, self.Nx), np.arange(1, self.Ny), indexing="ij")
        return I.ravel(), J.ravel()

    def __repr__(self):
        return f"Malla({self.Nx}x{self.Ny}: {self.N} nodos, {self.n} incógnitas)"
