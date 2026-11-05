"""Parámetros de la simulación. Cambiar aquí los valores por defecto."""
from dataclasses import dataclass


@dataclass
class Parametros:
    # Dominio (número de intervalos; los puntos son Nx+1 por Ny+1). h = 1 y nu = 1.
    Nx: int = 80
    Ny: int = 8
    # Velocidad de entrada
    V0: float = 1.0
    # Condición de salida: "cero" (vx = 0) o "neumann" (vx(Nx) = vx(Nx-1))
    salida: str = "cero"
    # Valores iniciales: "exponencial", "difusion" o "constante"
    perfil_inicial: str = "constante"
    vy0: float = 0.01
    # Newton-Raphson
    metodo: str = "jacobi"          # inversa, directo, richardson, jacobi, gauss-seidel, ...
    tol_newton: float = 1e-10       # ||F||inf < tol_newton
    max_newton: int = 30
    # Sistema lineal J h = -F (métodos iterativos)
    tol_lineal: float = 1e-12       # ||b - J h||inf < tol_lineal
    max_lineal: int = 200_000
