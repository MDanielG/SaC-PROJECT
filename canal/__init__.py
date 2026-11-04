"""Simulación de flujo incompresible estacionario en un canal 2D (Navier-Stokes, diferencias finitas).

Grupo 12 · Simulación. Versión del Avance 3.

Módulos:
  config          parámetros por defecto
  malla           numeración de nodos e incógnitas
  modelo          el canal (malla + V0 + condición de salida)
  fronteras       condiciones de frontera
  ecuaciones      sistema no lineal F(v) = 0
  jacobiano       Jacobiano analítico (disperso) y numérico
  inicializacion  valores iniciales de vx y vy
  lineales        métodos para J h = -F (Richardson, Jacobi, Gauss-Seidel, gradiente...)
  newton          Newton-Raphson
  convergencia    pruebas de convergencia antes de iterar
  graficas        mapas de calor y gráficas
"""
__version__ = "3.0"

from .config import Parametros
from .malla import Malla
from .modelo import Canal
from .ecuaciones import residuo
from .jacobiano import jacobiano, jacobiano_numerico
from .inicializacion import condicion_inicial
from .newton import newton, nombre_metodo
from . import lineales, convergencia, graficas
