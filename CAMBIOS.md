# Registro de cambios

## Avance 3 · versión 3.0 (noviembre de 2026)

**Modelo**
- **Fronteras corregidas** según la profesora:
  - El borde superior pasó a ser una pared (vx = vy = 0). Antes tenía vx = V0.
  - En la salida, vx = 0. Antes era vx(Nx) = vx(Nx−1), que queda como opción `salida="neumann"`.
  - Las cuatro esquinas son pared.
- **Valor inicial por defecto:** vx = V0·e^(−π·i/Ny), vy = 0,01·sen(πi/Nx)·sen(πj/Ny).

**Métodos**
- Newton ya no usa la inversa: resuelve J·h = −F con métodos iterativos. Están disponibles Richardson, Jacobi, Gauss–Seidel, gradiente descendente, gradiente conjugado y gradiente conjugado sobre las ecuaciones normales.
- La inversa (`--metodo inversa`) y el método directo LU (`--metodo directo`) se conservan para comparar.
- Gauss–Seidel usa una sustitución hacia adelante por bandas (LAPACK `dtbtrs`).
- Se agregaron pruebas de convergencia antes de iterar: dominancia diagonal, ‖I − Q⁻¹J‖∞ y radio espectral.
- El Jacobiano se arma como matriz dispersa. Por eso ya se puede usar la malla oficial de 400 × 40.

**Código**
- El código se reorganizó en el paquete modular `canal/`, con `main.py` como punto de entrada.
- Se agregaron pruebas automáticas en `tests/`.
- Los mapas de calor de vx usan la escala de colores viridis, como en los avances anteriores. Se cambia en `CMAP_VX`, dentro de `canal/graficas.py`.
- Este proyecto reemplaza a los archivos sueltos anteriores:
  - `navier_stokes_avance2.py`
  - `navier_stokes_avance2_valores_constantes.py`
  - `navier_stokes_avance3.py`
  - `Navier_Stockes.py`

**Resultados de referencia en 80 × 8** (verificados en `tests/test_newton.py`):
- 4 pasos de Newton.
- Iteraciones lineales totales: 1646 con Jacobi y 821 con Gauss–Seidel.
- Radio espectral: 0,962 para Jacobi, 0,925 para Gauss–Seidel y 6,85 para Richardson, que diverge.

## Avance 2

- Sistema no lineal F(v) = 0 con 1458 ecuaciones (malla de 80 × 8).
- Jacobiano analítico verificado contra el numérico.
- Newton–Raphson con la inversa explícita del Jacobiano. Convergencia cuadrática en 4 iteraciones.
- Fronteras de entonces (luego corregidas): borde superior con vx = V0 y salida con derivada cero.

**No se trajo a esta versión**, porque eran pruebas puntuales del Avance 2:
- Newton alternado (bloque vx y luego bloque vy).
- El método de punto fijo.
- Los experimentos con Jacobianos mal derivados.

## Avance 1

- Malla de 80 × 8 (proporción 10:1 de la oficial de 400 × 40) con 729 nodos y 1458 incógnitas.
- Discretización con diferencias centradas y tipos de ecuación según la posición del nodo.
