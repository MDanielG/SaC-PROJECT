# Simulación de flujo en un canal 2D (Navier–Stokes)

**Grupo 12 · Simulación.** Flujo incompresible y estacionario en un canal sin obstáculos. Las ecuaciones 4.60 y 4.61 se discretizan con diferencias finitas y se resuelven con Newton–Raphson.

Este es el **único código del proyecto**. Se actualiza en cada avance; los cambios de cada uno están en `CAMBIOS.md`.

Versión actual: **Avance 3**.

## Estructura

```
simulacion_canal/
├── main.py                  Punto de entrada: corre una simulación completa
├── canal/                   El paquete con el modelo y los métodos
│   ├── config.py            Parámetros por defecto (malla, V0, tolerancias, método)
│   ├── malla.py             Numeración de nodos e incógnitas, vector <-> matrices
│   ├── modelo.py            El canal: malla + V0 + condición de salida
│   ├── fronteras.py         Condiciones de frontera (valores, filas de F y de J)
│   ├── ecuaciones.py        Sistema no lineal F(v) = 0
│   ├── jacobiano.py         Jacobiano analítico (disperso) y numérico (verificación)
│   ├── inicializacion.py    Valores iniciales de vx y vy
│   ├── lineales.py          Métodos para J·h = −F: Richardson, Jacobi, Gauss–Seidel, gradientes
│   ├── newton.py            Newton–Raphson (inversa, directo o un método iterativo)
│   ├── convergencia.py      Pruebas de convergencia antes de iterar
│   └── graficas.py          Mapas de calor y gráficas de convergencia
├── experimentos/            Scripts de cada avance (comparaciones y figuras de la presentación)
│   ├── avance3_comparacion.py
│   └── avance3_figuras.py
├── tests/                   Pruebas automáticas (pytest)
├── resultados/              Figuras y archivos generados (se puede borrar y regenerar)
├── CAMBIOS.md               Qué cambió en cada avance
└── requirements.txt
```

Cada módulo depende solo de los de arriba en la lista: `malla` → `modelo` → `fronteras` → `ecuaciones` → `jacobiano` → `lineales` → `newton`.

## Instalación

```
pip install -r requirements.txt
```

## Cómo se usa

Todo se corre desde la carpeta `simulacion_canal/`.

```
python main.py                                    # 80x8 con Jacobi y los valores por defecto
python main.py --malla 400x40 --metodo jacobi     # malla oficial
python main.py --metodo inversa --diagnostico     # método del Avance 2 + pruebas de convergencia
python main.py --salida neumann --inicial difusion
python main.py --listar                           # métodos disponibles
python main.py --help                             # todas las opciones

python -m experimentos.avance3_comparacion --mallas 80x8 100x10 200x20 400x40
python -m experimentos.avance3_figuras

python -m pytest                                  # pruebas
```

Para usarlo desde otro script o un notebook:

```python
import canal as cn
c = cn.Canal(80, 8)                          # Nx, Ny, V0=1, salida="cero"
v0 = cn.condicion_inicial(c, "exponencial")
res = cn.newton(c, v0, "gauss-seidel")      # res.v, res.pasos, res.tiempo, res.convergio
vx, vy = c.malla.separar(res.v)
cn.graficas.campos(c, res.v, "resultados/mi_figura.png")
```

## Dónde cambiar cada cosa

| Quiero cambiar… | Archivo |
|---|---|
| Valores por defecto (malla, V0, tolerancias, método) | `canal/config.py` |
| Una condición de frontera | `canal/fronteras.py` (valores, residuo y filas del Jacobiano, los tres juntos) |
| Las ecuaciones de los nodos interiores | `canal/ecuaciones.py` **y** `canal/jacobiano.py` |
| Los valores iniciales | `canal/inicializacion.py` |
| Agregar un método para el sistema lineal | `canal/lineales.py`: escribir la función y registrarla en `METODOS` |
| El criterio de parada de Newton | `canal/newton.py` |
| Cómo se ven las figuras | `canal/graficas.py` |

## Reglas para los próximos avances

1. **No se crean copias del código** (`..._avance4.py`, `..._final.py`). Se modifica este proyecto.
2. Cada responsabilidad nueva va en su propio módulo dentro de `canal/`. Si un archivo crece demasiado o mezcla dos temas, se divide.
3. Los scripts que solo producen resultados de un avance van en `experimentos/avanceN_*.py`. Deben usar el paquete `canal`, sin copiar código.
4. Si se cambia `ecuaciones.py` o `fronteras.py`, se corre `python -m pytest`. La prueba del Jacobiano numérico detecta si el Jacobiano quedó desactualizado.
5. Al terminar un avance se actualizan `CAMBIOS.md` y `__version__` en `canal/__init__.py`.

## Modelo (resumen)

- **Malla:** nodos (i, j), con i = 0…Nx y j = 0…Ny, y h = 1. La oficial es de 400 × 40; para pruebas se usa 80 × 8.
- **Incógnitas:** v = [vx ; vy], con k = i·(Ny+1) + j para vx y N + k para vy.
- **Nodos interiores:**
  - F^x = 4vx − (vxE+vxW+vxN+vxS) + ½vx(vxE−vxW) + ½vy(vxN−vxS)
  - F^y igual, cambiando vx por vy en los vecinos.
- **Fronteras:**
  - Entrada: vx = V0, vy = 0.
  - Paredes de arriba y de abajo: vx = vy = 0.
  - Salida: vx = vy = 0.
- **Newton:** J(v_k)·h_k = −F(v_k) y v_(k+1) = v_k + h_k, hasta que ‖F‖∞ < 10⁻¹⁰. El sistema lineal se resuelve hasta ‖b − J·h‖∞ < 10⁻¹².
