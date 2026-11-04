"""Punto de entrada: corre una simulación completa.

Ejemplos:
  python main.py                                  # 80x8, Jacobi, valores por defecto
  python main.py --malla 400x40 --metodo gauss-seidel
  python main.py --metodo inversa --diagnostico   # método del Avance 2 + pruebas de convergencia
  python main.py --listar                         # muestra los métodos disponibles
"""
import argparse
import numpy as np
import canal as cn
from canal import Parametros, lineales
from canal.newton import METODOS_DIRECTOS


def leer_argumentos():
    p = Parametros()
    ap = argparse.ArgumentParser(description="Flujo en un canal 2D con Newton-Raphson")
    ap.add_argument("--malla", default=f"{p.Nx}x{p.Ny}", help="Nx x Ny, por ejemplo 80x8 o 400x40")
    ap.add_argument("--v0", type=float, default=p.V0, help="velocidad de entrada")
    ap.add_argument("--salida", default=p.salida, choices=["cero", "neumann"])
    ap.add_argument("--inicial", default=p.perfil_inicial, choices=["exponencial", "difusion", "constante"])
    ap.add_argument("--metodo", default=p.metodo,
                    choices=list(METODOS_DIRECTOS) + list(lineales.METODOS))
    ap.add_argument("--tol", type=float, default=p.tol_newton, help="tolerancia de Newton")
    ap.add_argument("--tol-lineal", type=float, default=p.tol_lineal, help="tolerancia del sistema lineal")
    ap.add_argument("--diagnostico", action="store_true", help="pruebas de convergencia antes de iterar")
    ap.add_argument("--sin-graficas", action="store_true")
    ap.add_argument("--carpeta", default="resultados", help="dónde guardar las figuras")
    ap.add_argument("--listar", action="store_true", help="mostrar los métodos disponibles y salir")
    a = ap.parse_args()
    a.Nx, a.Ny = (int(x) for x in a.malla.lower().split("x"))
    return a


def main():
    a = leer_argumentos()
    if a.listar:
        for k, v in METODOS_DIRECTOS.items(): print(f"  {k:28s} {v}")
        for k, (_, v) in lineales.METODOS.items(): print(f"  {k:28s} {v}")
        return

    canal = cn.Canal(a.Nx, a.Ny, a.v0, a.salida)
    print(canal)
    v0 = cn.condicion_inicial(canal, a.inicial)

    if a.diagnostico:
        print("\nPruebas de convergencia con los valores iniciales:")
        cn.convergencia.imprimir(cn.convergencia.diagnostico(canal, v0, densa=canal.n <= 3000))

    print(f"\nNewton-Raphson con {cn.nombre_metodo(a.metodo)}:")
    res = cn.newton(canal, v0, a.metodo, tol=a.tol, tol_lineal=a.tol_lineal, verbose=True)
    estado = "convergió" if res.convergio else f"NO convergió ({res.motivo})"
    print(f"\n{estado} en {res.iteraciones} pasos de Newton, "
          f"{res.iter_lineales_total} iteraciones lineales, {res.tiempo:.2f} s")

    vx, vy = canal.malla.separar(res.v)
    jc = canal.Ny // 2
    print(f"vx en el centro (j = {jc}):",
          ", ".join(f"i={i}: {vx[i, jc]:.4g}" for i in (1, 5, 10, 20, canal.Nx // 2) if i <= canal.Nx))
    print(f"máx |vy| = {np.max(np.abs(vy)):.1e}")

    if not a.sin_graficas and res.convergio:
        prefijo = f"canal_{a.Nx}x{a.Ny}_{a.metodo}"
        for ruta in cn.graficas.todas(canal, res, v0, a.carpeta, prefijo):
            print("  figura:", ruta)


if __name__ == "__main__":
    main()
