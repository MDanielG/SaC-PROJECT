"""Avance 3: compara la inversa (Avance 2) con los métodos lineales dentro de Newton.

Uso (desde la carpeta del proyecto):
  python -m experimentos.avance3_comparacion                  # mallas 80x8 y 100x10
  python -m experimentos.avance3_comparacion --mallas 80x8 200x20 400x40
Guarda resultados/avance3_comparacion.json y una tabla en pantalla.
"""
import argparse, json, os
import numpy as np
import canal as cn
from canal import lineales

METODOS = ["inversa", "directo"] + list(lineales.METODOS)
LIMITE_INVERSA = 9000          # incógnitas: por encima, la matriz densa no cabe en memoria


def comparar(Nx, Ny, metodos=METODOS, max_lineal=200_000):
    canal = cn.Canal(Nx, Ny)
    v0 = cn.condicion_inicial(canal, "exponencial")
    ref = cn.newton(canal, v0, "directo").v
    filas = []
    for met in metodos:
        if met == "inversa" and canal.n > LIMITE_INVERSA:
            filas.append({"malla": f"{Nx}x{Ny}", "n": canal.n, "metodo": met, "omitido": True,
                          "memoria_GB": canal.n**2 * 8 / 1e9})
            print(f"  {met:28s} omitido: la matriz densa ocupa {canal.n**2 * 8 / 1e9:.1f} GB")
            continue
        r = cn.newton(canal, v0, met, max_lineal=max_lineal)
        fila = {"malla": f"{Nx}x{Ny}", "n": canal.n, "metodo": met, "tiempo_s": r.tiempo,
                "convergio": r.convergio, "pasos_newton": r.iteraciones if r.convergio else None,
                "iter_lineales": [p.iter_lineales for p in r.pasos[:-1]] if r.convergio else
                                 [p.iter_lineales for p in r.pasos],
                "iter_lineales_total": r.iter_lineales_total,
                "norma_final": r.pasos[-1].norma_inf,
                "dif_vs_directo": float(np.max(np.abs(r.v - ref)))}
        filas.append(fila)
        txt = (f"{fila['pasos_newton']} pasos, {fila['iter_lineales_total']:6d} it. lineales, {r.tiempo:8.3f} s"
               if r.convergio else f"NO converge ({r.motivo})")
        print(f"  {cn.nombre_metodo(met):36s} {txt}")
    return filas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mallas", nargs="+", default=["80x8", "100x10"])
    ap.add_argument("--metodos", nargs="+", default=METODOS)
    ap.add_argument("--salida", default=os.path.join("resultados", "avance3_comparacion.json"))
    a = ap.parse_args()
    todas = []
    for m in a.mallas:
        Nx, Ny = (int(x) for x in m.split("x"))
        print(f"\nMalla {m}:")
        todas += comparar(Nx, Ny, a.metodos)
    os.makedirs(os.path.dirname(a.salida), exist_ok=True)
    json.dump(todas, open(a.salida, "w"), indent=1)
    print("\nGuardado en", a.salida)


if __name__ == "__main__":
    main()
