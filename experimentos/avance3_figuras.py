"""Avance 3: figuras de la presentación (malla 80x8).

  - diagnóstico de convergencia (dominancia diagonal, norma, radio espectral)
  - residuo del sistema lineal en el primer paso de Newton, por método
  - mapas de calor de la solución con Gauss-Seidel
  - prueba de la tolerancia 1e-30 (se estanca en el redondeo)

Uso: python -m experimentos.avance3_figuras
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import canal as cn
from canal import lineales
from canal.ecuaciones import residuo
from canal.jacobiano import jacobiano

CARPETA = "resultados"


def main():
    canal = cn.Canal(80, 8)
    v0 = cn.condicion_inicial(canal, "exponencial")

    print("Diagnóstico de convergencia (valores iniciales):")
    cn.convergencia.imprimir(cn.convergencia.diagnostico(canal, v0))

    # 1. Solución con Gauss-Seidel y sus mapas
    res = cn.newton(canal, v0, "gauss-seidel")
    for ruta in cn.graficas.todas(canal, res, v0, CARPETA, "avance3_80x8_gauss_seidel"):
        print("  figura:", ruta)

    # 2. Residuo lineal en el primer paso de Newton
    J, b = jacobiano(canal, v0), -residuo(canal, v0)
    hist = {}
    for clave, (f, nombre) in lineales.METODOS.items():
        r = f(J, b, tol=1e-12, maxit=3000)
        hist[nombre] = (r.historial, r.convergio)
    print("  figura:", cn.graficas.convergencia_lineal(
        hist, os.path.join(CARPETA, "avance3_convergencia_interna.png"),
        titulo="Primer paso de Newton (80 × 8): residuo del sistema lineal"))

    # 3. Tolerancia 1e-30
    r30 = cn.newton(canal, v0, "directo", tol=1e-30, max_iter=8)
    j30 = lineales.jacobi(J, b, tol=1e-30, maxit=1500)
    eps = np.finfo(float).eps
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    axs[0].semilogy([p.k for p in r30.pasos], [p.norma_inf for p in r30.pasos], "o-")
    axs[0].set_title("Newton con tolerancia 10⁻³⁰"); axs[0].set_xlabel("paso de Newton k"); axs[0].set_ylabel("‖F‖∞")
    axs[1].semilogy(j30.historial)
    axs[1].set_title("Jacobi (primer paso) con tolerancia 10⁻³⁰"); axs[1].set_xlabel("iteración m"); axs[1].set_ylabel("‖b − J·h‖∞")
    for ax in axs:
        ax.axhline(eps, color="C3", ls="--", lw=1); ax.grid(alpha=0.3)
    axs[0].text(0.2, 3e-14, "épsilon de máquina 2,2 × 10⁻¹⁶", color="C3", fontsize=9)
    fig.tight_layout(); ruta = os.path.join(CARPETA, "avance3_tolerancia_1e-30.png")
    fig.savefig(ruta, dpi=150); plt.close(fig)
    print("  figura:", ruta)
    print("  Newton con 1e-30:", ", ".join(f"{p.norma_inf:.1e}" for p in r30.pasos))
    print(f"  Jacobi con 1e-30: mínimo {min(j30.historial):.1e}")


if __name__ == "__main__":
    main()
