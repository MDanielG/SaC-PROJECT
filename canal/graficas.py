"""Mapas de calor y gráficas de convergencia (se guardan como PNG)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

UMBRAL_CERO = 1e-12      # |vy| por debajo de esto se considera redondeo y se grafica como 0
CMAP_VX = "viridis"      # colores de los mapas de vx (morado = 0, amarillo = V0); cambiar aquí


def _guardar(fig, ruta, dpi=150):
    os.makedirs(os.path.dirname(ruta) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(ruta, dpi=dpi)
    plt.close(fig)
    return ruta


def mapa(ax, campo, titulo, cmap=CMAP_VX, norm=None, etiqueta="m/s", vmin=None, vmax=None):
    im = ax.imshow(campo.T, origin="lower", aspect="auto", cmap=cmap, norm=norm,
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(titulo, fontsize=10, loc="left")
    ax.set_ylabel("y (índice j)")
    plt.colorbar(im, ax=ax, label=etiqueta, pad=0.01)
    return im


def mapa_vy(ax, vy, titulo):
    m = np.max(np.abs(vy))
    if m < UMBRAL_CERO:
        mapa(ax, np.zeros_like(vy), titulo + rf" — prácticamente nula (máx |vy| = {m:.0e})",
             cmap="RdBu_r", vmin=-1, vmax=1)
    else:
        mapa(ax, vy, titulo + f"  (máx |vy| = {m:.1e})", cmap="RdBu_r", norm=TwoSlopeNorm(0, -m, m))


def mapa_divergente(ax, campo, titulo, etiqueta="m/s"):
    l = max(np.max(np.abs(campo)), 1e-30)
    mapa(ax, campo, titulo, cmap="RdBu_r", norm=TwoSlopeNorm(0, -l, l), etiqueta=etiqueta)


def divergencia(vx, vy):
    return np.gradient(vx, axis=0) + np.gradient(vy, axis=1)


# ------------------------------------------------------------------
def campos(canal, v, ruta):
    """vx, vy, rapidez y divergencia de la solución."""
    vx, vy = canal.malla.separar(v)
    fig, axs = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    mapa(axs[0], vx, r"Componente $V^x_{i,j}$", vmin=0, vmax=canal.V0)
    mapa_vy(axs[1], vy, r"Componente $V^y_{i,j}$")
    mapa(axs[2], np.sqrt(vx**2 + vy**2), r"Rapidez $\sqrt{(V^x)^2+(V^y)^2}$", vmin=0, vmax=canal.V0)
    mapa_divergente(axs[3], divergencia(vx, vy), r"Divergencia discreta $\nabla_h\cdot V$ (diagnóstico)", "1/s")
    axs[3].set_xlabel("x (índice i)")
    return _guardar(fig, ruta)


def iteraciones(canal, resultado, ruta):
    """vx y vy en cada paso de Newton."""
    n = len(resultado.pasos)
    fig, axs = plt.subplots(n, 2, figsize=(12, 1.8 * n), sharex=True, squeeze=False)
    for r, p in enumerate(resultado.pasos):
        vx, vy = canal.malla.separar(p.v)
        mapa(axs[r, 0], vx, f"vx, paso {p.k}  (‖F‖∞ = {p.norma_inf:.1e})", vmin=0, vmax=canal.V0)
        mapa_vy(axs[r, 1], vy, f"vy, paso {p.k}")
    axs[-1, 0].set_xlabel("x (índice i)"); axs[-1, 1].set_xlabel("x (índice i)")
    return _guardar(fig, ruta, dpi=130)


def entrada(canal, v, v0, ruta):
    """Zoom a la entrada: solución, valor inicial y su diferencia."""
    vx, _ = canal.malla.separar(v)
    vx0, _ = canal.malla.separar(v0)
    corte = min(canal.Nx, 5 * canal.Ny) + 1
    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    mapa(axs[0], vx[:corte], rf"Solución $V^x$ (zoom 0 ≤ i ≤ {corte - 1})", vmin=0, vmax=canal.V0)
    mapa(axs[1], vx0[:corte], r"Valor inicial $V^x_{(0)}$", vmin=0, vmax=canal.V0)
    mapa_divergente(axs[2], (vx - vx0)[:corte], r"Diferencia $V^x - V^x_{(0)}$")
    axs[2].set_xlabel("x (índice i)")
    return _guardar(fig, ruta)


def log_vx(canal, v, ruta):
    vx, _ = canal.malla.separar(v)
    fig, ax = plt.subplots(figsize=(10, 2.6))
    with np.errstate(divide="ignore", invalid="ignore"):
        lv = np.log10(np.where(vx > 0, vx, np.nan))
    mapa(ax, lv, r"$\log_{10} V^x$ — decaimiento a lo largo del canal", etiqueta=r"log$_{10}$(m/s)")
    ax.set_xlabel("x (índice i)")
    return _guardar(fig, ruta)


def convergencia_newton(resultados, ruta, tol=1e-10):
    """‖F‖∞ por paso de Newton. `resultados` = un ResultadoNewton o una lista."""
    if not isinstance(resultados, (list, tuple)):
        resultados = [resultados]
    from .newton import nombre_metodo
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for r in resultados:
        ax.semilogy([p.k for p in r.pasos], [max(p.norma_inf, 1e-300) for p in r.pasos], "o-",
                    label=nombre_metodo(r.metodo))
    ax.axhline(tol, ls="--", color="gray", label=f"tolerancia {tol:.0e}")
    ax.set_xlabel("paso de Newton k"); ax.set_ylabel(r"$\|F(v_k)\|_\infty$")
    ax.set_title("Convergencia de Newton–Raphson"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    return _guardar(fig, ruta)


def convergencia_lineal(historiales, ruta, tol=1e-12, titulo="Residuo del sistema lineal"):
    """historiales = {nombre: (lista de ||b - J h||inf, convergió)}."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    xmax = 10
    for nombre, (hist, ok) in historiales.items():
        ax.semilogy(np.arange(len(hist)), hist, lw=1.6,
                    label=f"{nombre} ({len(hist) - 1} it.)" if ok else f"{nombre} (no converge)")
        if ok: xmax = max(xmax, len(hist))
    ax.axhline(tol, color="gray", lw=0.8, ls="--")
    ax.set_ylim(tol / 100, 1e4); ax.set_xlim(0, 1.1 * xmax)
    ax.set_xlabel("iteración del método lineal (m)"); ax.set_ylabel("‖b − J·h‖∞")
    ax.set_title(titulo); ax.legend(fontsize=8.5, loc="upper right"); ax.grid(alpha=0.3)
    return _guardar(fig, ruta)


def todas(canal, resultado, v0, carpeta, prefijo):
    """Genera todas las figuras de una corrida y devuelve las rutas."""
    base = os.path.join(carpeta, prefijo)
    return [campos(canal, resultado.v, f"{base}_campos.png"),
            iteraciones(canal, resultado, f"{base}_iteraciones.png"),
            entrada(canal, resultado.v, v0, f"{base}_entrada.png"),
            log_vx(canal, resultado.v, f"{base}_log.png"),
            convergencia_newton(resultado, f"{base}_convergencia.png")]
