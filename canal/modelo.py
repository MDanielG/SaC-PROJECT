"""El problema físico: la malla más los datos del canal (V0 y condición de salida)."""
from .malla import Malla


class Canal:
    def __init__(self, Nx=80, Ny=8, V0=1.0, salida="cero"):
        if salida not in ("cero", "neumann"):
            raise ValueError("salida debe ser 'cero' o 'neumann'")
        self.malla = Malla(Nx, Ny)
        self.V0 = V0
        self.salida = salida

    @classmethod
    def desde_parametros(cls, p):
        return cls(p.Nx, p.Ny, p.V0, p.salida)

    # atajos a la malla
    def __getattr__(self, nombre):
        if nombre == "malla":
            raise AttributeError(nombre)
        return getattr(self.malla, nombre)

    def __repr__(self):
        return f"Canal({self.malla}, V0={self.V0}, salida='{self.salida}')"
