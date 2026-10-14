import numpy as np 
from collections.abc import Callable

from .integrator import rk4, euler, symplectic_euler


class ODESolver:
    """Clase abstracta base para todos los metodos de integracion."""
    def __init__(self, function: Callable, y0: list|np.ndarray, t0: float, tf: float, dt: float):
        self.function=function
        self.initial_state=np.asarray(y0,dtype=np.float64)
        self.t0, self.tf, self.dt = t0,tf,dt
        self.method=""
        self.times=None
        self.trajectories=None
    def solve(self):
        raise NotImplementedError("Error: Ocupa ser implementado en una subclase")
    def __repr__(self):
        return f"<{self.__class__.__name__} | Metodo: {self.method} | Estado inicial: {self.initial_state} | tiempo inicial y final: {self.t0,self.tf} | dt: {self.dt}>"
    __str__=__repr__
    def print_state(self):
        pass
    def write_report(self):
        pass

class Euler(ODESolver):
    """Clase concreta que implementa el método de Euler con análisis de estabilidad."""
    def __init__(self, function, y0, t0, tf, dt):
        super().__init__(function, y0, t0, tf, dt)
        self.method="Euler"
        self.max_variation=0.0
        self.inestability_alarm=0.0
    def solve(self):
        """Ejecuta la funcion Euler hecha con el motor de Numba y luego procesa la estabilidad."""
        self.times, self.trajectories= euler(self.function, self.initial_state, self.t0, self.tf, self.dt)
        self.analyze_stability()
        return self.times, self.trajectories
    def analyze_stability(self):
        """
        Calcula la magnitud de los saltos y cuenta las alertas evaluando 
        toda la matriz de trayectoria ya resuelta.
        """        
        # En Python, el equivalente a tu calcularNormaSalto y verificarEstabilidad
        # se hace sobre TODA la matriz al mismo tiempo usando np.diff y np.linalg.norm.
        
        # Calcula la diferencia entre cada fila (Y_{n+1} - Y_n). Esto te da todos los saltos.
        
class RK4(ODESolver):
    pass
class Sympletic_Euler(ODESolver):
    pass