from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from .integrator import euler, rk4, symplectic_euler


class ODESolver(ABC):
    """
    Clase base abstracta para solucionadores de Problemas de Valor Inicial (Cauchy).

    Diseñada para operar tanto sobre trayectorias individuales y0 in R^N
    como sobre ensambles matriciales Y0 in R^{N x M}
    """

    def __init__(
        self, function: Callable, y0: list | np.ndarray, t0: float, tf: float, dt: float
    ) -> None:
        if dt <= 0.0:
            raise ValueError(
                f"El paso de integración dt debe ser estrictamente positivo (recibido: {dt})."
            )
        if tf <= t0:
            raise ValueError(f"El tiempo final tf ({tf}) debe ser mayor que t0 ({t0}).")
        self.function: Callable = function
        self.initial_state: np.ndarray = np.asarray(y0, dtype=np.float64)
        self.t0: float = float(t0)
        self.tf: float = float(tf)
        self.dt: float = float(dt)
        self.method: str = ""
        self.times: np.ndarray | None = None
        self.trajectories: np.ndarray | None = None
        self.derivatives: np.ndarray | None = None
        # Métricas de diagnóstico numérico y estabilidad
        self.max_variation: float = 0.0
        self.instability_alarms: int = 0

        # Semilla teórica Cota empírica de Lipschitz L y rigidez L*dt
        self.empirical_lipschitz: float = 0.0
        self.stiffness_indicator: float = 0.0

    @property
    def dimension(self) -> int:
        """Dimensión N del espacio de estados (o espacio fase)."""
        return self.initial_state.shape[0]

    @property
    def is_ensemble(self) -> bool:
        """Indica si el estado inicial es una matriz 2D (N, M) de múltiples condiciones iniciales."""
        return self.initial_state.ndim == 2

    @abstractmethod
    def _run_integrator(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Método interno que cada subclase implementa llamando a su kernel @njit respectivo."""
        raise NotImplementedError

    def solve(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Ejecuta el motor de integración compilado en Numba y realiza el análisis
        vectorizado de estabilidad y constante de Lipschitz sobre el flujo resultante.

        Retorna:
            tuple[np.ndarray, np.ndarray, np.ndarray]: (times, trajectories, derivatives)
        """
        self.times, self.trajectories, self.derivatives = self._run_integrator()
        self.analyze_stability()
        return self.times, self.trajectories, self.derivatives

    def get_derivatives(self) -> np.ndarray:
        """
        Devuelve la matriz/tensor de derivadas evaluadas en cada instante de tiempo.
        Si la simulación aún no se ha ejecutado, lanza un RuntimeError.
        """
        if self.derivatives is None:
            raise RuntimeError(
                "El sistema aún no ha sido resuelto. Ejecuta .solve() primero."
            )
        return self.derivatives

    def analyze_stability(self) -> None:
        """
        Evalúa vectorialmente la magnitud de los saltos de estado ||Y_{n+1} - Y_n||_2,
        contabiliza las alertas de crecimiento abrupto y estima la constante de Lipschitz L.
        Reemplaza a calcularNormaSalto() y verificarEstabilidad() de C++ sin usar bucles.
        """
        if self.trajectories is None or self.derivatives is None:
            raise RuntimeError(
                "No hay trayectorias para analizar. Ejecuta .solve() primero."
            )

        # 1. Diferencias finitas de estado: Delta Y_n = Y_{n+1} - Y_n a lo largo del eje temporal (axis=0)
        state_jumps = np.diff(self.trajectories, axis=0)

        # Si el estado es 1D (steps, N), calculamos la norma vectorial en axis=1.
        # Si es un ensamble 2D (steps, N, M), calculamos la norma de Frobenius por paso en axis=(1, 2).
        norm_axes = 1 if self.trajectories.ndim == 2 else (1, 2)
        jump_magnitudes = np.linalg.norm(state_jumps, axis=norm_axes)

        if jump_magnitudes.size == 0:
            return

        # A) Máxima variación registrada en un solo paso: max_n ||Y_{n+1} - Y_n||
        self.max_variation = float(np.max(jump_magnitudes))

        # B) Alertas de inestabilidad: pasos donde ||Delta Y_n|| > 2.0 * ||Delta Y_{n-1}||
        #    (con ||Delta Y_{n-1}|| > 0.0
        if jump_magnitudes.size > 1:
            prev_jumps = jump_magnitudes[:-1]
            curr_jumps = jump_magnitudes[1:]
            alarm_mask = (prev_jumps > 0.0) & (curr_jumps > 2.0 * prev_jumps)
            self.instability_alarms = int(np.sum(alarm_mask))
        else:
            self.instability_alarms = 0

        # C) Estimación empírica de la constante de Lipschitz local:
        #    L_emp = sup_n ||F(t_{n+1}, Y_{n+1}) - F(t_n, Y_n)|| / ||Y_{n+1} - Y_n||
        deriv_jumps = np.diff(self.derivatives, axis=0)
        deriv_jump_magnitudes = np.linalg.norm(deriv_jumps, axis=norm_axes)

        valid_steps = jump_magnitudes > 1e-15
        if np.any(valid_steps):
            local_lipschitz_ratios = (
                deriv_jump_magnitudes[valid_steps] / jump_magnitudes[valid_steps]
            )
            self.empirical_lipschitz = float(np.max(local_lipschitz_ratios))
            self.stiffness_indicator = self.empirical_lipschitz * self.dt
        else:
            self.empirical_lipschitz = 0.0
            self.stiffness_indicator = 0.0

    def print_state(self, step_index: int = -1) -> str:
        """
        Devuelve una cadena formateada con el método, tiempo, componentes del estado
        y componentes de la derivada en el índice temporal solicitado (por defecto el último).
        """
        if self.trajectories is None or self.times is None or self.derivatives is None:
            return "Simulación no ejecutada."

        t_val = self.times[step_index]
        state_val = self.trajectories[step_index].ravel()
        deriv_val = self.derivatives[step_index].ravel()

        state_str = " ".join(f"{val:.6f}" for val in state_val)
        deriv_str = " ".join(f"{val:.6f}" for val in deriv_val)
        return f"{self.method} {t_val:.6f} {state_str} {deriv_str}"

    def save_trajectory_txt(
        self, filename: str, include_method_col: bool = False
    ) -> None:
        """
        Guarda la tabla completa [t, Y_1, ..., Y_N, dY_1, ..., dY_N] de una sola vez.
        """
        if self.trajectories is None or self.times is None or self.derivatives is None:
            raise RuntimeError("No hay datos para guardar. Ejecuta .solve() primero.")

        steps = self.times.shape[0]
        flat_states = self.trajectories.reshape(steps, -1)
        flat_derivs = self.derivatives.reshape(steps, -1)
        data_matrix = np.column_stack((self.times, flat_states, flat_derivs))

        if not include_method_col:
            header = (
                f"Metodo: {self.method} | dt: {self.dt} | "
                f"Columnas: t, Estado (dim={flat_states.shape[1]}), Derivada (dim={flat_derivs.shape[1]})"
            )
            np.savetxt(filename, data_matrix, fmt="%.8e", header=header)
        else:
            num_cols = data_matrix.shape[1]
            row_format = f"{self.method} " + " ".join(["%.6f"] * num_cols)
            np.savetxt(filename, data_matrix, fmt=row_format)

    def write_report(self, filename: str) -> None:
        """
        Escribe en modo append ('a') las métricas de estabilidad y la cota de Lipschitz.
        """
        report_line = (
            f"Metodo: {self.method:<18} | dt: {self.dt:<10.6f} | "
            f"Max Variacion: {self.max_variation:<12.6e} | "
            f"Alertas Inestabilidad: {self.instability_alarms:<5d} | "
            f"L_emp: {self.empirical_lipschitz:<12.6e} | "
            f"Rigidez (L*dt): {self.stiffness_indicator:<10.6e}\n"
        )
        with open(filename, "a", encoding="utf-8") as file:
            file.write(report_line)

    def __repr__(self) -> str:
        status = "Resuelto" if self.trajectories is not None else "Pendiente"
        return (
            f"<{self.__class__.__name__} | Metodo: {self.method} | "
            f"Dimension y0: {self.initial_state.shape} | "
            f"Intervalo: [{self.t0}, {self.tf}] | dt: {self.dt} | Estado: {status}>"
        )

    __str__ = __repr__


class Euler(ODESolver):
    """Implementación concreta del método de Euler explícito de 1er Orden."""

    def __init__(
        self,
        function: Callable,
        y0: list | np.ndarray,
        t0: float,
        tf: float,
        dt: float,
    ) -> None:
        super().__init__(function, y0, t0, tf, dt)
        self.method = "Euler"

    def _run_integrator(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return euler(self.function, self.initial_state, self.t0, self.tf, self.dt)


class RK4(ODESolver):
    """Implementación concreta del método clásico de Runge-Kutta de 4to Orden."""

    def __init__(
        self,
        function: Callable,
        y0: list | np.ndarray,
        t0: float,
        tf: float,
        dt: float,
    ) -> None:
        super().__init__(function, y0, t0, tf, dt)
        self.method = "RK4"

    def _run_integrator(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return rk4(self.function, self.initial_state, self.t0, self.tf, self.dt)


class SymplecticEuler(ODESolver):
    """
    Implementación concreta del método de Euler Simpléctico para sistemas Hamiltonianos.
    Exige un espacio de estados de dimensión par N = 2n: [q_1, ..., q_n, p_1, ..., p_n].
    """

    def __init__(
        self,
        function: Callable,
        y0: list | np.ndarray,
        t0: float,
        tf: float,
        dt: float,
    ) -> None:
        super().__init__(function, y0, t0, tf, dt)
        if self.dimension % 2 != 0:
            raise ValueError(
                f"SymplecticEuler requiere un espacio fase de dimensión par (2n), "
                f"pero se recibió dimensión {self.dimension}."
            )
        self.method = "Euler_Simplectico"

    def _run_integrator(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return symplectic_euler(
            self.function, self.initial_state, self.t0, self.tf, self.dt
        )
