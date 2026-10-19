"""
propagators.py
==============
Jerarquía del Propagador Principal (Matriz Fundamental Canónica) Phi(t, t0) in GL^+(N, R)
que materializa el isomorfismo inverso de evaluación E_{t0}^{-1}: R^N -> S = ker(D_A).

Referencias del Tratado:
- Sección 2.2 : Teorema 2.4 (Isomorfismo canónico E_{t0}: S -> R^N).
- Sección 3.2 : Matriz Fundamental Psi(t), Wronskiano W(t) y Teorema 3.3
                (Fórmula de Abel-Liouville-Jacobi y Corolario 3.4 sobre sl(N, R)).
- Sección 3.3 : Definición 3.5 (Base Canónica de S y Propagador Principal Phi(t, t0)).
- Sección 4.1 : Serie de Peano-Baker (Ec. 43) y Grupo de Lie exp(A(t - t0)) (Teorema 4.2).
- Sección 6.2 : Teorema 6.4 (Construcción exacta de la base vía Algoritmo de Putzer).
- Sección 8.3 : Pipeline Tensorial (Primera contracción de Einstein Ec. 103).
"""

from abc import ABC, abstractmethod
from collections.abc import Callable

import numba as nb
import numpy as np

from .ode_solver import RK4, ODESolver
from .putzer_kernels import (
    build_putzer_polynomial_basis,
    compute_putzer_coefficients_grid,
    leja_order_eigenvalues,
)


class BasePropagator(ABC):
    """
    Clase base abstracta para el Propagador Principal Phi(t, t0) in GL^+(N, R),
    única solución del Problema de Cauchy Matricial Canónico (Ecuación 38):
        d/dt Phi(t, t0) = A(t) Phi(t, t0),    Phi(t0, t0) = I_N.

    Sus N columnas {phi_{(1)}(t), ..., phi_{(N)}(t)} constituyen la Base Canónica
    del espacio vectorial de soluciones S = ker(d/dt Id - A(t)) (Definición 3.5).
    """

    def __init__(self, dimension: int, t0: float = 0.0) -> None:
        if dimension <= 0:
            raise ValueError(
                f"La dimensión N del espacio de estados R^N debe ser positiva (recibido: {dimension})."
            )
        # Dimensión N = dim_R(S) = dim_R(R^N) por el Teorema Fundamental de Isomorfismo (Teorema 2.4)
        self.dimension: int = int(dimension)
        # Instante base t0 de la proyección de fibra E_{t0}(x) = x(t0) donde Phi(t0, t0) = I_N
        self.t0: float = float(t0)

    @abstractmethod
    def evaluate_basis(self, t_grid: np.ndarray) -> np.ndarray:
        """
        Evalúa el Propagador Principal Phi(t_alpha, t0) sobre la malla temporal t_grid.

        Parámetros:
            t_grid : np.ndarray de forma (K,) con los instantes temporales {t_0, ..., t_{K-1}}.

        Retorna:
            np.ndarray (3D, float64, forma (K, N, N)):
                Tensor Phi[alpha, i, j] = Phi^i_j(t_alpha, t0), donde para cada instante alpha,
                la columna j-ésima Phi[alpha, :, j] = E_{t0}^{-1}(e_{(j+1)})(t_alpha) es el
                (j+1)-ésimo vector de la base canónica de S evaluado en t_alpha (Ecuación 36).
        """
        raise NotImplementedError

    @abstractmethod
    def trace_integral(self, t_grid: np.ndarray) -> np.ndarray:
        """
        Evalúa la integral acumulada de la traza del generador infinitesimal A(s)
        que gobierna la dilatación de volumen en la Fórmula de Abel-Liouville-Jacobi (Ecuación 29):
            I(t_alpha) = int_{t0}^{t_alpha} Tr(A(s)) ds.

        Retorna:
            np.ndarray (1D, float64, forma (K,)): Vector con la integral de la traza en cada t_alpha.
        """
        raise NotImplementedError

    @staticmethod
    def compute_wronskian(phi_tensor: np.ndarray) -> np.ndarray:
        """
        Evalúa el Determinante Wronskiano W(t_alpha) = det(Phi(t_alpha, t0)) (Ecuación 27).

        Geométricamente, |W(t_alpha)| es el volumen N-dimensional del paralelepípedo
        generado en R^N por los N vectores de la base canónica {phi_{(1)}(t_alpha), ..., phi_{(N)}(t_alpha)},
        con orientación positivamente preservada (W(t) > 0 => Phi(t) in GL^+(N, R), Corolario 3.4).

        Retorna:
            np.ndarray (1D, float64, forma (K,)): Determinante W(t_alpha) en cada instante.
        """
        phi_arr = np.asarray(phi_tensor, dtype=np.float64)
        if phi_arr.ndim != 3 or phi_arr.shape[1] != phi_arr.shape[2]:
            raise ValueError(
                f"El tensor Phi debe tener forma (K, N, N), pero se recibió {phi_arr.shape}."
            )
        # Cálculo vectorizado del determinante sobre las K matrices N x N (ejes 1 y 2)
        return np.linalg.det(phi_arr)

    def verify_abel_liouville(
        self, t_grid: np.ndarray, phi_tensor: np.ndarray | None = None
    ) -> float:
        """
        Verifica numéricamente el Teorema de Abel-Liouville-Jacobi (Teorema 3.3, Ecuación 29):
            det(Phi(t_alpha, t0)) = det(Phi(t0, t0)) * exp( int_{t0}^{t_alpha} Tr(A(s)) ds )
                                  = exp( int_{t0}^{t_alpha} Tr(A(s)) ds ),   pues det(I_N) = 1.

        Retorna:
            float: Error relativo uniforme máximo max_alpha |W_num(t_alpha) - W_exact(t_alpha)| / |W_exact(t_alpha)|.
        """
        t_arr = np.asarray(t_grid, dtype=np.float64)
        if phi_tensor is None:
            phi_tensor = self.evaluate_basis(t_arr)

        # Wronskiano numérico W_num(t_alpha) = det(Phi(t_alpha, t0))
        w_num = self.compute_wronskian(phi_tensor)

        # Wronskiano teórico exacto por la identidad de Jacobi dW/dt = Tr(A(t)) W(t) (Ecuación 28)
        int_trace = self.trace_integral(t_arr)
        w_exact = np.exp(int_trace)

        # Cota de error relativo protegida contra subdesbordamiento (underflow) cuando Tr(A)*t << 0
        denom = np.maximum(np.abs(w_exact), 1e-30)
        rel_error = np.abs(w_num - w_exact) / denom
        return float(np.max(rel_error))

    def is_volume_preserving(
        self, t_grid: np.ndarray | None = None, tol: float = 1e-12
    ) -> bool:
        """
        Caracteriza si el flujo conserva el volumen en el espacio de estados (Corolario 3.4):
            det(Phi(t, t0)) == 1  <==>  Tr(A(t)) == 0  <==>  A(t) in sl(N, R),
        donde sl(N, R) es el Álgebra de Lie del Grupo Especial Lineal SL(N, R).
        """
        if t_grid is None:
            t_sample = np.array([self.t0, self.t0 + 1.0], dtype=np.float64)
        else:
            t_sample = np.asarray(t_grid, dtype=np.float64)

        int_trace = self.trace_integral(t_sample)
        return bool(np.all(np.abs(int_trace) <= tol))


class PutzerPropagator(BasePropagator):
    """
    Propagador Algebraico Exacto Phi(t, t0) = exp(A * (t - t0)) para sistemas
    lineales autónomos dx/dt = A x (Secciones 4, 6 y 8.3 del Tratado).

    Construye la Base Canónica de S sin calcular vectores propios, sin invertir
    matrices de cambio de base y sin truncar series de Taylor sobre A, utilizando
    el colapso dimensional del Teorema de Cayley-Hamilton (dim C[A] <= N) y la
    base polinomial de Newton-Horner {P_0, ..., P_{N-1}} (Teorema 6.4).
    """

    def __init__(self, a_matrix: np.ndarray, t0: float = 0.0) -> None:
        a_arr = np.asarray(a_matrix, dtype=np.float64)
        if a_arr.ndim != 2 or a_arr.shape[0] != a_arr.shape[1]:
            raise ValueError(
                f"El endomorfismo A debe ser una matriz cuadrada 2D (N, N), pero se recibió {a_arr.shape}."
            )

        super().__init__(dimension=a_arr.shape[0], t0=t0)
        self.a_matrix: np.ndarray = a_arr

        # Invariante escalar de primer orden: Tr(A) = sum_{i=1}^N A^i_i (gobierna el Wronskiano, Ec. 30)
        self.trace_a: float = float(np.trace(self.a_matrix))

        # Constante global de Lipschitz exacta del campo lineal: L = ||A||_op = sigma_max(A) (Proposición 2.1)
        self.operator_norm: float = float(np.linalg.norm(self.a_matrix, ord=2))

        # 1. Espectro sigma(A) = {lambda_1, ..., lambda_N} ordenado por puntos de Leja (Sección 8.2.1)
        raw_eigs = np.linalg.eigvals(self.a_matrix).astype(np.complex128)
        self.lambdas: np.ndarray = leja_order_eigenvalues(raw_eigs)

        # 2. Precalculo único del tensor de operadores de Putzer P[k, i, j] = (P_k)^i_j (Ec. 101)
        #    y diagnóstico algebraico del Teorema de Cayley-Hamilton ||p_A(A)||_F / ||A||_F^N (Teorema 6.1)
        self.p_tensor: np.ndarray
        self.cayley_hamilton_residual: float
        self.p_tensor, self.cayley_hamilton_residual = build_putzer_polynomial_basis(
            self.a_matrix, self.lambdas
        )

    def evaluate_basis(self, t_grid: np.ndarray) -> np.ndarray:
        """
        Ensambla el Tensor de la Base Canónica Phi[alpha, i, j] = exp(A * (t_alpha - t0))^i_j
        mediante la suma finita de Putzer (Teorema 6.4, Ecuación 70 y Ecuación 103):
            Phi^i_j(t_alpha, t0) = sum_{k=0}^{N-1} r_{k+1}(t_alpha - t0) * (P_k)^i_j.
        """
        t_arr = np.asarray(t_grid, dtype=np.float64).ravel()

        # Etapa 2 del Pipeline Tensorial (Ec. 102): Matriz dinámica R[alpha, k] = r_{k+1}(t_alpha - t0)
        r_matrix = compute_putzer_coefficients_grid(self.lambdas, t_arr, self.t0)

        # Etapa 3 del Pipeline Tensorial (Ec. 103): Contracción de Einstein sobre el índice modal k
        # 'ak,kij->aij' suma sobre k in {0, ..., N-1} para cada instante a y cada entrada matricial (i, j)
        phi_complex = np.einsum(
            "ak,kij->aij", r_matrix, self.p_tensor, optimize=True
        )

        # Como A in R^{N x N} y Phi(t0, t0) = I_N in R^{N x N}, por unicidad (Proposición 2.1)
        # la parte imaginaria residual (~1e-16j de pares conjugados) se anula idénticamente
        return np.real(phi_complex)

    def trace_integral(self, t_grid: np.ndarray) -> np.ndarray:
        """
        Para A constante, int_{t0}^{t_alpha} Tr(A) ds = Tr(A) * (t_alpha - t0) (Ecuación 30).
        """
        t_arr = np.asarray(t_grid, dtype=np.float64).ravel()
        return self.trace_a * (t_arr - self.t0)


@nb.njit
def _build_matrix_rhs_kernel(a_func: Callable) -> Callable:
    """
    Fábrica compilada en Numba que transforma un operador matricial dependiente
    del tiempo A(t) en el campo vectorial matricial derecho F(t, Phi) = A(t) @ Phi
    de la Ecuación Diferencial Matricial Homogénea dPhi/dt = A(t)Phi (Ecuación 26).
    """

    @nb.njit
    def matrix_rhs(t: float, phi_mat: np.ndarray) -> np.ndarray:
        return a_func(t) @ phi_mat

    return matrix_rhs


class NumericalMatrixPropagator(BasePropagator):
    """
    Semilla No Autónoma para el Propagador Principal Phi(t, t0) de sistemas
    lineales con generador dependiente del tiempo dx/dt = A(t) x (Secciones 2, 3 y 4.1).

    Matemática:
        Cuando [A(t_1), A(t_2)] != 0, el propagador viene dado por la serie de
        Peano-Baker (Ecuación 43) y no por exp(A(t - t0)). Sin embargo, el Teorema 2.4
        (dim S = N) y la Definición 3.5 siguen vigentes: Phi(t, t0) es la única solución
        del Problema de Cauchy Matricial Canónico (Ecuación 38):
            d/dt Phi(t, t0) = A(t) Phi(t, t0),    Phi(t0, t0) = I_N.

        Esta clase reutiliza ODESolver (por defecto RK4) pasando Y0 = I_N in R^{N x N}
        para integrar la base canónica una sola vez, permitiendo luego simular M
        condiciones iniciales por contracción tensorial igual que en el caso constante.
    """

    def __init__(
        self,
        a_func: Callable[[float], np.ndarray],
        dimension: int,
        t0: float,
        tf: float,
        dt: float,
        solver_cls: type[ODESolver] = RK4,
        is_numba_a_func: bool = True,
    ) -> None:
        super().__init__(dimension=dimension, t0=t0)
        self.a_func: Callable[[float], np.ndarray] = a_func
        self.tf: float = float(tf)
        self.dt: float = float(dt)
        self.solver_cls: type[ODESolver] = solver_cls

        # Si a_func(t) está compilada con @njit, construimos el kernel matricial F(t, Phi) = A(t) @ Phi
        if is_numba_a_func:
            self.matrix_rhs: Callable = _build_matrix_rhs_kernel(a_func)
        else:
            self.matrix_rhs = lambda t, phi_mat: a_func(t) @ phi_mat

        # Condición inicial canónica Phi(t0, t0) = I_N (Ecuación 38)
        identity_n = np.eye(self.dimension, dtype=np.float64)

        # Instanciamos el integrador matricial sobre el espacio End(R^N) ~= R^{N x N}
        self.matrix_solver: ODESolver = self.solver_cls(
            function=self.matrix_rhs,
            y0=identity_n,
            t0=self.t0,
            tf=self.tf,
            dt=self.dt,
        )

    def evaluate_basis(self, t_grid: np.ndarray | None = None) -> np.ndarray:
        """
        Resuelve una única vez la EDO matricial dPhi/dt = A(t)Phi partiendo de I_N
        y retorna el tensor de la Base Canónica Phi de forma (K, N, N).
        """
        if self.matrix_solver.trajectories is None:
            self.matrix_solver.solve()

        assert self.matrix_solver.trajectories is not None
        return self.matrix_solver.trajectories

    def trace_integral(self, t_grid: np.ndarray | None = None) -> np.ndarray:
        """
        Evalúa mediante cuadratura trapezoidal compuesta la integral acumulada
        I(t_alpha) = int_{t0}^{t_alpha} Tr(A(s)) ds de la Fórmula general de
        Abel-Liouville-Jacobi (Teorema 3.3, Ecuación 29).
        """
        if t_grid is None:
            if self.matrix_solver.times is None:
                self.matrix_solver.solve()
            assert self.matrix_solver.times is not None
            t_arr = self.matrix_solver.times
        else:
            t_arr = np.asarray(t_grid, dtype=np.float64).ravel()

        k_steps = t_arr.shape[0]
        traces = np.empty(k_steps, dtype=np.float64)
        for alpha in range(k_steps):
            traces[alpha] = float(np.trace(self.a_func(float(t_arr[alpha]))))

        # Cuadratura trapezoidal acumulada: int_{t_0}^{t_alpha} Tr(A(s)) ds
        int_trace = np.zeros(k_steps, dtype=np.float64)
        if k_steps > 1:
            dt_steps = np.diff(t_arr)
            trapz_increments = 0.5 * (traces[:-1] + traces[1:]) * dt_steps
            int_trace[1:] = np.cumsum(trapz_increments)

        return int_trace
