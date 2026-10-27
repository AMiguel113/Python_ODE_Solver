"""
linear_simulator.py
===================
Capa de Simulación Algebraica del Espacio Vectorial de Soluciones S = ker(d/dt Id - A)
mediante Proyección en la Base Canónica Phi(t, t0) y Contracciones Tensoriales.

Referencias del Tratado:
- Sección 2.2 : Operador Diferencial D_A = d/dt Id - A y Teorema 2.4 (Isomorfismo E_{t0}: S -> R^N).
- Sección 3.3 : Teorema 3.6 (Representación por Coordenadas en la Base Canónica: x(t; c) = Phi(t, t0) c).
- Sección 5.2 : Teorema 5.2 (Descomposición Primaria en Subespacios Propios Generalizados y Jordan-Chevalley).
- Sección 5.3 : Teorema 5.3 (Criterio Espectral de Estabilidad de Lyapunov: Hurwitz, Marginal y Secular).
- Sección 8.3 : Etapa 4 del Pipeline Tensorial (Segunda Contracción de Einstein Ec. 104).
"""

from dataclasses import dataclass

import numpy as np

from .ode_solver import DynamicalTrajectoryBase
from .propagators import BasePropagator, NumericalMatrixPropagator, PutzerPropagator


@dataclass(frozen=True)
class LyapunovSpectrumReport:
    """
    Certificado algebraico inmutable del destino asintótico (t -> +inf) del espacio
    de soluciones S = ker(d/dt Id - A) según el Teorema 5.3 del Tratado (Sección 5.3).

    Atributos:
        spectral_abscissa   : alpha(A) = max_j Re(lambda_j) (tasa exponencial dominante).
        regime              : Clasificación ('ASYMPTOTICALLY_STABLE_HURWITZ', 'LYAPUNOV_MARGINALLY_STABLE',         'SECULAR_POLYNOMIAL_UNSTABLE','EXPONENTIALLY_UNSTABLE').
        is_hurwitz          : True si y solo si alpha(A) < 0 (lim_{t->inf} ||e^{At} c|| = 0).
        is_lyapunov_stable  : True si y solo si sup_{t>=0} ||e^{At}||_op < inf.
        has_marginal_jordan : True si existe algún lambda_j con Re(lambda_j) == 0 y N_j != 0 (g_j < m_j).
        eigenvalues         : Espectro completo sigma(A) in C^N ordenado por puntos de Leja.
        marginal_modes_info : Lista de tuplas (lambda_j, m_j_algebraica, g_j_geometrica) sobre Re(lambda) == 0.
    """

    spectral_abscissa: float
    regime: str
    is_hurwitz: bool
    is_lyapunov_stable: bool
    has_marginal_jordan: bool
    eigenvalues: np.ndarray
    marginal_modes_info: tuple[tuple[complex, int, int], ...]


class LinearSpaceSimulator(DynamicalTrajectoryBase):
    """
    Simulador Algebraico del Espacio Vectorial de Soluciones S = ker(D_A) (dim_R S = N)
    para sistemas dinámicos lineales dx/dt = A(t) x con estado inicial c0 in R^N
    o ensamble masivo C0 = [c^{(1)} | ... | c^{(M)}] in R^{N x M}.

    En lugar de integrar curvas individuales paso a paso, construye una única vez la
    Base Canónica Phi(t, t0) = E_{t0}^{-1}(I_N) in GL^+(N, R) y proyecta cualquier
    volumen de condiciones iniciales mediante contracciones tensoriales (Teorema 3.6, Ec. 104).
    """

    def __init__(
        self,
        generator_or_propagator: np.ndarray | BasePropagator,
        y0: list | np.ndarray | None = None,
        t0: float = 0.0,
        tf: float = 1.0,
        dt: float | None = None,
    ) -> None:
        # 1. Resolución polimórfica del Propagador Principal Phi(t, t0) in GL^+(N, R)
        if isinstance(generator_or_propagator, BasePropagator):
            self.propagator: BasePropagator = generator_or_propagator
            n_dim = self.propagator.dimension
            t0_effective = self.propagator.t0
            method_name = f"LinearSim[{self.propagator.__class__.__name__}]"
        else:
            a_arr = np.asarray(generator_or_propagator, dtype=np.float64)
            self.propagator = PutzerPropagator(a_matrix=a_arr, t0=t0)
            n_dim = self.propagator.dimension
            t0_effective = float(t0)
            method_name = "Putzer_Tensor_Simulator"

        # Si el usuario instancia el espacio S antes de elegir un estado inicial particular c0,
        # adoptamos la base canónica I_N in R^{N x N} como estado inicial por defecto (Def. 3.5).
        if y0 is None:
            y0_init: np.ndarray = np.eye(n_dim, dtype=np.float64)
        else:
            y0_init = np.asarray(y0, dtype=np.float64)
            if y0_init.shape[0] != n_dim:
                raise ValueError(
                    f"Incompatibilidad de dimensión en el isomorfismo E_{{t0}}: el espacio S tiene "
                    f"dimensión N={n_dim}, pero el estado inicial tiene {y0_init.shape[0]} filas."
                )

        # Si el propagador es no autónomo y trae su propio horizonte tf/dt, sincronizamos
        if isinstance(self.propagator, NumericalMatrixPropagator):
            tf_effective = self.propagator.tf if tf <= t0_effective else float(tf)
            dt_effective = self.propagator.dt if dt is None else float(dt)
        else:
            tf_effective = float(tf) if tf > t0_effective else t0_effective + 1.0
            dt_effective = dt

        super().__init__(
            y0=y0_init,
            t0=t0_effective,
            tf=tf_effective,
            dt=dt_effective,
            method=method_name,
        )

        # Tensor de la Base Canónica Phi[alpha, i, j] = Phi^i_j(t_alpha, t0) cacheado tras simular
        self.canonical_basis_tensor: np.ndarray | None = None

        # Diagnóstico espectral de Lyapunov a priori (Teorema 5.3) si el generador A es autónomo
        self.lyapunov_report: LyapunovSpectrumReport | None = (
            self.analyze_lyapunov_spectrum()
            if isinstance(self.propagator, PutzerPropagator)
            else None
        )

    def analyze_lyapunov_spectrum(
        self,
        imag_axis_tol: float = 1e-9,
        cluster_tol: float = 1e-5,
        svd_rank_tol: float = 1e-9,
    ) -> LyapunovSpectrumReport:
        """
        Dictamina a priori el comportamiento asintótico (t -> +inf) de todo el espacio
        de soluciones S = ker(d/dt Id - A) aplicando el Teorema 5.3 (Sección 5.3) y la
        descomposición de Jordan-Chevalley A|_{E_{lambda_j}} = S_j + N_j (Teorema 5.2).

        Matemática:
            Sea alpha(A) = max_j Re(lambda_j) la abscisa espectral de A:
            1. Si alpha(A) < -imag_axis_tol: Todos los modos decaen exponencialmente => Matriz de Hurwitz (Estabilidad Asintótica).
            2. Si alpha(A) > +imag_axis_tol:
               Existe al menos un modo con Re(lambda_j) > 0 => Inestabilidad Exponencial.
            3. Si |alpha(A)| <= imag_axis_tol:
               Inspeccionamos todos los autovalores marginales con |Re(lambda_j)| <= imag_axis_tol.
               Para cada raíz distinta lambda_j sobre el eje imaginario con multiplicidad
               algebraica m_j = dim E_{lambda_j}, calculamos su multiplicidad geométrica:
                   g_j = dim ker(A - lambda_j I_N) = N - rank(A - lambda_j I_N)
               mediante los valores singulares (SVD) de (A - lambda_j I_N).
               - Si g_j == m_j para todo modo marginal, el operador nilpotente N_j == 0
                 (bloques de Jordan 1x1) => Estabilidad Marginal / Conservativa de Lyapunov.
               - Si g_j < m_j para algún modo marginal, N_j != 0 y emergen términos seculares
                 polinomiales t^k e^{i omega_j t} (Ec. 57) => Inestabilidad Secular Polinomial.
        """
        if not isinstance(self.propagator, PutzerPropagator):
            raise TypeError(
                "El criterio espectral de Lyapunov (Teorema 5.3) requiere un generador autónomo "
                "A constante (PutzerPropagator). Para A(t) dependiente del tiempo, los autovalores "
                "instantáneos no determinan la estabilidad asintótica."
            )

        a_mat = self.propagator.a_matrix
        n_dim = self.propagator.dimension
        lambdas = self.propagator.lambdas

        # Abscisa espectral: alpha(A) = max_{1 <= j <= N} Re(lambda_j) (Teorema 5.3)
        real_parts = np.real(lambdas)
        spectral_abscissa = float(np.max(real_parts))

        # Caso 1: Matriz de Hurwitz (alpha(A) < 0 estrictamente)
        if spectral_abscissa < -imag_axis_tol:
            return LyapunovSpectrumReport(
                spectral_abscissa=spectral_abscissa,
                regime="ASYMPTOTICALLY_STABLE_HURWITZ",
                is_hurwitz=True,
                is_lyapunov_stable=True,
                has_marginal_jordan=False,
                eigenvalues=lambdas.copy(),
                marginal_modes_info=(),
            )

        # Caso 3a: Inestabilidad Exponencial (alpha(A) > 0 estrictamente)
        if spectral_abscissa > imag_axis_tol:
            return LyapunovSpectrumReport(
                spectral_abscissa=spectral_abscissa,
                regime="EXPONENTIALLY_UNSTABLE",
                is_hurwitz=False,
                is_lyapunov_stable=False,
                has_marginal_jordan=False,
                eigenvalues=lambdas.copy(),
                marginal_modes_info=(),
            )

        # Caso 2 y 3b: Frontera Crítica |alpha(A)| <= imag_axis_tol (Modos sobre el eje imaginario i*R)
        # Filtramos las raíces que yacen sobre el eje imaginario |Re(lambda)| <= imag_axis_tol
        marginal_mask = np.abs(real_parts) <= imag_axis_tol
        marginal_eigs = lambdas[marginal_mask]

        # Agrupamos raíces marginales indistinguibles dentro de cluster_tol para determinar m_j
        clusters: list[list[complex]] = []
        for eig in marginal_eigs:
            placed = False
            for cluster in clusters:
                if abs(eig - cluster[0]) <= cluster_tol:
                    cluster.append(complex(eig))
                    placed = True
                    break
            if not placed:
                clusters.append([complex(eig)])

        identity_n = np.eye(n_dim, dtype=np.complex128)
        a_complex = a_mat.astype(np.complex128)

        has_marginal_jordan = False
        marginal_info_list: list[tuple[complex, int, int]] = []

        for cluster in clusters:
            # Multiplicidad algebraica m_j = número de raíces en el cluster y centroide espectral
            m_alg = len(cluster)
            lam_center = complex(np.mean(cluster))
            # Proyectamos exactamente sobre el eje imaginario i * Im(lam_center) para máxima precisión
            lam_imag = 0.0 + 1j * lam_center.imag

            # Matriz característica desplazada: M_j = A - lambda_j I_N in C^{N x N}
            shifted_matrix = a_complex - lam_imag * identity_n

            # Valores singulares sigma_1 >= sigma_2 >= ... >= sigma_N >= 0 de (A - lambda_j I_N)
            singular_values = np.linalg.svd(shifted_matrix, compute_uv=False)
            sigma_scale = max(float(singular_values[0]), 1.0) if singular_values.size > 0 else 1.0
            null_tol = svd_rank_tol * sigma_scale

            # Multiplicidad geométrica g_j = dim ker(A - lambda_j I_N) = #{sigma_r <= null_tol}
            g_geom = int(np.sum(singular_values <= null_tol))
            # Por teoría espectral, toda raíz del polinomio característico tiene al menos g_j >= 1
            g_geom = max(g_geom, 1)

            marginal_info_list.append((lam_imag, m_alg, g_geom))

            # Si g_geom < m_alg, entonces ker(A - lambda_j I_N) != E_{lambda_j}, por lo que
            # el operador nilpotente N_j = (A - lambda_j I_N)|_{E_{lambda_j}} != 0 (Teorema 5.2 y 5.3)
            if g_geom < m_alg:
                has_marginal_jordan = True

        if has_marginal_jordan:
            regime = "SECULAR_POLYNOMIAL_UNSTABLE"
            is_lyap_stable = False
        else:
            regime = "LYAPUNOV_MARGINALLY_STABLE"
            is_lyap_stable = True

        return LyapunovSpectrumReport(
            spectral_abscissa=spectral_abscissa,
            regime=regime,
            is_hurwitz=False,
            is_lyapunov_stable=is_lyap_stable,
            has_marginal_jordan=has_marginal_jordan,
            eigenvalues=lambdas.copy(),
            marginal_modes_info=tuple(marginal_info_list),
        )

    def simulate(
        self,
        c0: list | np.ndarray | None = None,
        t_grid: np.ndarray | None = None,
        return_basis: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Simula de forma simultánea una condición inicial c0 in R^N o un ensamble masivo
        C0 = [c^{(1)} | ... | c^{(M)}] in R^{N x M} proyectando sobre la Base Canónica
        Phi(t, t0) del espacio de soluciones S (Teorema 3.6 y Sección 8.3, Ec. 104).

        Matemática:
            1. Construcción de la Base Canónica de S:
                   Phi[alpha, i, j] = Phi^i_j(t_alpha, t0) in R^{K x N x N}.
            2. Segunda Contracción Tensorial (Coordenadas en la Base Canónica, Ec. 39 y 104):
               - Para una fibra 1D c0 in R^N:
                     x^i(t_alpha) = sum_{j=1}^N Phi^i_j(t_alpha, t0) c0^j
               - Para un ensamble 2D C0 in R^{N x M}:
                     X^i_m(t_alpha) = sum_{j=1}^N Phi^i_j(t_alpha, t0) C0^j_m
            3. Derivadas exactas en el núcleo ker(D_A) (D_A x = 0 <==> dx/dt = A(t) x):
               - Para A constante: dX^i_m(t_alpha)/dt = sum_{k=1}^N A^i_k X^k_m(t_alpha).
               - Para A(t) variable: dX^i_m(t_alpha)/dt = sum_{k=1}^N A^i_k(t_alpha) X^k_m(t_alpha).

        Parámetros:
            c0           : Estado inicial (N,) o ensamble (N, M). Si es None, usa self.initial_state.
            t_grid       : Vector 1D (K,) de instantes de evaluación. Si es None, construye malla con t0, tf, dt.
            return_basis : Si es True, devuelve también el tensor 3D Phi de la base canónica.

        Retorna:
            (times, trajectories, derivatives) o (times, trajectories, derivatives, canonical_basis_tensor).
        """
        # 1. Actualización opcional del estado inicial c0 o ensamble C0
        if c0 is not None:
            c0_arr = np.asarray(c0, dtype=np.float64)
            if c0_arr.ndim not in (1, 2) or c0_arr.shape[0] != self.propagator.dimension:
                raise ValueError(
                    f"El estado inicial c0 debe tener forma ({self.propagator.dimension},) o "
                    f"({self.propagator.dimension}, M), pero se recibió {c0_arr.shape}."
                )
            self.initial_state = c0_arr

        # 2. Resolución de la malla temporal t_grid de forma (K,)
        if t_grid is None:
            if isinstance(self.propagator, NumericalMatrixPropagator):
                phi_tensor = self.propagator.evaluate_basis()
                assert self.propagator.matrix_solver.times is not None
                t_arr = self.propagator.matrix_solver.times
            else:
                dt_val = self.dt if self.dt is not None else (self.tf - self.t0) / 100.0
                n_intervals = int(np.floor((self.tf - self.t0) / dt_val))
                t_arr = self.t0 + np.arange(n_intervals + 1, dtype=np.float64) * dt_val
                phi_tensor = self.propagator.evaluate_basis(t_arr)
        else:
            t_arr = np.asarray(t_grid, dtype=np.float64).ravel()
            if t_arr.size == 0:
                raise ValueError("La malla temporal t_grid no puede estar vacía.")
            phi_tensor = self.propagator.evaluate_basis(t_arr)

        self.times = t_arr
        self.t0 = float(t_arr[0])
        self.tf = float(t_arr[-1])
        self.canonical_basis_tensor = phi_tensor

        # 3. Segunda Contracción Tensorial (Teorema 3.6, Ecuación 104) y Derivada Exacta en ker(D_A)
        if not self.is_ensemble:
            # Caso 1D: una sola fibra c0 in R^N -> trajectories y derivatives tienen forma (K, N)
            self.trajectories = np.einsum(
                "aij,j->ai", phi_tensor, self.initial_state, optimize=True
            )
            if isinstance(self.propagator, PutzerPropagator):
                self.derivatives = np.einsum(
                    "ik,ak->ai",
                    self.propagator.a_matrix,
                    self.trajectories,
                    optimize=True,
                )
            elif isinstance(self.propagator, NumericalMatrixPropagator):
                # Para dPhi/dt = A(t)Phi, como x(t) = Phi(t)c0, entonces dx/dt = dPhi/dt * c0
                assert self.propagator.matrix_solver.derivatives is not None
                dphi_tensor = self.propagator.matrix_solver.derivatives
                self.derivatives = np.einsum(
                    "aij,j->ai", dphi_tensor, self.initial_state, optimize=True
                )
            else:
                self.derivatives = np.gradient(self.trajectories, self.times, axis=0)
        else:
            # Caso 2D: ensamble masivo C0 in R^{N x M} -> trajectories y derivatives tienen forma (K, N, M)
            self.trajectories = np.einsum(
                "aij,jm->aim", phi_tensor, self.initial_state, optimize=True
            )
            if isinstance(self.propagator, PutzerPropagator):
                self.derivatives = np.einsum(
                    "ik,akm->aim",
                    self.propagator.a_matrix,
                    self.trajectories,
                    optimize=True,
                )
            elif isinstance(self.propagator, NumericalMatrixPropagator):
                assert self.propagator.matrix_solver.derivatives is not None
                dphi_tensor = self.propagator.matrix_solver.derivatives
                self.derivatives = np.einsum(
                    "aij,jm->aim", dphi_tensor, self.initial_state, optimize=True
                )
            else:
                self.derivatives = np.gradient(self.trajectories, self.times, axis=0)

        # 4. Diagnóstico empírico de variación y constante de Lipschitz heredado de DynamicalTrajectoryBase
        self.analyze_stability()

        if return_basis:
            return self.times, self.trajectories, self.derivatives, phi_tensor
        return self.times, self.trajectories, self.derivatives