"""
Kernels numéricos y algebraicos compilados en Numba (JIT) para la construcción
exacta de la Matriz Fundamental Principal Phi(t, t0) = exp(A * (t - t0))
mediante el Teorema de Cayley-Hamilton y el Algoritmo Matricial de Putzer.

- Teorema de Cayley-Hamilton y colapso dimensional de C[A].
- Base Polinomial de Newton-Horner {P_0, ..., P_{N-1}}.
- Ordenamiento de Leja y (Evaluación exacta libre de divisiones de r(t) vía recurrencia bidiagonal escalada).
"""

import numba as nb
import numpy as np


@nb.njit(cache=True)
def leja_order_eigenvalues(eigs: np.ndarray) -> np.ndarray:
    """
    Ordena los N valores propios {lambda_1, ..., lambda_N} siguiendo una
    secuencia de puntos de Leja en el plano complejo C.

    Matemática:
        1. Elige como primer nodo lambda_{(0)} el autovalor de mayor módulo:
        |lambda_{(0)}| = max_j |lambda_j|.
        2. Para cada paso k = 1, ..., N-1, elige entre los autovalores restantes
        aquel que maximiza la productoria de distancias a los ya elegidos:
        lambda_{(k)} = argmax_{lambda in restantes} prod_{m=0}^{k-1} |lambda - lambd{(m)}|.

    Propósito numérico:
        Equilibra las escalas de los factores (A - lambda_k I_N) al construir el
        producto acumulativo P_k, evitando el crecimiento exponencial prematuro
        (overflow) o la anulación prematura (underflow) de la norma ||P_k||_F.
    """
    n_eigs = eigs.shape[0]
    ordered = np.empty(n_eigs, dtype=np.complex128)
    used = np.zeros(n_eigs, dtype=np.bool_)

    # 1. Primer punto de Leja: el valor propio de mayor módulo |lambda_j|
    idx0 = 0
    max_mod = -1.0
    for j in range(n_eigs):
        mod_j = np.abs(eigs[j])
        if mod_j > max_mod:
            max_mod = mod_j
            idx0 = j

    ordered[0] = eigs[idx0]
    used[idx0] = True

    # 2. Selección greedy de los siguientes N - 1 puntos de Leja
    for k in range(1, n_eigs):
        best_idx = -1
        best_prod = -1.0
        for j in range(n_eigs):
            if not used[j]:
                # Productoria de distancias |eigs[j] - ordered[m]| para m = 0, ..., k-1
                prod_dist = 1.0
                for m in range(k):
                    prod_dist *= np.abs(eigs[j] - ordered[m])
                if prod_dist > best_prod:
                    best_prod = prod_dist
                    best_idx = j

        ordered[k] = eigs[best_idx]
        used[best_idx] = True

    return ordered


@nb.njit(cache=True, fastmath=True)
def eval_putzer_vector_single_t(lambdas: np.ndarray, dt_val: float) -> np.ndarray:
    """
    Evalúa de forma exacta el vector de funciones escalares de Putzer
    r(dt_val) = [r_1(dt_val), r_2(dt_val), ..., r_N(dt_val)]^T in C^N
    para un desplazamiento temporal dt_val = t - t0 .

    Matemática:
        El sistema escalar en cascada de Putzer :
            dr_1/dt = lambda_1 * r_1,             r_1(0) = 1
            dr_k/dt = lambda_k * r_k + r_{k-1},   r_k(0) = 0  (k = 2, ..., N)
        equivale a la EDO matricial dr/dt = J_Putzer * r con r(0) = e_{(1)},
        donde J_Putzer es la matriz bidiagonal inferior con
        autovalores en la diagonal y unos en la primera subdiagonal.
        Por tanto, r(dt_val) es la primera columna de exp(dt_val * J_Putzer).
    """
    n_dim = lambdas.shape[0]

    # Caso trivial en el origen temporal t == t0 (dt_val == 0): r(0) = e_{(1)}
    if dt_val == 0.0:
        r0 = np.zeros(n_dim, dtype=np.complex128)
        r0[0] = 1.0 + 0.0j
        return r0

    # Paso 1: Determinar el factor de escalamiento 2^s tal que
    # |tau| * max(1, max_j |lambda_j|) <= 0.25, donde tau = dt_val / 2^s
    max_lam = 0.0
    for i in range(n_dim):
        abs_lam = np.abs(lambdas[i])
        if abs_lam > max_lam:
            max_lam = abs_lam

    scale_metric = np.abs(dt_val) * max(max_lam, 1.0)
    s_scale = 0
    while scale_metric > 0.25:
        scale_metric *= 0.5
        s_scale += 1

    tau = dt_val / (2.0**s_scale)

    # Paso 2: Evaluar la serie de Taylor Psi_bi(tau) = sum_{m=0}^{q} (tau^m / m!) J_Putzer^m
    # Con |tau * lambda_max| <= 0.25 y q = 18, el resto de truncamiento es < 1e-26 (<< eps_mach)
    t_mat = np.zeros((n_dim, n_dim), dtype=np.complex128)
    term = np.zeros((n_dim, n_dim), dtype=np.complex128)
    next_term = np.zeros((n_dim, n_dim), dtype=np.complex128)

    # Inicialización m = 0: J_Putzer^0 = I_N
    for i in range(n_dim):
        t_mat[i, i] = 1.0 + 0.0j
        term[i, i] = 1.0 + 0.0j

    q_order = 18
    for m in range(1, q_order + 1):
        inv_m = tau / float(m)

        # Producto matricial esparcido por la izquierda con J_Putzer:
        # Como (J_Putzer)_{i, i} = lambda_i y (J_Putzer)_{i, i-1} = 1,
        # la entrada (i, j) del nuevo término solo combina dos entradas del término previo:
        # NextTerm[i, j] = (tau / m) * (lambda_i * Term[i, j] + Term[i-1, j] si i > j)
        for i in range(n_dim):
            lam_i = lambdas[i]
            for j in range(i + 1):
                val = lam_i * term[i, j]
                if i > j:
                    val += term[i - 1, j]
                next_term[i, j] = val * inv_m

        # Acumulación en la suma parcial de Taylor T_mat (solo mitad triangular inferior j <= i)
        for i in range(n_dim):
            for j in range(i + 1):
                term[i, j] = next_term[i, j]
                t_mat[i, j] += term[i, j]

    # Paso 3: Fase de duplicación (Squaring) mediante la Ley de Grupo Uniparamétrico:
    # Psi_bi(2^k * tau) = [Psi_bi(2^{k-1} * tau)]^2 repetido s_scale veces.
    # Como T_mat es triangular inferior, el índice interno k solo corre en j <= k <= i.
    temp_prod = np.zeros((n_dim, n_dim), dtype=np.complex128)
    for _ in range(s_scale):
        for i in range(n_dim):
            for j in range(i + 1):
                acc = 0.0 + 0.0j
                for k in range(j, i + 1):
                    acc += t_mat[i, k] * t_mat[k, j]
                temp_prod[i, j] = acc

        for i in range(n_dim):
            for j in range(i + 1):
                t_mat[i, j] = temp_prod[i, j]

    # Paso 4: Proyección sobre la condición inicial e_{(1)} = [1, 0, ..., 0]^T
    # Equivale a extraer la primera columna (j = 0) de exp(dt_val * J_Putzer)
    r_vec = np.empty(n_dim, dtype=np.complex128)
    for i in range(n_dim):
        r_vec[i] = t_mat[i, 0]

    return r_vec


@nb.njit(parallel=True, cache=True, fastmath=True)
def compute_putzer_coefficients_grid(
    lambdas: np.ndarray, t_grid: np.ndarray, t0: float = 0.0
) -> np.ndarray:
    """
    Evalúa en paralelo sobre todos los hilos de CPU (vía numba.prange) la matriz
    de coeficientes dinámicos de Putzer R in C^{K x N} :
        R[alpha, k] = r_{k+1}(t_alpha - t0),   alpha in {0, ..., K-1}, k in {0, ..., N-1}.

    Por qué es paralelizable en O(1) temporal:
        A diferencia de RK4 o Euler (donde el paso alpha depende secuencialmente
        del paso alpha - 1), la solución algebraica en el grupo de Lie permite
        evaluar cada instante t_alpha de forma completamente independiente.
    """
    k_steps = t_grid.shape[0]
    n_dim = lambdas.shape[0]
    r_matrix = np.empty((k_steps, n_dim), dtype=np.complex128)

    for alpha in nb.prange(k_steps):
        dt_alpha = t_grid[alpha] - t0
        r_matrix[alpha, :] = eval_putzer_vector_single_t(lambdas, dt_alpha)

    return r_matrix


def build_putzer_polynomial_basis(
    a_matrix: np.ndarray, lambdas: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    Construye el Tensor de la Base Polinomial de Operadores de Newton-Horner
    P in C^{N x N x N}  y calcula
    el residuo relativo del Teorema de Cayley-Hamilton .

    Matemática:
        P_0 = I_N
        P_k = (A - lambda_k I_N) P_{k-1},   para k = 1, ..., N-1.
        P_N = (A - lambda_N I_N) P_{N-1} = p_A(A) == 0_{N x N} (por Cayley-Hamilton).

    Retorna:
        tuple[np.ndarray, float]:
            - p_tensor    : Tensor 3D de forma (N, N, N) donde p_tensor[k, :, :] = P_k.
            - ch_residual : Residuo normalizado ||P_N||_F / max(||A||_F, 1)^N.
    """
    n_dim = a_matrix.shape[0]
    p_tensor = np.zeros((n_dim, n_dim, n_dim), dtype=np.complex128)
    identity_n = np.eye(n_dim, dtype=np.complex128)
    a_complex = np.asarray(a_matrix, dtype=np.complex128)

    # P_0 = I_N
    p_tensor[0] = identity_n

    # Recurrencia de escalera de Putzer: P_k = (A - lambda_k I_N) @ P_{k-1}
    for k in range(1, n_dim):
        p_tensor[k] = (a_complex - lambdas[k - 1] * identity_n) @ p_tensor[k - 1]

    # Evaluación terminal en k = N: Anulación por el Teorema de Cayley-Hamilton
    p_terminal = (a_complex - lambdas[n_dim - 1] * identity_n) @ p_tensor[n_dim - 1]
    norm_a = max(float(np.linalg.norm(a_complex, ord="fro")), 1.0)
    ch_residual = float(np.linalg.norm(p_terminal, ord="fro") / (norm_a**n_dim))

    return p_tensor, ch_residual