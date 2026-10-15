import time

import numpy as np
from numba import njit

from src.ode_solver import RK4
from src.plotter import Plotter


# ==============================================================================
# OPCIÓN A: VERSIÓN VECTORIAL (Espacio de estados como vector 1D de 54 dimensiones)
# ==============================================================================
@njit
def calcular_aceleraciones_vector(t, y):
    num_planetas = 9
    derivadas = np.zeros_like(y)

    G = 4.0 * np.pi * np.pi
    M = 1.0

    for i in range(num_planetas):
        idx = i * 6

        x = y[idx]
        yy = y[idx + 1]
        z = y[idx + 2]
        vx = y[idx + 3]
        vy = y[idx + 4]
        vz = y[idx + 5]

        derivadas[idx] = vx
        derivadas[idx + 1] = vy
        derivadas[idx + 2] = vz

        r = np.sqrt(x * x + yy * yy + z * z)
        k = -(G * M) / (r * r * r)

        derivadas[idx + 3] = k * x
        derivadas[idx + 4] = k * yy
        derivadas[idx + 5] = k * z

    return derivadas


# ==============================================================================
# OPCIÓN B: VERSIÓN MATRICIAL TENSORIAL (Espacio de estados como matriz 6x9)
# ==============================================================================
@njit
def calcular_aceleraciones_matriz(t, Y):
    G = 4.0 * np.pi * np.pi
    M = 1.0

    R = Y[:3, :]  # Matriz de posiciones (3x9)
    V = Y[3:, :]  # Matriz de velocidades (3x9)

    r = np.empty(9, dtype=np.float64)
    for i in range(9):
        r[i] = np.sqrt(R[0, i] ** 2 + R[1, i] ** 2 + R[2, i] ** 2)

    k = -(G * M) / (r**3)

    Derivadas = np.empty_like(Y)
    Derivadas[:3, :] = V  # Derivada de posición = velocidad

    for i in range(9):
        Derivadas[3, i] = k[i] * R[0, i]
        Derivadas[4, i] = k[i] * R[1, i]
        Derivadas[5, i] = k[i] * R[2, i]

    return Derivadas


if __name__ == "__main__":
    print("======================================================================")
    print("   STRESS TEST SISTEMA SOLAR: 250 AÑOS (VECTOR VS MATRIZ)            ")
    print("======================================================================\n")

    CI_solar = [
        # Mercurio
        0.0508011,
        -0.4011642,
        -0.2195576,
        0.02231875 * 365.25,
        0.00485553 * 365.25,
        0.00027924 * 365.25,
        # Venus
        0.0655112,
        -0.6589741,
        -0.3006194,
        0.02000833 * 365.25,
        0.00205477 * 365.25,
        -0.00034178 * 365.25,
        # Tierra
        -0.6349052,
        0.6910548,
        0.2995998,
        -0.01343443 * 365.25,
        -0.01023223 * 365.25,
        -0.00443607 * 365.25,
        # Marte
        -0.8326456,
        -1.1782356,
        -0.5179253,
        0.01227603 * 365.25,
        -0.00570021 * 365.25,
        -0.00294625 * 365.25,
        # Júpiter
        -5.385202,
        -0.840171,
        -0.229015,
        0.001099944 * 365.25,
        -0.006518169 * 365.25,
        -0.002820667 * 365.25,
        # Saturno
        -3.743403,
        7.568504,
        3.287166,
        -0.005379637 * 365.25,
        -0.002232371 * 365.25,
        -0.000690547 * 365.25,
        # Urano
        18.38175,
        -7.25798,
        -3.43875,
        0.001544588 * 365.25,
        0.003135267 * 365.25,
        0.001351424 * 365.25,
        # Neptuno
        21.25332,
        -19.49023,
        -8.50658,
        0.002199232 * 365.25,
        0.002084787 * 365.25,
        0.000798469 * 365.25,
        # Plutón
        -4.11218,
        -29.56880,
        -7.98970,
        0.003169840 * 365.25,
        -0.000584314 * 365.25,
        -0.001138225 * 365.25,
    ]

    dt = 0.005
    anos_a_simular = 250.0  # 50,000 pasos temporales
    nombres_planetas = [
        "Mercurio",
        "Venus",
        "Tierra",
        "Marte",
        "Júpiter",
        "Saturno",
        "Urano",
        "Neptuno",
        "Plutón",
    ]

    # --------------------------------------------------------------------------
    # 1. EJECUCIÓN Y GRÁFICA: MODELO VECTORIAL (54D)
    # --------------------------------------------------------------------------
    print("Ejecutando Modelo Vectorial (54D) para 250 años...")
    motor_vectorial = RK4(
        calcular_aceleraciones_vector, y0=CI_solar, t0=0.0, tf=anos_a_simular, dt=dt
    )
    motor_vectorial.solve()  # Compilación JIT

    inicio_vec = time.perf_counter()
    t_vec, traj_vec, deriv_vec = motor_vectorial.solve()
    fin_vec = time.perf_counter()
    tiempo_vectorial = fin_vec - inicio_vec
    print(f">> Tiempo Vectorial: {tiempo_vectorial:.4f} segundos\n")

    x_vec = np.column_stack([traj_vec[:, 6 * i] for i in range(9)])
    y_vec = np.column_stack([traj_vec[:, 6 * i + 1] for i in range(9)])
    z_vec = np.column_stack([traj_vec[:, 6 * i + 2] for i in range(9)])

    # Creamos un objeto Plotter exclusivo para la gráfica vectorial
    plotter_vectorial = Plotter(figsize=(10, 8))
    print("Abriendo Gráfica 1: Sistema Solar (Modelo Vectorial 54D)...")
    plotter_vectorial.plot_3d(
        x_data=x_vec,
        y_data=y_vec,
        z_data=z_vec,
        title=f"Sistema Solar - Modelo Vectorial (54D) | {tiempo_vectorial:.3f} s",
        xlabel="X (UA)",
        ylabel="Y (UA)",
        zlabel="Z (UA)",
        labels=nombres_planetas,
        equal_box=True,
        view_angle=(30, -50),
        show=True,
    )

    # --------------------------------------------------------------------------
    # 2. EJECUCIÓN Y GRÁFICA: MODELO MATRICIAL (6x9)
    # --------------------------------------------------------------------------
    print("Ejecutando Modelo Matricial (6x9) para 250 años...")
    CI_matriz = np.array(CI_solar, dtype=np.float64).reshape((9, 6)).T
    motor_matricial = RK4(
        calcular_aceleraciones_matriz, y0=CI_matriz, t0=0.0, tf=anos_a_simular, dt=dt
    )
    motor_matricial.solve()  # Compilación JIT

    inicio_mat = time.perf_counter()
    t_mat, traj_mat, deriv_mat = motor_matricial.solve()
    fin_mat = time.perf_counter()
    tiempo_matricial = fin_mat - inicio_mat
    print(f">> Tiempo Matricial: {tiempo_matricial:.4f} segundos\n")

    x_mat = traj_mat[:, 0, :]
    y_mat = traj_mat[:, 1, :]
    z_mat = traj_mat[:, 2, :]

    # Creamos un segundo objeto Plotter independiente para la gráfica matricial
    plotter_matricial = Plotter(figsize=(10, 8))
    print("Abriendo Gráfica 2: Sistema Solar (Modelo Matricial 6x9)...")
    plotter_matricial.plot_3d(
        x_data=x_mat,
        y_data=y_mat,
        z_data=z_mat,
        title=f"Sistema Solar - Modelo Matricial (Tensor 6x9) | {tiempo_matricial:.3f} s",
        xlabel="X (UA)",
        ylabel="Y (UA)",
        zlabel="Z (UA)",
        labels=nombres_planetas,
        equal_box=True,
        view_angle=(30, -50),
        show=True,
    )
