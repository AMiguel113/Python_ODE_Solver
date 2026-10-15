import time

import numpy as np
from numba import njit

from src.ode_solver import RK4
from src.plotter import Plotter


# ==============================================================================
# 1. MODELO VECTORIAL: 3 Cuerpos (18 dimensiones)
# ==============================================================================
@njit
def calcular_aceleraciones_3c_vector(t, y):
    num_cuerpos = 3
    derivadas = np.zeros_like(y)
    G = 1.0
    m = 1.0

    # Extraemos posiciones y velocidades del vector plano de 18 elementos
    R = np.empty((3, 3), dtype=np.float64)
    V = np.empty((3, 3), dtype=np.float64)
    for i in range(num_cuerpos):
        idx = i * 6
        R[0, i] = y[idx]
        R[1, i] = y[idx + 1]
        R[2, i] = y[idx + 2]
        V[0, i] = y[idx + 3]
        V[1, i] = y[idx + 4]
        V[2, i] = y[idx + 5]

    Aceleraciones = np.zeros((3, 3), dtype=np.float64)
    for i in range(num_cuerpos):
        for j in range(num_cuerpos):
            if i != j:
                dx = R[0, j] - R[0, i]
                dy = R[1, j] - R[1, i]
                dz = R[2, j] - R[2, i]
                dist3 = (dx * dx + dy * dy + dz * dz) ** 1.5
                Aceleraciones[0, i] += G * m * dx / dist3
                Aceleraciones[1, i] += G * m * dy / dist3
                Aceleraciones[2, i] += G * m * dz / dist3

    for i in range(num_cuerpos):
        idx = i * 6
        derivadas[idx] = V[0, i]
        derivadas[idx + 1] = V[1, i]
        derivadas[idx + 2] = V[2, i]
        derivadas[idx + 3] = Aceleraciones[0, i]
        derivadas[idx + 4] = Aceleraciones[1, i]
        derivadas[idx + 5] = Aceleraciones[2, i]

    return derivadas


# ==============================================================================
# 2. MODELO MATRICIAL TENSORIAL: 3 Cuerpos (Matriz 6x3)
# ==============================================================================
@njit
def calcular_aceleraciones_3c_matriz(t, Y):
    G = 1.0
    m = 1.0
    R = Y[:3, :]  # Posiciones (3x3)
    V = Y[3:, :]  # Velocidades (3x3)

    Derivadas = np.empty_like(Y)
    Derivadas[:3, :] = V

    Aceleraciones = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            if i != j:
                dx = R[0, j] - R[0, i]
                dy = R[1, j] - R[1, i]
                dz = R[2, j] - R[2, i]
                dist3 = (dx * dx + dy * dy + dz * dz) ** 1.5
                Aceleraciones[0, i] += G * m * dx / dist3
                Aceleraciones[1, i] += G * m * dy / dist3
                Aceleraciones[2, i] += G * m * dz / dist3

    Derivadas[3:, :] = Aceleraciones
    return Derivadas


# ==============================================================================
# 3. CASO LÍMITE MASIVO: Anillo de N Cuerpos (Matriz 6xN vs Vector 6N)
# ==============================================================================
@njit
def calcular_aceleraciones_anillo_matriz(t, Y):
    """
    Simula un sistema masivo de N cuerpos interactuando gravitacionalmente
    usando la estructura matricial de (6, N).
    """
    G = 1.0
    N = Y.shape[1]
    R = Y[:3, :]
    V = Y[3:, :]

    Derivadas = np.empty_like(Y)
    Derivadas[:3, :] = V

    Aceleraciones = np.zeros((3, N), dtype=np.float64)
    mass = 1.0 / N  # Masa distribuida

    for i in range(N):
        for j in range(N):
            if i != j:
                dx = R[0, j] - R[0, i]
                dy = R[1, j] - R[1, i]
                dz = R[2, j] - R[2, i]
                dist3 = (
                    dx * dx + dy * dy + dz * dz + 1e-4
                ) ** 1.5  # Softening para estabilidad numérica
                Aceleraciones[0, i] += G * mass * dx / dist3
                Aceleraciones[1, i] += G * mass * dy / dist3
                Aceleraciones[2, i] += G * mass * dz / dist3

    Derivadas[3:, :] = Aceleraciones
    return Derivadas


if __name__ == "__main__":
    print("======================================================================")
    print("   BENCHMARK: 3 CUERPOS (VECTOR VS MATRIZ) Y CASO LÍMITE MASIVO       ")
    print("======================================================================\n")

    # Condiciones iniciales de la órbita en "Ocho"
    x1, y1 = -0.97000436, 0.24308753
    x2, y2 = -x1, -y1
    x3, y3 = 0.0, 0.0
    vx3, vy3 = 0.93240737, 0.86473146
    vx1, vy1 = -vx3 / 2.0, -vy3 / 2.0
    vx2, vy2 = vx1, vy1

    CI_3c_flat = np.array(
        [
            x1,
            y1,
            0.0,
            vx1,
            vy1,
            0.0,
            x2,
            y2,
            0.0,
            vx2,
            vy2,
            0.0,
            x3,
            y3,
            0.0,
            vx3,
            vy3,
            0.0,
        ],
        dtype=np.float64,
    )

    dt = 0.002
    tf = 20.0  # Múltiples periodos del ocho

    # --------------------------------------------------------------------------
    # 1. 3 CUERPOS - MODO VECTORIAL (18D)
    # --------------------------------------------------------------------------
    print("Ejecutando 3 Cuerpos - Modo Vectorial (18D)...")
    motor_vec = RK4(
        calcular_aceleraciones_3c_vector, y0=CI_3c_flat, t0=0.0, tf=tf, dt=dt
    )
    motor_vec.solve()  # JIT

    t0 = time.perf_counter()
    _, traj_vec, _ = motor_vec.solve()
    t1 = time.perf_counter()
    print(f">> Tiempo Vectorial (3 Cuerpos): {t1 - t0:.4f} s\n")

    x_v = np.column_stack([traj_vec[:, 6 * i] for i in range(3)])
    y_v = np.column_stack([traj_vec[:, 6 * i + 1] for i in range(3)])
    z_v = np.column_stack([traj_vec[:, 6 * i + 2] for i in range(3)])

    plotter_v = Plotter(figsize=(9, 8))
    plotter_v.plot_3d(
        x_data=x_v,
        y_data=y_v,
        z_data=z_v,
        title=f"3 Cuerpos en Ocho - Vectorial (18D) | {t1 - t0:.3f} s",
        xlabel="X",
        ylabel="Y",
        zlabel="Z",
        labels=["Cuerpo 1", "Cuerpo 2", "Cuerpo 3"],
        equal_box=True,
        view_angle=(90, -90),
        show=True,
    )

    # --------------------------------------------------------------------------
    # 2. 3 CUERPOS - MODO MATRICIAL (6x3)
    # --------------------------------------------------------------------------
    print("Ejecutando 3 Cuerpos - Modo Matricial (6x3)...")
    CI_3c_mat = CI_3c_flat.reshape((3, 6)).T  # Forma (6, 3)
    motor_mat = RK4(
        calcular_aceleraciones_3c_matriz, y0=CI_3c_mat, t0=0.0, tf=tf, dt=dt
    )
    motor_mat.solve()  # JIT

    t0 = time.perf_counter()
    _, traj_mat, _ = motor_mat.solve()
    t1 = time.perf_counter()
    print(f">> Tiempo Matricial (6x3): {t1 - t0:.4f} s\n")

    x_m = traj_mat[:, 0, :]
    y_m = traj_mat[:, 1, :]
    z_m = traj_mat[:, 2, :]

    plotter_m = Plotter(figsize=(9, 8))
    plotter_m.plot_3d(
        x_data=x_m,
        y_data=y_m,
        z_data=z_m,
        title=f"3 Cuerpos en Ocho - Matricial (6x3) | {t1 - t0:.3f} s",
        xlabel="X",
        ylabel="Y",
        zlabel="Z",
        labels=["Cuerpo 1", "Cuerpo 2", "Cuerpo 3"],
        equal_box=True,
        view_angle=(90, -90),
        show=True,
    )

    # --------------------------------------------------------------------------
    # 3. CASO LÍMITE MASIVO: ANILLO DE N = 100 CUERPOS (Matriz 6 x 100)
    # --------------------------------------------------------------------------
    N_cuerpos = 100
    print(
        f"--- CASO LÍMITE MASIVO: Simulación de N = {N_cuerpos} cuerpos (600 EDOs) ---"
    )

    # Generamos un anillo circular de 100 cuerpos con velocidades tangenciales
    theta = np.linspace(0, 2 * np.pi, N_cuerpos, endpoint=False)
    radio = 5.0
    R_init = np.array(
        [radio * np.cos(theta), radio * np.sin(theta), np.zeros(N_cuerpos)]
    )
    # Velocidad orbital aproximada para equilibrio circular central
    v_circ = np.sqrt(1.0 / radio)
    V_init = np.array(
        [-v_circ * np.sin(theta), v_circ * np.cos(theta), np.zeros(N_cuerpos)]
    )

    CI_masivo = np.vstack((R_init, V_init))  # Forma (6, 100)

    motor_masivo = RK4(
        calcular_aceleraciones_anillo_matriz, y0=CI_masivo, t0=0.0, tf=5.0, dt=0.005
    )
    motor_masivo.solve()  # JIT

    t0 = time.perf_counter()
    _, traj_masivo, _ = motor_masivo.solve()
    t1 = time.perf_counter()
    print(
        f">> Tiempo para N = {N_cuerpos} cuerpos ({N_cuerpos * 6} EDOs): {t1 - t0:.4f} segundos\n"
    )

    # Extraemos las coordenadas del tensor (steps, 6, 100)
    x_mass = traj_masivo[:, 0, :]
    y_mass = traj_masivo[:, 1, :]
    z_mass = traj_masivo[:, 2, :]

    plotter_mass = Plotter(figsize=(10, 8))
    print(f"Abriendo Gráfica del Caso Límite Masivo (N = {N_cuerpos})...")
    plotter_mass.plot_3d(
        x_data=x_mass,
        y_data=y_mass,
        z_data=z_mass,
        title=f"Caso Límite: Anillo de N = {N_cuerpos} Cuerpos ({N_cuerpos * 6} EDOs) - RK4",
        xlabel="X",
        ylabel="Y",
        zlabel="Z",
        equal_box=True,
        view_angle=(45, -60),
        show=True,
    )
