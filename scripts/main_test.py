import numpy as np
from numba import njit

from src.ode_solver import RK4
from src.plotter import Plotter


# ==============================================================================
# PRUEBA 2: Sistema No Lineal de Lorenz en 3D
# ==============================================================================
@njit
def lorenz_system(t, y):
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    out = np.empty_like(y)
    out[0] = sigma * (y[1] - y[0])
    out[1] = y[0] * (rho - y[2]) - y[1]
    out[2] = y[0] * y[1] - beta * y[2]
    return out

@njit
def aizawa_system(t, y):
    """
    Sistema dinámico no lineal del Atractor de Aizawa en R^3.
    y[0] = x, y[1] = y, y[2] = z
    """
    x, yy, z = y[0], y[1], y[2]

    # Parámetros estándar de Aizawa
    a = 0.95
    b = 0.7
    c = 0.6
    d = 3.5
    e = 0.25
    f = 0.1

    dx = (z - b) * x - d * yy
    dy = d * x + (z - b) * yy
    dz = c + a * z - (z**3) / 3.0 - (x**2 + yy**2) * (1.0 + e * z) + f * z * (x**3)

    out = np.empty_like(y)
    out[0] = dx
    out[1] = dy
    out[2] = dz
    return out

@njit
def roessler_system(t, y):
    """
    dx/dt = -y - z
    dy/dt = x + a*y
    dz/dt = b + z*(x - c)
    """
    x, yy, z = y[0], y[1], y[2]
    
    # Parámetros clásicos de Rössler
    a = 0.2
    b = 0.2
    c = 5.7

    dx = -yy - z
    dx_dt = dx
    dy_dt = x + a * yy
    dz_dt = b + z * (x - c)

    out = np.empty_like(y)
    out[0] = dx_dt
    out[1] = dy_dt
    out[2] = dz_dt
    return out


if __name__ == "__main__":
    plotter = Plotter()

    # --- 2. Ejecutamos RK4 sobre el Atractor de Lorenz 3D ---
    sol_lorenz = RK4(lorenz_system, [1.0, 1.0, 1.0], 0.0, 50.0, 0.01)
    t_lor, tr_lor, deriv_lor = sol_lorenz.solve()
    print(sol_lorenz)
    # C) Espacio Fase 3D con plot_3d
    plotter.plot_3d(
        x_data=tr_lor[:, 0],
        y_data=tr_lor[:, 1],
        z_data=tr_lor[:, 2],
        title="Atractor de Lorenz (RK4)",
        xlabel="x",
        ylabel="y",
        zlabel="z",
        labels=["Trayectoria RK4"],
    )
    
    y0_aizawa = [0.1, 0.0, 0.0]
    t0, tf, dt = 0.0, 50.0, 0.01

    print("Calculando trayectorias del Atractor de Aizawa con RK4...")
    solver = RK4(aizawa_system, y0=y0_aizawa, t0=t0, tf=tf, dt=dt)
    times, trajectories, derivatives = solver.solve()

    # Imprimimos reporte de estabilidad y Lipschitz empírico
    print(solver)
    solver.write_report("reporte_estabilidad.txt")

    # Guardamos los datos en texto para respaldo
    solver.save_trajectory_txt("data/aizawa.txt")
    print("Generando visualización 3D del atractor...")
    plotter.plot_3d(
        x_data=trajectories[:, 0],
        y_data=trajectories[:, 1],
        z_data=trajectories[:, 2],
        title="Atractor de Aizawa - Espacio Fase 3D (RK4)",
        xlabel="Eje X",
        ylabel="Eje Y",
        zlabel="Eje Z",
        labels=["Órbita Aizawa"],
        equal_box=True,      # Mantiene la caja isométrica (equivalente a set view equal xyz de Gnuplot)
        view_angle=(30, -45),# Ángulo de elevación y azimut inicial de la cámara
        save_path="figs/aizawa_3d.png", # Guarda automáticamente en figs/
        show=True
    )
    y0 = [0.1, 0.1, 0.1]
    t0, tf, dt = 0.0, 100.0, 0.005  # dt fino para alta resolución en la banda caótica

    print("Integrando el Atractor de Rössler con tu motor RK4...")
    solver = RK4(roessler_system, y0=y0, t0=t0, tf=tf, dt=dt)
    times, trajectories, derivatives = solver.solve()

    print(solver)
    solver.write_report("reporte_estabilidad.txt")

    # 3. Graficamos usando TU clase Plotter y aplicamos un toque de colorimetría caótica
    plotter = Plotter(figsize=(10, 8))

    # Extraemos las componentes X, Y, Z del tensor de trayectorias
    x = trajectories[:, 0]
    y = trajectories[:, 1]
    z = trajectories[:, 2]

    # Truco visualizador profesional manteniendo tu Plotter: 
    # Usamos directamente Matplotlib sobre la figura base para mapear el eje Z al color (profundidad del atractor)
    fig, ax = plotter.plot_3d(
        x_data=x, y_data=y, z_data=z,
        title="Atractor de Rössler - Integrado con tu motor RK4",
        xlabel="X", ylabel="Y", zlabel="Z",
        equal_box=True,
        view_angle=(25, -70),
        show=False # No la muestra de inmediato para maquillar el color
    )

    # Añadimos un mapa de colores basado en la altura Z (z-depth coloring)
    # Borramos la línea monocromática por defecto y pintamos un gradiente continuo
    ax.clear()
    
    # Creamos segmentos de línea coloreados por la coordenada Z
    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    lc = Line3DCollection(segments, cmap="plasma", linewidth=0.8, alpha=0.85)
    lc.set_array(z) # El color cambia dinámicamente según la profundidad Z
    ax.add_collection3d(lc)

    # Reconfiguramos los límites y etiquetas de tu plotter
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_zlim(z.min(), z.max())
    ax.set_title("Atractor de Rössler (Z-Depth Plasma Colormap)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=25, azim=-70)
    
    print("Mostrando visualización interactiva...")
    import matplotlib.pyplot as plt
    plt.show()