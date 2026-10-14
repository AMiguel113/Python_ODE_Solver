from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D


class Plotter:
    """
    Motor de visualización científica  en 2D y 3D con Matplotlib.
    """

    def __init__(self, style: str = "seaborn-v0_8-whitegrid", figsize: tuple[int, int] = (9, 6)) -> None:
        """
        Inicializa la configuración estética global del graficador.
        """
        self.figsize = figsize
        # Intentamos aplicar un estilo limpio; si no está instalado, configuramos rejilla manual
        try:
            plt.style.use(style)
        except OSError:
            plt.rcParams["axes.grid"] = True
            plt.rcParams["grid.alpha"] = 0.4

    @staticmethod
    def _normalize_series(data: np.ndarray | Sequence[np.ndarray]) -> list[np.ndarray]:
        """
        Normaliza la entrada para que siempre sea una lista de vectores 1D,
        permitiendo que el mismo método grafique 1 o N curvas sin duplicar código.
        
        - Si recibe un vector 1D de forma (steps,): devuelve [data].
        - Si recibe una matriz 2D de forma (steps, M): separa las M columnas en una lista.
        - Si recibe una lista/tupla de vectores 1D: la convierte en lista de np.ndarray.
        """
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                return [data]
            if data.ndim == 2:
                # Cada columna j es una serie independiente de longitud 'steps'
                return [data[:, j] for j in range(data.shape[1])]
            raise ValueError(f"Se esperaba un arreglo 1D o 2D, pero se recibió forma {data.shape}.")
        
        return [np.asarray(arr, dtype=np.float64).ravel() for arr in data]

    def plot_2d(
        self,
        x_data: np.ndarray | Sequence[np.ndarray],
        y_data: np.ndarray | Sequence[np.ndarray],
        title: str = "Gráfica 2D",
        xlabel: str = "Eje X",
        ylabel: str = "Eje Y",
        labels: Sequence[str] | None = None,
        lw: float = 1.8,
        alpha: float = 0.9,
        equal_aspect: bool = False,
        save_path: str | None = None,
        dpi: int = 300,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """
        Método universal 2D: sirve para Series de Tiempo (x=t, y=Y),
        Espacio Fase 2D (x=q, y=p), Múltiples Series o Retratos Fase de Ensambles.

        Parámetros:
            x_data : Vector 1D (ej. 'times' compartido) o lista/matriz de vectores X.
            y_data : Vector 1D, matriz 2D (steps, M) o lista de vectores Y.
            equal_aspect : Si es True, fuerza escala isométrica 1:1 (útil en órbitas/espacio fase).
            save_path : Si se especifica una ruta (ej. 'fase.png'), exporta la figura automáticamente.
        """
        x_list = self._normalize_series(x_data)
        y_list = self._normalize_series(y_data)

        # Si se pasó un solo eje X (como el vector 'times') y múltiples Y, repetimos X para cada Y
        if len(x_list) == 1 and len(y_list) > 1:
            x_list = x_list * len(y_list)

        if len(x_list) != len(y_list):
            raise ValueError(
                f"Incompatibilidad de dimensiones: se recibieron {len(x_list)} series en X "
                f"y {len(y_list)} series en Y."
            )

        fig, ax = plt.subplots(figsize=self.figsize)

        for i, (x_arr, y_arr) in enumerate(zip(x_list, y_list)):
            curve_label = labels[i] if (labels is not None and i < len(labels)) else None
            ax.plot(x_arr, y_arr, lw=lw, alpha=alpha, label=curve_label)

        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)

        if equal_aspect:
            ax.set_aspect("equal", adjustable="datalim")

        if labels is not None:
            ax.legend(frameon=True, loc="best")

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

        if show:
            plt.show()

        return fig, ax

    def plot_3d(
        self,
        x_data: np.ndarray | Sequence[np.ndarray],
        y_data: np.ndarray | Sequence[np.ndarray],
        z_data: np.ndarray | Sequence[np.ndarray],
        title: str = "Espacio Fase 3D",
        xlabel: str = "X",
        ylabel: str = "Y",
        zlabel: str = "Z",
        labels: Sequence[str] | None = None,
        lw: float = 1.5,
        alpha: float = 0.85,
        mark_start: bool = True,
        equal_box: bool = False,
        view_angle: tuple[float, float] = (25.0, -60.0),
        save_path: str | None = None,
        dpi: int = 300,
        show: bool = True,
    ) -> tuple[Figure, Axes3D]:
        """
        Método universal 3D: grafica una o múltiples trayectorias en R^3
        (ej. Atractor de Lorenz, órbitas keplerianas 3D o subespacios del flujo).

        Parámetros:
            equal_box  : Equivalente a 'set view equal xyz' en Gnuplot.
            view_angle : Tupla (elevación, azimut) equivalente a 'set view' en Gnuplot.
            mark_start : Si es True, dibuja un punto sobre la condición inicial c_0 de cada curva.
        """
        x_list = self._normalize_series(x_data)
        y_list = self._normalize_series(y_data)
        z_list = self._normalize_series(z_data)

        if not (len(x_list) == len(y_list) == len(z_list)):
            raise ValueError(
                f"Las listas X ({len(x_list)}), Y ({len(y_list)}) y Z ({len(z_list)}) "
                "deben tener el mismo número de trayectorias."
            )

        fig = plt.figure(figsize=self.figsize)
        ax: Axes3D = fig.add_subplot(111, projection="3d")

        for i, (x_arr, y_arr, z_arr) in enumerate(zip(x_list, y_list, z_list)):
            curve_label = labels[i] if (labels is not None and i < len(labels)) else None
            line = ax.plot(x_arr, y_arr, z_arr, lw=lw, alpha=alpha, label=curve_label)[0]

            if mark_start and x_arr.size > 0:
                ax.scatter(
                    x_arr[0], y_arr[0], z_arr[0],
                    color=line.get_color(), s=25, marker="o"
                )

        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_zlabel(zlabel, fontsize=10)
        ax.view_init(elev=view_angle[0], azim=view_angle[1])

        if equal_box:
            # Calcula los rangos reales para mantener una caja 3D isométrica sin deformar órbitas
            all_x = np.concatenate(x_list)
            all_y = np.concatenate(y_list)
            all_z = np.concatenate(z_list)
            max_range = np.array([
                np.ptp(all_x), np.ptp(all_y), np.ptp(all_z)
            ]).max() / 2.0

            mid_x = (all_x.max() + all_x.min()) * 0.5
            mid_y = (all_y.max() + all_y.min()) * 0.5
            mid_z = (all_z.max() + all_z.min()) * 0.5

            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            ax.set_box_aspect([1.0, 1.0, 1.0])

        if labels is not None:
            ax.legend(frameon=True, loc="best")

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

        if show:
            plt.show()

        return fig, ax