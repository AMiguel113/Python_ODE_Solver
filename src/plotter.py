from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D


class Plotter:
    """
    Motor de visualización científica en 2D y 3D con Matplotlib.
    Soporta series temporales 1D/2D, tensores de ensamble 3D (K, N, M)
    y diagnósticos geométricos de sistemas lineales (Wronskiano / Abel-Liouville).
    """

    def __init__(
        self, style: str = "seaborn-v0_8-whitegrid", figsize: tuple[int, int] = (9, 6)
    ) -> None:
        """
        Inicializa la configuración estética global del graficador.
        """
        self.figsize = figsize
        try:
            plt.style.use(style)
        except OSError:
            plt.rcParams["axes.grid"] = True
            plt.rcParams["grid.alpha"] = 0.4

    @staticmethod
    def _normalize_series(data: np.ndarray | Sequence[np.ndarray]) -> list[np.ndarray]:
        """
        Normaliza la entrada para que siempre sea una lista de vectores 1D,
        permitiendo que el mismo método grafique 1 o M curvas sin duplicar código.

        - Si recibe un vector 1D de forma (K,): devuelve [data].
        - Si recibe una matriz 2D de forma (K, M): separa las M columnas en una lista.
        - Si recibe una lista/tupla de vectores 1D: la convierte en lista de np.ndarray.
        """
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                return [data]
            if data.ndim == 2:
                return [data[:, j] for j in range(data.shape[1])]
            raise ValueError(
                f"Se esperaba un arreglo 1D o 2D, pero se recibió forma {data.shape}. "
                f"Para tensores 3D de ensamble (K, N, M), usa plot_phase_space o extract_coordinate."
            )

        return [np.asarray(arr, dtype=np.float64).ravel() for arr in data]

    @staticmethod
    def extract_coordinate(trajectories: np.ndarray, coord_index: int) -> np.ndarray:
        """
        Proyecta el tensor de trayectorias sobre la coordenada canónica i = coord_index.

        - Si trajectories tiene forma (K, N) [una sola fibra]: devuelve vector 1D (K,).
        - Si trajectories tiene forma (K, N, M) [ensamble de M fibras]: devuelve matriz 2D (K, M).
        """
        arr = np.asarray(trajectories, dtype=np.float64)
        if arr.ndim == 2:
            return arr[:, coord_index]
        if arr.ndim == 3:
            return arr[:, coord_index, :]
        raise ValueError(
            f"El tensor de trayectorias debe tener dimensión 2 (K, N) o 3 (K, N, M), "
            f"pero se recibió forma {arr.shape}."
        )

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
        mark_start: bool = False,
        save_path: str | None = None,
        dpi: int = 300,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """
        Método universal 2D: sirve para Series de Tiempo (x=t, y=Y),
        Espacio Fase 2D (x=q, y=p), Múltiples Series o Retratos Fase de Ensambles.
        """
        x_list = self._normalize_series(x_data)
        y_list = self._normalize_series(y_data)

        if len(x_list) == 1 and len(y_list) > 1:
            x_list = x_list * len(y_list)

        if len(x_list) != len(y_list):
            raise ValueError(
                f"Incompatibilidad de dimensiones: se recibieron {len(x_list)} series en X "
                f"y {len(y_list)} series en Y."
            )

        fig, ax = plt.subplots(figsize=self.figsize)

        for i, (x_arr, y_arr) in enumerate(zip(x_list, y_list)):
            curve_label = (
                labels[i] if (labels is not None and i < len(labels)) else None
            )
            line = ax.plot(x_arr, y_arr, lw=lw, alpha=alpha, label=curve_label)[0]
            if mark_start and x_arr.size > 0:
                ax.scatter(
                    x_arr[0],
                    y_arr[0],
                    color=line.get_color(),
                    s=22,
                    marker="o",
                    zorder=5,
                )

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
        Método universal 3D: grafica una o múltiples trayectorias en R^3.
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
            curve_label = (
                labels[i] if (labels is not None and i < len(labels)) else None
            )
            line = ax.plot(x_arr, y_arr, z_arr, lw=lw, alpha=alpha, label=curve_label)[
                0
            ]

            if mark_start and x_arr.size > 0:
                ax.scatter(
                    x_arr[0],
                    y_arr[0],
                    z_arr[0],
                    color=line.get_color(),
                    s=25,
                    marker="o",
                )

        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_zlabel(zlabel, fontsize=10)
        ax.view_init(elev=view_angle[0], azim=view_angle[1])

        if equal_box:
            all_x = np.concatenate(x_list)
            all_y = np.concatenate(y_list)
            all_z = np.concatenate(z_list)
            max_range = (
                np.array([np.ptp(all_x), np.ptp(all_y), np.ptp(all_z)]).max() / 2.0
            )

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

    def plot_phase_space(
        self,
        trajectories: np.ndarray,
        coords: tuple[int, int] | tuple[int, int, int] = (0, 1),
        title: str = "Retrato Fase del Ensamble",
        axis_labels: Sequence[str] | None = None,
        labels: Sequence[str] | None = None,
        lw: float = 1.5,
        alpha: float = 0.85,
        mark_start: bool = True,
        equal_aspect: bool = False,
        view_angle: tuple[float, float] = (25.0, -60.0),
        save_path: str | None = None,
        dpi: int = 300,
        show: bool = True,
    ) -> tuple[Figure, Axes | Axes3D]:
        """
        Grafica directamente un tensor de trayectorias 2D (K, N) o 3D (K, N, M)
        en el subespacio de coordenadas indicado por 'coords' (2D o 3D),
        reutilizando plot_2d y plot_3d sin duplicar código.
        """
        if len(coords) == 2:
            i, j = coords
            x_proj = self.extract_coordinate(trajectories, i)
            y_proj = self.extract_coordinate(trajectories, j)
            xl = (
                axis_labels[0]
                if (axis_labels and len(axis_labels) >= 2)
                else f"x_{i + 1}"
            )
            yl = (
                axis_labels[1]
                if (axis_labels and len(axis_labels) >= 2)
                else f"x_{j + 1}"
            )
            return self.plot_2d(
                x_data=x_proj,
                y_data=y_proj,
                title=title,
                xlabel=xl,
                ylabel=yl,
                labels=labels,
                lw=lw,
                alpha=alpha,
                equal_aspect=equal_aspect,
                mark_start=mark_start,
                save_path=save_path,
                dpi=dpi,
                show=show,
            )

        if len(coords) == 3:
            i, j, k = coords
            x_proj = self.extract_coordinate(trajectories, i)
            y_proj = self.extract_coordinate(trajectories, j)
            z_proj = self.extract_coordinate(trajectories, k)
            xl = (
                axis_labels[0]
                if (axis_labels and len(axis_labels) >= 3)
                else f"x_{i + 1}"
            )
            yl = (
                axis_labels[1]
                if (axis_labels and len(axis_labels) >= 3)
                else f"x_{j + 1}"
            )
            zl = (
                axis_labels[2]
                if (axis_labels and len(axis_labels) >= 3)
                else f"x_{k + 1}"
            )
            return self.plot_3d(
                x_data=x_proj,
                y_data=y_proj,
                z_data=z_proj,
                title=title,
                xlabel=xl,
                ylabel=yl,
                zlabel=zl,
                labels=labels,
                lw=lw,
                alpha=alpha,
                mark_start=mark_start,
                equal_box=equal_aspect,
                view_angle=view_angle,
                save_path=save_path,
                dpi=dpi,
                show=show,
            )

        raise ValueError(
            f"El parámetro 'coords' debe ser una tupla de 2 o 3 índices, pero se recibió {coords}."
        )

    def plot_wronskian(
        self,
        times: np.ndarray,
        phi_tensor: np.ndarray,
        trace_a: float | None = None,
        t0: float | None = None,
        title: str = "Evolución del Wronskiano det(Φ(t)) vs. Abel-Liouville-Jacobi",
        save_path: str | None = None,
        dpi: int = 300,
        show: bool = True,
    ) -> tuple[Figure, Axes]:
        """
        Grafica la evolución del determinante Wronskiano W(t) = det(Phi(t))
        a partir del tensor de la Matriz Fundamental Principal Phi de forma (K, N, N).
        Si se provee trace_a = Tr(A), superpone la curva analítica exacta
        W(t0) * exp(Tr(A) * (t - t0)) del Teorema 3.3 (Abel-Liouville-Jacobi).
        """
        phi_arr = np.asarray(phi_tensor, dtype=np.float64)
        if phi_arr.ndim != 3 or phi_arr.shape[1] != phi_arr.shape[2]:
            raise ValueError(
                f"Se esperaba un tensor de propagador Phi de forma (K, N, N), "
                f"pero se recibió {phi_arr.shape}."
            )

        t_arr = np.asarray(times, dtype=np.float64)
        w_num = np.linalg.det(phi_arr)

        if trace_a is not None:
            t_ref = float(t_arr[0]) if t0 is None else float(t0)
            w0 = float(w_num[0]) if t0 is None else 1.0
            w_exact = w0 * np.exp(float(trace_a) * (t_arr - t_ref))
            series_y = np.column_stack((w_num, w_exact))
            labels = [
                "W(t) = det(Φ(t)) [Numérico]",
                "W(t₀) exp(Tr(A)(t - t₀)) [Abel-Liouville]",
            ]
        else:
            series_y = w_num
            labels = ["W(t) = det(Φ(t))"]

        return self.plot_2d(
            x_data=t_arr,
            y_data=series_y,
            title=title,
            xlabel="Tiempo t",
            ylabel="Volumen Orientado W(t)",
            labels=labels,
            save_path=save_path,
            dpi=dpi,
            show=show,
        )
