import numpy as np
from numba import njit


@njit
def euler_step(function, t, y, dt):
    """
    Ejecuta un paso de integración espacial usando el método de Euler de 1er Orden.

    Parametros:
    function : Callable
        Función matemática del sistema dinámico que retorna las derivadas. y'=f(t,y)
        f(t: float, y: np.ndarray) -> np.ndarray.
    t : float
        Tiempo actual t_n.
    y : np.ndarray
        Vector de estado actual Y_n (posiciones, velocidades, etc).
    dt : float
        Tamaño del paso de integración.
    Return
    np.ndarray
        tuple[np.ndarray, np.ndarray]: (nuevo_estado Y_{n+1}, derivada_actual k_1)
    """
    k1 = function(t, y)
    next_y = y + dt * k1
    return next_y, k1


@njit
def rk4_step(function, t, y, dt):
    k1 = function(t, y)
    k2 = function(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = function(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = function(t + dt, y + dt * k3)
    next_y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return next_y, k1


@njit
def symplectic_euler_step(function, t, y, dt):
    """
    Ejecuta un paso de integración con el método de Euler Simpléctico (1er orden)
    para sistemas Hamiltonianos separables en espacio fase de dimensión par N = 2n.

    Convención de coordenadas del estado 'y':
        - Primera mitad y[:half] : Coordenadas generalizadas q^i (posiciones)
        - Segunda mitad y[half:] : Momentos conjugados / velocidades p_i (o v^i)

    Soporta tanto vector 1D (2n,) como matriz 2D (2n, M).

    Retorna:
        tuple[np.ndarray, np.ndarray]: (nuevo_estado Y_{n+1}, derivada_actual k_1)
    """
    k1 = function(t, y)
    half = y.shape[0] // 2
    new_y = np.copy(y)
    new_y[half:] = y[half:] + dt * k1[half:]
    new_y[:half] = y[:half] + dt * new_y[half:]
    return (new_y, k1)


@njit
def rk4(function, y0, t0, tf, dt):
    """
    Integra un Problema de Cauchy completo mediante Runge-Kutta de 4to Orden (RK4).

    Retorna:
        tuple: (times, trajectories, derivatives)
    """
    n_intervals = int(np.floor((tf - t0) / dt))
    steps = n_intervals + 1
    times = np.empty(steps, dtype=np.float64)
    trajectories = np.zeros((steps,) + y0.shape, dtype=np.float64)
    derivatives = np.zeros((steps,) + y0.shape, dtype=np.float64)
    actual_y = np.copy(y0)
    for i in range(n_intervals):
        actual_t = t0 + i * dt
        times[i] = actual_t
        trajectories[i] = actual_y
        actual_y, k1 = rk4_step(function, actual_t, actual_y, dt)
        derivatives[i] = k1

    # Registro del estado y derivada en el instante final tf
    final_t = t0 + n_intervals * dt
    times[n_intervals] = final_t
    trajectories[n_intervals] = actual_y
    derivatives[n_intervals] = function(final_t, actual_y)
    return times, trajectories, derivatives


@njit
def euler(function, y0, t0, tf, dt):
    """
    Integra un Problema de Cauchy completo mediante Euler explícito.

    Parámetros:
        function : Callable compilado con @njit -> f(t, y)
        y0       : np.ndarray de forma (N,) o (N, M)
        t0, tf   : float (intervalo temporal [t0, tf])
        dt       : float (paso de integración)

    Retorna:
        tuple: (times, trajectories, derivatives)
            - times        : np.ndarray de forma (steps,)
            - trajectories : np.ndarray de forma (steps, N) o (steps, N, M)
            - derivatives  : np.ndarray de forma (steps, N) o (steps, N, M)
    """
    n_intervals = int(np.floor((tf - t0) / dt))
    steps = n_intervals + 1
    times = np.empty(steps, dtype=np.float64)
    trajectories = np.zeros((steps,) + y0.shape, dtype=np.float64)
    derivatives = np.zeros((steps,) + y0.shape, dtype=np.float64)
    actual_y = np.copy(y0)
    for i in range(n_intervals):
        actual_t = t0 + i * dt
        times[i] = actual_t
        trajectories[i] = actual_y
        actual_y, k1 = euler_step(function, actual_t, actual_y, dt)
        derivatives[i] = k1
    # Registro del estado y derivada en el instante final tf
    final_t = t0 + n_intervals * dt
    times[n_intervals] = final_t
    trajectories[n_intervals] = actual_y
    derivatives[n_intervals] = function(final_t, actual_y)
    return times, trajectories, derivatives


@njit
def symplectic_euler(function, y0, t0, tf, dt):
    n_intervals = int(np.floor((tf - t0) / dt))
    steps = n_intervals + 1
    times = np.empty(steps, dtype=np.float64)
    trajectories = np.zeros((steps,) + y0.shape, dtype=np.float64)
    derivatives = np.zeros((steps,) + y0.shape, dtype=np.float64)
    actual_y = np.copy(y0)
    for i in range(n_intervals):
        actual_t = t0 + i * dt
        times[i] = actual_t
        trajectories[i] = actual_y
        actual_y, k1 = symplectic_euler_step(function, actual_t, actual_y, dt)
        derivatives[i] = k1

    # Registro del estado y derivada en el instante final tf
    final_t = t0 + n_intervals * dt
    times[n_intervals] = final_t
    trajectories[n_intervals] = actual_y
    derivatives[n_intervals] = function(final_t, actual_y)

    return times, trajectories, derivatives
