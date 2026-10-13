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
            El nuevo vector de estado Y_{n+1} tras el avance temporal.
        """   
    k1=function(t,y)
    step=k1*dt
    return y+step
@njit
def rk4_step(function, t, y, dt): 
    k1=function(t,y)
    k2=function(t+0.5*dt,y+0.5*dt*k1)
    k3=function(t + 0.5*dt,  y + 0.5*dt*k2)
    k4=function(t+dt,y+dt*k3)
    step=(dt/6.0)*(k1+2*k2+2*k3+k4)
    return y+step
@njit
def symplectic_euler_step(function, t, y, dt): 
    k1=function(t,y)
    half=len(y)//2
    new_y=np.copy(y)
    new_y[half:]=y[half:]+dt*k1[half:]
    new_y[:half]=y[:half]+dt*new_y[half:]
    return new_y

@njit
def rk4(function,y0,t0,tf,dt):
    """
    Resuelve un Problema de Valor Inicial (Cauchy) utilizando el método de Runge-Kutta de 4to Orden.
    Parametros:
    function : Callable
        Función matemática del sistema dinámico que retorna las derivadas. y'=f(t,y)
    y0 : np.ndarray
        Vector 1D con el estado inicial (condiciones iniciales) del sistema.
    t0 : float
        Instante de tiempo inicial de la simulación.
    tf : float
        Instante de tiempo final de la simulación.
    dt : float
        Tamaño del paso temporal de integración.

    Return
    tuple
        Una tupla (times, trajectories) que contiene:
        - times (np.ndarray): Vector 1D con los instantes de tiempo evaluados.
        - trajectories (np.ndarray): Matriz 2D de dimensiones (pasos, len(y0)). 
        Cada fila 'i' representa el estado del sistema en el instante 'times[i]'.
    """
    steps=int((tf-t0)/dt)
    times=np.linspace(t0,tf,steps)
    trajectories=np.zeros((steps,len(y0)),dtype=np.float64)
    actual_y=np.copy(y0)
    for i in range(steps):
        trajectories[i]=actual_y
        actual_t=times[i]
        actual_y=rk4_step(function,actual_t,actual_y,dt)
    return times, trajectories

@njit
def euler(function,y0,t0,tf,dt):
    steps=int((tf-t0)/dt)
    times=np.linspace(t0,tf,steps)
    trajectories=np.zeros((steps,len(y0)),dtype=np.float64)
    actual_y=np.copy(y0)
    for i in range(steps):
        trajectories[i]=actual_y
        actual_t=times[i]
        actual_y=euler_step(function,actual_t,actual_y,dt)
    return times, trajectories


@njit
def symplectic_euler(function,y0,t0,tf,dt):
    steps=int((tf-t0)/dt)
    times=np.linspace(t0,tf,steps)
    trajectories=np.zeros((steps,len(y0)),dtype=np.float64)
    actual_y=np.copy(y0)
    for i in range(steps):
        trajectories[i]=actual_y
        actual_t=times[i]
        actual_y=symplectic_euler_step(function,actual_t,actual_y,dt)
    return times, trajectories