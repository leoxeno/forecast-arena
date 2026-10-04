"""Deterministic generators for chaotic benchmark series.

Every generator integrates from a fixed initial condition, discards a transient,
and returns a ``pandas.DataFrame`` with one column per state variable. The
same call always returns the same array, so a task built on these needs no
data file and no download.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


def _integrate(rhs, x0, n_points, dt, transient, **kwargs) -> np.ndarray:
    total = n_points + transient
    t_eval = np.arange(total) * dt
    sol = solve_ivp(
        rhs, (0.0, t_eval[-1]), x0, t_eval=t_eval, method="RK45",
        rtol=1e-9, atol=1e-12, args=tuple(kwargs.values()) if kwargs else None,
    )
    return sol.y.T[transient:]


def lorenz63(n_points: int = 2000, dt: float = 0.01, sigma: float = 10.0,
             rho: float = 28.0, beta: float = 8.0 / 3.0, transient: int = 1000) -> pd.DataFrame:
    """Lorenz (1963) system, the canonical chaotic attractor."""
    def rhs(t, s, sigma, rho, beta):
        x, y, z = s
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
    xyz = _integrate(rhs, [1.0, 1.0, 1.0], n_points, dt, transient, sigma=sigma, rho=rho, beta=beta)
    return pd.DataFrame(xyz, columns=["X", "Y", "Z"])


def rossler(n_points: int = 2000, dt: float = 0.05, a: float = 0.2, b: float = 0.2,
            c: float = 5.7, transient: int = 1000) -> pd.DataFrame:
    """Rossler (1976) attractor."""
    def rhs(t, s, a, b, c):
        x, y, z = s
        return [-y - z, x + a * y, b + z * (x - c)]
    xyz = _integrate(rhs, [1.0, 1.0, 1.0], n_points, dt, transient, a=a, b=b, c=c)
    return pd.DataFrame(xyz, columns=["X", "Y", "Z"])


def thomas(n_points: int = 2000, dt: float = 0.2, b: float = 0.208186,
           transient: int = 1000) -> pd.DataFrame:
    """Thomas' cyclically symmetric attractor (Thomas, 1999)."""
    def rhs(t, s, b):
        x, y, z = s
        return [np.sin(y) - b * x, np.sin(z) - b * y, np.sin(x) - b * z]
    xyz = _integrate(rhs, [0.1, 0.0, 0.0], n_points, dt, transient, b=b)
    return pd.DataFrame(xyz, columns=["X", "Y", "Z"])


def mackey_glass(n_points: int = 2000, tau: int = 17, beta: float = 0.2, gamma: float = 0.1,
                 n: int = 10, dt: float = 1.0, substeps: int = 10, transient: int = 1000) -> pd.DataFrame:
    """Mackey-Glass delay differential equation, sampled every ``dt`` time units.

    Integrated with an explicit Euler scheme on a grid of ``dt / substeps``,
    with a constant history of 1.2 before time zero.
    """
    h = dt / substeps
    delay_steps = int(round(tau / h))
    total_steps = (n_points + transient) * substeps
    x = np.empty(total_steps + delay_steps + 1)
    x[: delay_steps + 1] = 1.2
    for i in range(delay_steps, total_steps + delay_steps):
        x_tau = x[i - delay_steps]
        x[i + 1] = x[i] + h * (beta * x_tau / (1.0 + x_tau ** n) - gamma * x[i])
    series = x[delay_steps::substeps][transient: transient + n_points]
    return pd.DataFrame({"X": series})


def henon(n_points: int = 2000, a: float = 1.4, b: float = 0.3, transient: int = 500) -> pd.DataFrame:
    """Henon (1976) map."""
    total = n_points + transient
    x = np.empty(total)
    y = np.empty(total)
    x[0], y[0] = 0.1, 0.1
    for i in range(1, total):
        x[i] = 1.0 - a * x[i - 1] ** 2 + y[i - 1]
        y[i] = b * x[i - 1]
    return pd.DataFrame({"X": x[transient:], "Y": y[transient:]})


def logistic(n_points: int = 2000, r: float = 3.9, transient: int = 500) -> pd.DataFrame:
    """Logistic map in its chaotic regime."""
    total = n_points + transient
    x = np.empty(total)
    x[0] = 0.4
    for i in range(1, total):
        x[i] = r * x[i - 1] * (1.0 - x[i - 1])
    return pd.DataFrame({"X": x[transient:]})


GENERATORS = {
    "lorenz63": lorenz63,
    "rossler": rossler,
    "thomas": thomas,
    "mackey_glass": mackey_glass,
    "henon": henon,
    "logistic": logistic,
}

__all__ = ["lorenz63", "rossler", "thomas", "mackey_glass", "henon", "logistic", "GENERATORS"]
