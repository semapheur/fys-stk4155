from typing import Callable

import numpy as np


def sinusoidal_initial(x: np.ndarray) -> np.ndarray:
  return np.sin(np.pi * x)


def analytic_solution(x: np.ndarray, t: np.ndarray, alpha: float = 1.0) -> np.ndarray:
  return np.sin(np.pi * x) * np.exp(-(alpha * np.pi**2) * t)


def heat_equation_forward_euler(
  initial_condition: Callable[[np.ndarray], np.ndarray],
  dx: float,
  dt: float | None = None,
  x_max: float = 1.0,
  t_max: float = 1.0,
  alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """
  Solves the 1D heat equation with Dirichlet boundary conditions using the explicit Euler method.

  This function computes the temperature distribution over a one-dimensional
  rod over time, given an initial temperature distribution and boundary
  conditions. The problem is discretized using the explicit Euler method with
  finite differences.

  Parameters:
    - `initial_condition: Callable[[np.ndarray], np.ndarray]`: A function that
        returns the initial temperature distribution as a numpy array.
    - `dx: float`: Spatial step size.
    - `dt: float | None`: Time step size. If None, it will be calculated using
        the stability condition dt = dx^2 / (2 * alpha).
    - `x_max: float`: The maximum spatial coordinate (length of the rod).
    - `t_max: float`: The maximum time for which to simulate the heat equation.
    - `alpha: float`: The thermal diffusivity of the material.

  Returns:
    `np.ndarray`: A 2D array of shape (spatial_size, time_size) containing the
      temperature distribution at each spatial position and time step.

  Notes:
    The scheme is stable if the condition rho <= 0.5 is satisfied, where
    rho = alpha * dt / dx^2. A warning is printed if this condition is not met.
  """

  if dt is None:
    dt = dx**2 / (2 * alpha)

  x = np.arange(0, x_max + dx, dx)
  t = np.arange(0, t_max + dt, dt)

  nx = len(x)
  nt = len(t)

  rho = alpha * dt / (dx**2)
  print(f"Stability condition (should be <= 0.5): {rho}")
  if rho > 0.5:
    print("Warning: Numerical scheme may be unstable!")

  u_euler = np.zeros((len(x), len(t)))
  u_euler[:, 0] = initial_condition(x)

  for j in range(nt - 1):
    for i in range(1, nx - 1):
      u_euler[i, j + 1] = (
        rho * u_euler[i - 1, j]
        + (1 - 2 * rho) * u_euler[i, j]
        + rho * u_euler[i + 1, j]
      )

  return u_euler, x, t
