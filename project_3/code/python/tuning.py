import itertools
import time
from typing import cast, Callable, TypedDict

import jax
import jax.numpy as jnp

from python.heat_equation_jax import HeatEquationSolver, sinusoidal_initial


class TuningResult(TypedDict, total=False):
  hidden_layers: tuple[int, ...]
  final_loss: float
  training_time: float
  activation_function: str | None


def grid_search(
  spatial_size: int = 10,
  time_size: int = 10,
  epochs: int = 1000,
  activation_function: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.sigmoid,
) -> tuple[TuningResult, list[TuningResult]]:
  neuron_options = [8, 16, 32, 64, 128]
  hidden_layer_options = [1, 2, 3]

  # Create all combinations of the grid
  grid_combinations = []
  for num_layers in hidden_layer_options:
    for hidden_layers in itertools.product(neuron_options, repeat=num_layers):
      grid_combinations.append(hidden_layers)

  best_loss = float("inf")
  best_params = None

  results: list[TuningResult] = [{}] * len(grid_combinations)

  # Loop over all grid combinations
  for trial, hidden_layers in enumerate(grid_combinations):
    print(f"\nTrial {trial + 1}/{len(grid_combinations)}...")

    activation_functions = [activation_function] * len(hidden_layers)

    # Create the solver with current configuration
    solver = HeatEquationSolver(
      sinusoidal_initial, list(hidden_layers), activation_functions
    )

    # Train the model and get the final loss
    start_time = time.time()
    loss = solver.train_ffnn(spatial_size, time_size, epochs)
    duration = time.time() - start_time
    print(f"Training Time for this trial: {duration:.2f} seconds")

    results[trial] = TuningResult(
      hidden_layers=hidden_layers, final_loss=loss, training_time=duration
    )

    # Track the best configuration
    if loss < best_loss:
      best_loss = loss
      best_params = TuningResult(
        hidden_layers=hidden_layers,
        final_loss=loss,
        training_time=duration,
      )

  return cast(TuningResult, best_params), results


def tune_activation_functions(
  hidden_layers: list[int],
  activation_functions: list[Callable[[jnp.ndarray], jnp.ndarray]],
  spatial_size: int = 10,
  time_size: int = 10,
  epochs: int = 1000,
):
  results: list[TuningResult] = [{}] * len(activation_functions)

  for i, a in enumerate(activation_functions):
    solver = HeatEquationSolver(
      sinusoidal_initial, hidden_layers, [a] * len(hidden_layers)
    )
    loss = solver.train_ffnn(spatial_size, time_size, epochs)
    start_time = time.time()
    loss = solver.train_ffnn(spatial_size, time_size, epochs)
    duration = time.time() - start_time

    results[i] = TuningResult(
      hidden_layers=tuple(hidden_layers),
      final_loss=loss,
      training_time=duration,
      activation_function=a.__name__,
    )

  return results
