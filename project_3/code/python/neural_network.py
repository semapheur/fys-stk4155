from typing import Callable

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input


def keras_ffnn(
  input_dim: tuple[int, ...],
  output_dim: int,
  hidden_layers: list[int],
  activation_functions: list[str | Callable],
  output_activation: str | Callable,
) -> Sequential:
  """
  Builds a feedforward neural network with the given architecture and parameters.

  # Parameters
  - `input_dim: tuple[int, ...]`: The input dimension.
  - `output_dim: int`: The number of output features.
  - `hidden_layers: list[int]`: A list of number of neurons in each hidden layer.
  - `activation_functions: list[str | Callable]`: A list of activation functions for each hidden layer.
  - `output_activation: Optional[str | Callable] = None`: The activation function for the output layer (default: None).

  # Returns
  - `Sequential`: A feedforward neural network model with the given architecture and parameters.
  """
  model = Sequential()
  model.add(Input(shape=input_dim))

  for neurons, a in zip(hidden_layers, activation_functions):
    model.add(Dense(neurons, activation=a))

  model.add(Dense(output_dim, activation=output_activation))

  return model
