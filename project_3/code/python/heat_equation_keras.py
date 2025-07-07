from typing import Callable

import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam, Optimizer

from python.neural_network import keras_ffnn


def sinusoidal_initial(x: tf.Tensor) -> tf.Tensor:
  return tf.sin(tf.constant(np.pi, dtype=tf.float32) * x)


class HeatEquationSolver:
  def __init__(
    self,
    initial_condition: Callable[[tf.Tensor], tf.Tensor],
    hidden_layers: list[int],
    activation_functions: list[str | Callable],
  ):
    self.initial_condition = initial_condition
    self.model = keras_ffnn((2,), 1, hidden_layers, activation_functions, "linear")

  def ffnn_trial(self, x: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
    prediction = self.model(tf.concat([x, t], axis=1))
    return (1.0 - t) * self.initial_condition(x) + x * (1.0 - x) * t * prediction

  def cost_function(self, x: tf.Tensor, t: tf.Tensor) -> float:
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(x)
      tape.watch(t)

      u = self.ffnn_trial(x, t)

      # Having du_dx outside the with block causes d2u_dx2 to be None
      du_dx = tape.gradient(u, x)

    du_dt = tape.gradient(u, t)
    d2u_dx2 = tape.gradient(du_dx, x)
    residual = du_dt - d2u_dx2

    return residual

  def train_ffnn(
    self,
    spatial_size: int,
    time_size: int,
    epochs: int = 1000,
    optimizer: Optimizer = Adam(learning_rate=0.001),
  ) -> float:
    x = tf.cast(np.linspace(0, 1, spatial_size), tf.float32)
    t = tf.cast(np.linspace(0, 1, time_size), tf.float32)

    X, T = tf.meshgrid(x, t)

    x_train = tf.reshape(X, (-1, 1))
    t_train = tf.reshape(T, (-1, 1))

    for epoch in range(epochs):
      # Calculate loss
      with tf.GradientTape(persistent=True) as tape:
        self.cost_function(x_train, t_train)

        total_loss = tf.reduce_mean(tf.square(self.cost_function(x_train, t_train)))

      # Calculate gradients
      gradients = tape.gradient(total_loss, self.model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

      # Print loss periodically
      if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss.numpy()}")

    return total_loss.numpy()

  def calculate_solutions(
    self,
    spatial_size: int,
    time_size: int,
  ):
    x = np.linspace(0, 1, spatial_size)
    t = np.linspace(0, 1, time_size)
    X, T = np.meshgrid(x, t)

    X_flat = X.flatten()[:, np.newaxis]
    T_flat = T.flatten()[:, np.newaxis]
    domain = np.hstack([X_flat, T_flat])

    U_ffnn = self.model.predict(domain).reshape(X.shape)

    return U_ffnn
