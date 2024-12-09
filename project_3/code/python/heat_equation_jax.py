from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax


def sinusoidal_initial(x):
  return jnp.sin(jnp.pi * x)


class HeatEquationSolver:
  def __init__(
    self,
    initial_condition: Callable[[jnp.ndarray], jnp.ndarray],
    hidden_layers: list[int],
    activation_functions: list[Callable],
  ):
    self.initial_condition = initial_condition

    class FFNN(nn.Module):
      hidden_layers: list[int]
      activation_functions: list[Callable]

      @nn.compact
      def __call__(self, x):
        activations = self.activation_functions or [nn.relu] * len(self.hidden_layers)

        for neurons, a in zip(self.hidden_layers, activations):
          x = nn.Dense(
            features=neurons,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
          )(x)
          x = a(x)

        x = nn.Dense(1)(x)

        return x

    self.model = FFNN(hidden_layers, activation_functions)
    input_shape = (2,)
    self.key = jax.random.PRNGKey(0)
    self.params = self.model.init(self.key, jnp.zeros(input_shape))["params"]

  def ffnn_trial(self, params, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    x = jnp.atleast_2d(x)
    t = jnp.atleast_2d(t)

    x_t = jnp.concatenate([x, t], axis=1)

    prediction = self.model.apply({"params": params}, x_t)
    result = (1.0 - t) * self.initial_condition(x) + x * (1.0 - x) * t * prediction
    return result.squeeze()

  def cost_function(self, params, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    def u_func(x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
      return self.ffnn_trial(params, x, t)

    du_dt = jax.jacrev(u_func, argnums=1)(x, t)
    d2u_dx2 = jax.jacrev(jax.jacfwd(u_func, argnums=0), argnums=0)(x, t)
    residual = du_dt - d2u_dx2

    return residual

  def train_ffnn(
    self,
    spatial_size: int,
    time_size: int,
    epochs: int = 1000,
    learning_rate: float = 0.001,
  ):
    x = jnp.linspace(0, 1, spatial_size)
    t = jnp.linspace(0, 1, time_size)

    X, T = jnp.meshgrid(x, t)

    x_train = X.reshape(-1, 1)
    t_train = T.reshape(-1, 1)

    def loss_fn(params):
      residuals = jax.vmap(lambda x, t: self.cost_function(params, x, t))(
        x_train, t_train
      )
      return jnp.mean(jnp.square(residuals))

    # Optimizer
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(self.params)

    # Update step
    @jax.jit
    def update(
      params: optax.Params, opt_state: optax.OptState
    ) -> tuple[optax.Params, optax.OptState, float]:
      loss, grads = jax.value_and_grad(loss_fn)(params)
      updates, new_opt_state = optimizer.update(grads, opt_state)
      new_params = optax.apply_updates(params, updates)
      return new_params, new_opt_state, loss

    # Training loop with early stopping
    best_loss = float("inf")
    patience = 200
    wait = 0
    best_params = self.params

    for epoch in range(epochs):
      self.params, opt_state, loss = update(self.params, opt_state)

      if loss < best_loss:
        best_loss = loss
        best_params = self.params
        wait = 0
      else:
        wait += 1

      if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")
        print(f"Best loss: {best_loss}")

        if wait > patience:
          print(f"Early stopping at epoch {epoch}")
          break

    self.params = best_params

    return loss.item()

  def calculate_solution(
    self,
    spatial_size: int,
    time_size: int,
  ):
    x = jnp.linspace(0, 1, spatial_size)
    t = jnp.linspace(0, 1, time_size)
    X, T = jnp.meshgrid(x, t)

    X_flat = X.flatten()[:, jnp.newaxis]
    T_flat = T.flatten()[:, jnp.newaxis]

    # Call the neural network to get the solution U for all points in the domain
    u_ffnn = self.ffnn_trial(self.params, X_flat, T_flat).reshape(X.shape)

    return u_ffnn
