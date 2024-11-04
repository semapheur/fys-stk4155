using Random

function linreg_gradient_descent(
  X::Matrix{Float64},
  y::Vector{Float64},
  iter::Int,
  learning_rate::Union{Float64,Nothing}=Nothing,
)
  n, m = size(X)

  # Check that x and y have the same number of rows
  if n != length(y)
    throw(DimensionMismatch("X and y must have the same number of rows"))
  end

  if isnothing(learning_rate)
    hessian = (2.0 / n) * X' * X
    eigenvalues = eigvals(hessian)
    learning_rate = 1.0 / maximum(eigenvalues)
  end

  β = randn(m)

  for _ = 1:iter
    gradient = (2.0 / n) * X' * (X * β - y)
    β -= learning_rate * gradient
  end

  return β
end

function linreg_momentum_gradient_descent(
  X::Matrix{Float64},
  y::Vector{Float64},
  momentum_parameter::Float64,
  iter::Int,
  learning_rate::Union{Float64,Nothing}=Nothing,
)
  n, m = size(X)

  # Check that x and y have the same number of rows
  if n != length(y)
    throw(DimensionMismatch("X and y must have the same number of rows"))
  end

  if isnothing(learning_rate)
    hessian = (2.0 / n) * X' * X
    eigenvalues = eigvals(hessian)
    learning_rate = 1.0 / maximum(eigenvalues)
  end

  β = randn(m)
  momentum = zeros(m)

  for _ = 1:iter
    gradient = (2.0 / n) * X' * (X * β - y)
    momentum = momentum_parameter * momentum - learning_rate * gradient

    β -= momentum
  end

  return β
end

function linreg_stochastic_gradient_descent(
  X::Matrix{Float64},
  y::Vector{Float64},
  batch_size::Int,
  epochs::Int,
  learning_rate_params::NamedTuple,
  learning_rate_schedule::Function,
  replace::Bool=false,
)
  n, m = size(X)

  # Check that x and y have the same number of rows
  if n != length(y)
    throw(DimensionMismatch("X and y must have the same number of rows"))
  end

  # Check if batch_size is less than or equal to n
  if batch_size >= n
    throw(ArgumentError("batch_size must be less than or equal to n"))
  end

  β = randn(m)
  momentum = zeros(m)

  batches = div(length(y), batch_size)
  for epoch = 1:epochs
    indices = replace ? Nothing : shuffle(1:n)

    for batch = 1:batches
      if replace
        rand_index = batch_size * rand(1:n, batch_size)
        batch_indices = rand_index:rand_index+batch_size-1
      else
        start_index = (batch - 1) * batch_size + 1
        end_index = start_index + batch_size - 1
        batch_indices = indices[start_index:end_index]
      end

      X_batch = X[batch_indices, :]
      y_batch = y[batch_indices]
      gradient = (2.0 / batch_size) * X_batch' * (X_batch * β - y_batch)
      learning_rate = learning_rate_schedule(epoch, batch, batches, learning_rate_params...)
      momentum = momentum_parameter * momentum - learning_rate * gradient
      β -= momentum
    end
  end

  return β
end

function linreg_stochastic_momentum_gradient_descent(
  X::Matrix{Float64},
  y::Vector{Float64},
  batch_size::Int,
  epochs::Int,
  momentum_parameter::Float64,
  learning_rate_params::NamedTuple,
  learning_rate_schedule::Function,
)
  n, m = size(X)

  # Check that x and y have the same number of rows
  if n != length(y)
    throw(DimensionMismatch("X and y must have the same number of rows"))
  end

  # Check if batch_size is less than or equal to n
  if batch_size >= n
    throw(ArgumentError("batch_size must be less than or equal to n"))
  end

  β = randn(m)
  momentum = zeros(m)

  batches = div(length(y), batch_size)
  for epoch = 1:epochs
    for batch = 1:batches
      rand_index = batch_size * rand(1:length(y))
      X_batch = X[rand_index:rand_index+batch_size-1, :]
      y_batch = y[rand_index:rand_index+batch_size-1]
      gradient = (2.0 / batch_size) * X_batch' * (X_batch * β - y_batch)
      learning_rate = learning_rate_schedule(epoch, batch, batches, learning_rate_params...)
      momentum = momentum_parameter * momentum - learning_rate * gradient
      β -= momentum
    end
  end

  return β
end

"""
Step decay learning rate schedule.

The learning rate is decayed by a factor of `drop` every `epochs_drop` epochs.
https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1

# Parameters

- `epoch::Int`: The current epoch.
- `initial_learning_rate::Float64`: The initial learning rate. Defaults to 0.1.
- `drop::Float64`: The factor by which to decay the learning rate. Defaults to 0.5.
- `epochs_drop::Float64`:The number of epochs after which to decay the learning rate. Defaults to 10.0.

# Returns
- `learning_rate::Float64`: The learning rate at the current epoch.
"""
function step_decay(
  epoch::Int,
  batch::Int,
  batches::Int,
  initial_learning_rate::Float64=0.1,
  drop::Float64=0.5,
  epochs_drop::Float64=10.0,
)::Float64
  return initial_learning_rate * pow(drop, floor((1 + epoch) / epochs_drop))
end

function time_decay(
  epoch::Int,
  batch::Int,
  batches::Int,
  t0::Float64=1.0,
  t1::Float64=10.0,
)::Float64
  t = epoch * batches + batch
  return t0 / (t + t1)
end

function exponential_decay(
  epoch::Int,
  batch::Int,
  batches::Int,
  κ::Float64=0.1,
  initial_learning_rate::Float64=0.1,
)
  return initial_learning_rate * exp(-κ * batch)
end