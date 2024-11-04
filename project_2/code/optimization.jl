include("./neural_network.jl")
include("./preprocessing.jl")
include("./score.jl")

using Random
using Zygote
using Statistics

function ridge_cost(β::Vector{Float64}, X::Matrix{Float64}, y::Vector{Float64}, λ::Float64)
  n = size(X, 1)
  predictions = X * β
  mse = sum((predictions - y) .^ 2) / (2n)
  regularization = (λ / 2) * sum(β .^ 2)
  return mse + regularization
end

function ridge_gradient(
  β::Vector{Float64},
  X::Matrix{Float64},
  y::Vector{Float64},
  λ::Float64,
)
  f = β -> ridge_cost(β, X, y, λ)
  return Zygote.gradient(f, β)[1]
end

function linreg_gradient_descent(
  X::Matrix{Float64},
  y::Vector{Float64};
  iter::Int,
  l2_lambda::Float64=0.0,
  learning_rate::Union{Float64,Nothing}=Nothing,
)
  n, m = size(X)

  # Check that x and y have the same number of rows
  if n != length(y)
    throw(DimensionMismatch("X and y must have the same number of rows"))
  end

  if isnothing(learning_rate)
    hessian = (2.0 / n) * X' * X + 2.0 * l2_lambda * I
    eigenvalues = eigvals(hessian)
    learning_rate = 1.0 / maximum(eigenvalues)
  end

  β = zeros(m)
  losses = Vector{Float64}()
  for _ = 1:iter
    gradient = ((2.0 / n) * X' * (X * β - y) + 2.0 * l2_lambda * β)
    β -= learning_rate * gradient

    prediction = X * β
    loss = mean((prediction .- y) .^ 2)

    push!(losses, loss)
  end

  return β, losses
end

"""
Linear Regression with Gradient Descent Momentum

This function performs linear regression with gradient descent
and momentum.

Arguments:
- `X::Matrix{Float64}`: The design matrix of the model.
- `y::Vector{Float64}`: The target vector of the model.
- `iter::Int`: The number of iterations to perform.
- `momentum_parameter::Float64`: The momentum parameter. Defaults to 0.
- `l2_lambda::Float64`: The regularization parameter. Defaults to 0.
- `learning_rate::Union{Float64,Nothing}`: The learning rate. If `nothing`, it is set to 1 / max eigenvalue of the hessian.

Returns:
- The estimated coefficients of the model.
- The loss at each iteration.
"""
function linreg_gradient_descent_momentum(
  X::Matrix{Float64},
  y::Vector{Float64},
  iter::Int,
  momentum_parameter::Float64=0.0,
  l2_lambda::Float64=0.0,
  learning_rate::Union{Float64,Nothing}=Nothing,
)
  n, m = size(X)

  # Check that x and y have the same number of rows
  if n != length(y)
    throw(DimensionMismatch("X and y must have the same number of rows"))
  end

  if isnothing(learning_rate)
    hessian = (2.0 / n) * X' * X + 2.0 * l2_lambda * I
    eigenvalues = eigvals(hessian)
    learning_rate = 1.0 / maximum(eigenvalues)
  end

  β = randn(m)
  momentum = zeros(m)

  losses = Vector{Float64}()
  for _ = 1:iter
    gradient = ((2.0 / n) * X' * (X * β - y) + 2.0 * l2_lambda * β)
    momentum = momentum_parameter * momentum - learning_rate * gradient

    β -= momentum
    prediction = X * β
    loss = mean((prediction .- y) .^ 2)
    push!(losses, loss)
  end

  return β, losses
end

function linreg_stochastic_gradient_descent(
  X::Matrix{Float64},
  y::Vector{Float64},
  batch_size::Int,
  epochs::Int,
  learning_rate::Float64,
  momentum_parameter::Float64=0.0,
  l2_lambda::Float64=0.0,
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
  losses = Vector{Float64}()

  for epoch = 1:epochs
    epoch_loss = 0.0

    for (X_batch, y_batch) in get_batches(X, y, batch_size)
      gradient = (2.0 / batch_size) * X_batch' * (X_batch * β - y_batch)
      gradient =
        ((2.0 / batch_size) * X_batch' * (X_batch * β - y_batch) + 2.0 * l2_lambda * β)
      momentum = momentum_parameter * momentum - learning_rate * gradient
      β -= momentum

      predictions = X_batch * β
      batch_loss = mean((y_batch - predictions) .^ 2)
      epoch_loss += batch_loss
    end
    push!(losses, epoch_loss / size(X_batch, 1))
  end

  return β, loss_history
end

function linreg_adaptive_stochastic_gradient_descent(
  X::Matrix{Float64},
  y::Vector{Float64},
  epochs::Int,
  batch_size::Int,
  learning_rate::Float64=0.1,
  momentum_parameter::Float64=0.0,
  l2_lambda::Float64=0.0,
  epsilon::Float64=1e-6,
)
  n, m = size(X)

  # Check that x and y have the same number of rows
  if n != length(y)
    throw(DimensionMismatch("X and y must have the same number of rows"))
  end

  β = zeros(m)
  gradients = zeros(m)
  losses = Vector{Float64}()

  for epoch = 1:epochs
    epoch_loss = 0.0
    gradients = 0.0

    for (X_batch, y_batch) in get_batches(X, y, batch_size)
      gradient =
        gradient = (2.0 / n) * X_batch' * (X_batch * β - y_batch) + 2.0 * l2_lambda * β
      gradients += gradient .^ 2
      adaptive_lr = learning_rate / (sqrt(gradients) + epsilon)
      momentum = momentum_parameter * momentum - adaptive_lr * gradient
      β -= momentum

      predictions = X_batch * β
      batch_loss = mean((y_batch - predictions) .^ 2)
      epoch_loss += batch_loss
    end
    push!(losses, epoch_loss / size(X_batch, 1))
  end

  return β, losses
end

function linreg_rmsprop_stochastic_gradient_descent(
  X::Matrix{Float64},
  y::Vector{Float64},
  epochs::Int,
  batch_size::Int,
  learning_rate::Float64=0.1,
  momentum_parameter::Float64=0.0,
  l2_lambda::Float64=0.0,
  rho::Float64=0.99,
)
  n, m = size(X)

  # Check that x and y have the same number of rows
  if n != length(y)
    throw(DimensionMismatch("X and y must have the same number of rows"))
  end

  β = zeros(m)
  gradients = zeros(m)
  losses = Vector{Float64}()

  for epoch = 1:epochs
    epoch_loss = 0.0
    gradients = 0.0

    for (X_batch, y_batch) in get_batches(X, y, batch_size)
      gradient =
        gradient = (2.0 / n) * X_batch' * (X_batch * β - y_batch) + 2.0 * l2_lambda * β

      gradients = (rho * gradients) + (1.0 - rho) * gradient .^ 2
      update = learning_rate * gradient / (sqrt(gradients) .+ epsilon)
      β -= update

      predictions = X * β
      batch_loss = mean((y_batch - predictions) .^ 2)
      epoch_loss += batch_loss
    end

    push!(losses, epoch_loss / size(X_batch, 1))
  end

  return β, losses
end

function linreg_adam_stochastic_gradient_descent(
  X::Matrix{Float64},
  y::Vector{Float64},
  epochs::Int,
  batch_size::Int,
  learning_rate::Float64=0.1,
  l2_lambda::Float64=0.0,
  theta1::Float64=0.9,
  theta2::Float64=0.999,
  epsilon::Float64=1e-8,
)
  n, m = size(X)

  # Check that x and y have the same number of rows
  if n != length(y)
    throw(DimensionMismatch("X and y must have the same number of rows"))
  end

  β = zeros(m)
  losses = Vector{Float64}()

  for epoch = 1:epochs
    epoch_loss = 0.0
    gradients = 0.0

    for (X_batch, y_batch) in get_batches(X, y, batch_size)
      gradient =
        gradient = (2.0 / n) * X_batch' * (X_batch * β - y_batch) + 2.0 * l2_lambda * β

      first_moment = theta1 * first_moment + (1 - theta1) * gradients
      second_moment = theta2 * second_moment + (1 - theta2) * gradients * gradients

      first_term = first_moment / (1.0 - theta1^epoch)
      second_term = second_moment / (1.0 - theta2^epoch)

      update = learning_rate * first_term / (sqrt(second_term) .+ epsilon)
      β -= update

      predictions = X * β
      batch_loss = mean((y_batch - predictions) .^ 2)
      epoch_loss += batch_loss
    end

    push!(losses, epoch_loss / size(X_batch, 1))
  end

  return β, losses
end

struct RidgeScore
  degree::Int
  learning_rate::Float64
  l2_lambda::Float64
  momentum::Float64
  mse_mean_std::Tuple{Float64,Float64}
  r2_mean_std::Tuple{Float64,Float64}
end

function evaluate_model(
  x::Matrix{Float64},
  y::Vector{Float64},
  params::NamedTuple,
  model::Function,
  k_folds::Int,
)
  mse_scores = zeros(k_folds)
  r2_scores = zeros(k_folds)
  split = kfold_split(size(x, 1), k_folds, true)

  for i = 1:k_folds
    train_idx, val_idx = get_fold(split, i)

    x_train = x[train_idx, :]
    y_train = y[train_idx]

    x_val = x[val_idx, :]
    y_val = y[val_idx]

    β, _ = model(x_train, y_train, params...)

    y_pred = x_val * β
    mse_scores[i] = mean((y_val - y_pred) .^ 2)
    r2_scores[i] = r_squared(y_val, y_pred)
  end

  return skipmissing(mse_scores), skipmissing(r2_scores)
end

function ridge_momentum_gradient_descent_random_search(
  X::Matrix{Float64},
  y::Vector{Float64},
  k_folds::Int=10,
  trials::Int=100,
  max_degrees::Int=10,
  lr_range::Tuple{Float64,Float64}=(0.0001, 0.1),
  l2_lambda_range::Tuple{Float64,Float64}=(0.0001, 0.1),
  momentum_range::Tuple{Float64,Float64}=(0.0, 0.9),
)::Tuple{Int,Vector{RidgeScore}}
  best_score = Inf
  best_idx = nothing

  results = Vector{RidgeScore}()

  for _ = 1:trials
    degree = rand(0:max_degrees)
    lr = exp(uniform(log(lr_range[1]), log(lr_range[2])))
    l2_lambda = exp(uniform(log(l2_lambda_range[1]), log(l2_lambda_range[2])))
    momentum = uniform(momentum_range[1], momentum_range[2])

    X_poly = polynomial_design_matrix(X, degree)

    mse_scores, r2_scores = evaluate_model(
      X_poly,
      y,
      (iter=1000, momentum=momentum, l2_lambda=l2_lambda, lr=lr),
      linreg_gradient_descent_momentum,
      k_folds,
    )

    mse_mean = mean(mse_scores)

    scores = RidgeScore(
      degree,
      lr,
      l2_lambda,
      momentum,
      (mse_mean, std(mse_scores)),
      (mean(r2_scores), std(r2_scores)),
    )
    push!(results, scores)

    if mse_mean < best_score
      best_score = mse_mean
      best_idx = length(results)
    end
  end

  return best_idx, results
end

function uniform(a, b)
  return a + (b - a) * rand()
end