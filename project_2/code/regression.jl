include("./hypertuning.jl")
include("./preprocessing.jl")
include("./score.jl")

using LinearAlgebra

"""
Ordinary least squares

Implements the ordinary least squares algorithm using the normal equation or
the singular value decomposition.

# Parameters
- `X::Matrix{Float64}`: The design matrix.
- `y::Vector{Float64}`: The response variable.
- `svd::Bool`: Whether to use the singular value decomposition.

# Returns
- `β::Vector{Float64}`: The estimated regression coefficients.
"""
function ordinary_least_squares(
  X::Matrix{Float64},
  y::Vector{Float64},
  svd_solver::Bool=true,
)::Vector{Float64}

  # Check that x and y have the same number of rows
  if size(X, 1) != length(y)
    throw(DimensionMismatch("X and y must have the same number of rows"))
  end

  if svd_solver
    # Compute coefficients using singular value decomposition
    U, s, V = svd(X)
    β = V * (s .\ (U' * y))
  else
    # Compute coefficients using the normal equation 
    β = X \ y # backslash operator uses QR decomposition by default
  end

  return β
end

"""
Ridge regression

Implements the Ridge regression algorithm using the normal equation or the
singular value decomposition.

# Parameters
- `X::Matrix{Float64}`: The design matrix.
- `y::Vector{Float64}`: The response variable.
- `λ::Float64`: The regularization parameter.
- `svd::Bool`: Whether to use the singular value decomposition.

# Returns
- `β::Vector{Float64}`: The estimated coefficients.
"""
function ridge_regression(
  X::Matrix{Float64},
  y::Vector{Float64},
  λ::Float64,
  svd_solver::Bool=true,
)#::Vector{Float64}

  # Check that x and y have the same number of rows
  if size(X, 1) != length(y)
    throw(DimensionMismatch("X and y must have the same number of rows"))
  end

  if svd_solver
    U, s, V = svd(X)
    d = s ./ (s .^ 2 .+ λ)
    β = V * (d .* (U' * y))
  else
    β = inv(X' * X + λ * I) * X' * y
  end

  return β
end

"""
LASSO regression

Implements the LASSO (least absolute shrinkage and selection operator) regression 
algorithm using the coordinate descent algorithm.

# Parameters
- `X::Matrix{Float64}`: The design matrix.
- `y::Vector{Float64}`: The response variable.
- `λ::Float64`: The regularization parameter.
- `max_iter::Int=1000`: The maximum number of iterations. Defaults to 1000.
- `tol::Float64=1e-6`: The tolerance for convergence. Defaults to 1e-6.

# Returns
- `β::Vector{Float64}`: The estimated coefficients.
"""
function lasso_regression(
  X::Matrix{Float64},
  y::Vector{Float64},
  λ::Float64,
  max_iter::Int=1000,
  tol::Float64=1e-6,
)::Vector{Float64}
  n, m = size(X) # n: number of samples, m: number of features

  # Check that x and y have the same number of rows
  if n != length(y)
    throw(DimensionMismatch("X and y must have the same number of rows"))
  end

  β = zeros(m)

  for _ = 1:max_iter
    β_old = copy(β)

    for j = 1:m
      # Calculate the residual without the j-th feature
      residual = y - X * β + X[:, j] * β[j]
      # Update the j-th coefficient using soft thresholding
      β[j] = soft_threshold(X[:, j]' * residual, λ) / (X[:, j]' * X[:, j])
    end

    # Check for convergence
    if norm(β - β_old) < tol
      break
    end
  end

  return β
end

mutable struct LogisticRegression
  learning_rate::Float64
  l2_lambda::Float64
  num_iterations::Int
  beta_logreg::Union{Nothing,Vector{Float64}}

  function LogisticRegression(learning_rate, l2_lambda, num_iterations)
    new(learning_rate, l2_lambda, num_iterations, nothing)
  end
end

function train_logistic_regression!(model::LogisticRegression, X, y)
  n_data, num_features = size(X)
  model.beta_logreg = zeros(Float64, num_features)

  for _ = 1:model.num_iterations
    linear_model = X * model.beta_logreg
    y_predicted = sigmoid.(linear_model)

    # Gradient calculation with l2 regularization
    gradient = (X' * (y_predicted - y)) / n_data .+ model.l2_lambda * model.beta_logreg

    # Update beta_logreg
    model.beta_logreg -= model.learning_rate * gradient
  end
end

function predict_logistic_regression(
  model::LogisticRegression,
  X::Matrix{Float64},
)::Vector{Float64}
  linear_model = X * model.beta_logreg
  y_predicted = sigmoid.(linear_model)
  return [i >= 0.5 ? 1.0 : 0.0 for i in y_predicted]
end

function evaluate_regression_model(
  x::Matrix{Float64},
  y::Vector{Float64};
  degrees::UnitRange{Int},
  regression_model::Function,
  k_folds::Int=5,
  λ::Float64=0.0,
)
  split = kfold_split(size(x, 1), k_folds, true)

  mse = zeros(length(degrees), k_folds)
  r2 = zeros(length(degrees), k_folds)
  training_times = zeros(length(degrees), k_folds)

  for (i, p) in enumerate(degrees)
    x_poly = polynomial_design_matrix(x, p)

    for k = 1:k_folds
      train_idx, val_idx = get_fold(split, k)

      x_train = x_poly[train_idx, :]
      y_train = y[train_idx]

      x_val = x_poly[val_idx, :]
      y_val = y[val_idx]

      # Train and test model 
      training_start = time_ns()
      β = if λ == 0.0
        ordinary_least_squares(x_train, y_train)
      else
        regression_model(x_train, y_train, λ)
      end
      training_times[p+1, k] = (time_ns() - training_start) / 1e9

      y_pred = x_val * β

      mse[i, k] = mean((y_val - y_pred) .^ 2)
      r2[i, k] = r_squared(y_val, y_pred)
    end
  end

  return mse, r2, training_times
end

function regression_scores(
  model::String,
  mse_scores::Matrix{Float64},
  r2_scores::Matrix{Float64},
  times::Matrix{Float64},
  degrees::UnitRange{Int},
)

  # Calculate means and standard deviations
  mean_mse = mean(mse_scores, dims=2)
  best_fit = argmin(mean_mse)
  std_mse = std(mse_scores, dims=2)

  mean_r2 = mean(r2_scores, dims=2)
  std_r2 = std(r2_scores, dims=2)

  mean_time = mean(times, dims=2)
  std_time = std(times, dims=2)

  return MyModelScore(
    "$model (p=$(degrees[best_fit]))",
    (mean_mse[best_fit], std_mse[best_fit]),
    (mean_r2[best_fit], std_r2[best_fit]),
    (mean_time[best_fit], std_time[best_fit]),
  )
end

function evaluate_logistic_model(
  x::Matrix{Float64},
  y::Vector{Float64};
  logistic_model::LogisticRegression,
  k_folds::Int=5,
)
  split = kfold_split(size(x, 1), k_folds, true)

  accuracy = zeros(k_folds)
  training_times = zeros(k_folds)

  for k = 1:k_folds
    train_idx, val_idx = get_fold(split, k)

    x_train = x[train_idx, :]
    y_train = y[train_idx]

    x_val = x[val_idx, :]
    y_val = y[val_idx]

    # Train and test model 
    training_start = time_ns()
    train_logistic_regression!(logistic_model, x_train, y_train)
    training_times[k] = (time_ns() - training_start) / 1e9

    y_pred = predict_logistic_regression(logistic_model, x_val)

    accuracy[k] = accuracy_score(y_pred, y_val)
  end

  return accuracy, training_times
end