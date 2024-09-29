using Random
using Statistics

function scale_data(x::Matrix{Float64})::Matrix{Float64}
  means = mean(x, dims=1)
  stds = std(x, dims=1)
  x_scaled = (x .- means) ./ stds
  return x_scaled
end

function train_test_split(x::Matrix{Float64}, y::Vector{Float64}, train_ratio::Float64=0.8)
  # Check that train_ratio is between 0.0 and 1.0
  if train_ratio < 0.0 || train_ratio > 1.0
    throw(ArgumentError("train_ratio must be between 0.0 and 1.0"))
  end

  n, _ = size(x)

  # Check that x and y have the same number of rows
  if n != length(y)
    throw(DimensionMismatch("x and y must have the same number of rows"))
  end

  train_size = Int(train_ratio * n)
  indices = randperm(n)

  x_train = x[indices[1:train_size], :]
  y_train = y[indices[1:train_size]]

  x_test = x[indices[train_size+1:end], :]
  y_test = y[indices[train_size+1:end]]

  return x_train, x_test, y_train, y_test
end

function polynomial_design_matrix(x::Matrix{Float64}, degree::Int)::Matrix{Float64}
  n, m = size(x)

  X = ones(n, 1)
  for j = 1:m
    for i = 1:degree
      X = hcat(X, x[:, j] .^ i)
    end
  end
  return X
end

function r_squared(y_true::Vector{Float64}, y_pred::Vector{Float64})::Float64
  rss = sum((y_true .- y_pred) .^ 2) # residual sum of squares
  tss = sum((y_true .- mean(y_true)) .^ 2) # total sum of squares

  return 1 - rss / tss
end

function soft_threshold(x::Float64, λ::Float64)::Float64
  return sign(x) * max(abs(x) - λ, 0)
end

function ordinary_least_squares(
  X::Matrix{Float64},
  y::Vector{Float64},
  svd::Bool=false,
)::Vector{Float64}

  # Check that x and y have the same number of rows
  if size(x, 1) != length(y)
    throw(DimensionMismatch("X and y must have the same number of rows"))
  end

  if svd
    # Compute coefficients using singular value decomposition
    U, S, V = svd(X)
    β = V * inv(Diagonal(S)) * U' * y
  else
    # Compute coefficients using the normal equation
    β = inv(X' * X) * X' * y
  end

  return β
end

function ridge_regression(
  X::Matrix{Float64},
  y::Vector{Float64},
  λ::Float64,
  svd::Bool=false,
)::Vector{Float64}
  n, m = size(x)

  # Check that x and y have the same number of rows
  if n != length(y)
    throw(DimensionMismatch("X and y must have the same number of rows"))
  end

  I = IMatrix(m)
  if svd
    U, S, V = svd(X)
    R = U * S
    β = V * inv(R' * R + λ * I) * (R' * y)
  else
    β = inv(X' * X + λ * I) * (X' * y)
  end

  return β
end

function lasso_regression(
  X::Matrix{Float64},
  y::Vector{Float64},
  λ::Float64,
  max_iter::Int=1000,
  tol::Float64=1e-6,
)::Vector{Float64}
  n, m = size(x)

  # Check that x and y have the same number of rows
  if n != length(y)
    throw(DimensionMismatch("X and y must have the same number of rows"))
  end

  β = zeros(m)

  for _ = 1:max_iter
    beta_old = copy(beta)

    for j = 1:m
      residual = y - X * β + X[:, j] * β[j]
      β[j] = soft_threshold(X[:, j]' * residual[j], λ) / (X[:, j]' * X[:, j])
    end

    # Check for convergence
    if norm(beta - beta_old) < tol
      break
    end
  end

  return β
end

function franke(x::Float64, y::Float64)::Float64
  term1 = 0.75 * exp(-(9 * x .- 2)^2 / 4 - (9 * y - 2)^2 / 4)
  term2 = 0.75 * exp(-(9 * x + 1)^2 / 49 - (9 * y + 1)^2 / 10)
  term3 = 0.5 * exp(-(9 * x - 7)^2 / 4 - (9 * y - 3)^2 / 4)
  term4 = -0.2 * exp(-(9 * x - 4)^2 - (9 * y - 7)^2)
  return term1 + term2 + term3 + term4
end