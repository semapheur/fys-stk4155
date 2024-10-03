using Random
using Statistics

"""
Scale the data to have zero mean and unit variance.

# Parameters
- `x::Matrix{Float64}`: The input data.

# Returns
- `x_scaled::Matrix{Float64}`: The scaled data.
"""
function scale_data(x::Matrix{Float64})::Matrix{Float64}
  means = mean(x, dims=1)
  stds = std(x, dims=1)
  x_scaled = (x .- means) ./ stds
  return x_scaled
end

"""
Split the data into training and testing sets.

# Parameters
- `x::Matrix{Float64}`: The input data.
- `y::Vector{Float64}`: The response variable.
- `train_ratio::Float64=0.8`: The proportion of the data to include in the training set. Defaults to 0.8.

# Returns
- `x_train::Matrix{Float64}`: The training set
- `x_test::Matrix{Float64}`: The testing set.
- `y_train::Vector{Float64}`: The response variable for the training set.
- `y_test::Vector{Float64}`: The response variable for the testing set.
"""
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

"""
Generates a polynomial design matrix.

# Parameters
- `x::Matrix{Float64}`: The input data.
- `degree::Int`: The degree of the polynomial.

# Returns
- `X::Matrix{Float64}`: The polynomial design matrix.
"""
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

"""
Calculates the R-squared value of the prediction.

# Parameters
- `y_true::Vector{Float64}`: The target values.
- `y_pred::Vector{Float64}`: The predicted values.

# Returns
- `r2::Float64`: The R-squared value.
"""
function r_squared(y_true::Vector{Float64}, y_pred::Vector{Float64})::Float64
  rss = sum((y_true .- y_pred) .^ 2) # residual sum of squares
  tss = sum((y_true .- mean(y_true)) .^ 2) # total sum of squares

  return 1 - rss / tss
end

"""
Implements the soft thresholding operator.

# Parameters
- `x::Float64`: The value to be thresholded.
- `λ::Float64`: The regularization parameter.

# Returns
- `x_thresholded::Float64`: The thresholded value.
"""
function soft_threshold(x::Float64, λ::Float64)::Float64
  return sign(x) * max(abs(x) - λ, 0)
end

"""
Ordinary least squares

Implements the ordinary least squares algorithm using the normal equation or
the singular value decomposition.

# Parameters
- `X::Matrix{Float64}`: The design matrix.
- `y::Vector{Float64}`: The response variable.
- `svd::Bool=false`: Whether to use the singular value decomposition. Defaults to false.

# Returns
- `β::Vector{Float64}`: The estimated regression coefficients.
"""
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

"""
Ridge regression

Implements the Ridge regression algorithm using the normal equation or the
singular value decomposition.

# Parameters
- `X::Matrix{Float64}`: The design matrix.
- `y::Vector{Float64}`: The response variable.
- `λ::Float64`: The regularization parameter.
- `svd::Bool=false`: Whether to use the singular value decomposition. Defaults to false.

# Returns
- `β::Vector{Float64}`: The estimated coefficients.
"""
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
  n, m = size(x)

  # Check that x and y have the same number of rows
  if n != length(y)
    throw(DimensionMismatch("X and y must have the same number of rows"))
  end

  β = zeros(m)

  for _ = 1:max_iter
    β_old = copy(β)

    for j = 1:m
      residual = y - X * β + X[:, j] * β[j]
      β[j] = soft_threshold(X[:, j]' * residual[j], λ) / (X[:, j]' * X[:, j])
    end

    # Check for convergence
    if norm(β - β_old) < tol
      break
    end
  end

  return β
end

"""
The Franke function

The Franke function is a commonly used function for testing
interpolation algorithms. It is a 2D function ``f:ℝ² → ℝ`` of the form

``f(x, y) = 0.75 * exp(-(9x - 2)² / 4 - (9y - 2)² / 4) +
            0.75 * exp(-(9x + 1)² / 49 - (9y + 1)² / 10) +
            0.5 * exp(-(9x - 7)² / 4 - (9y - 3)² / 4) +
            -0.2 * exp(-(9x - 4)² - (9y - 7)²)``

where x and y are the input coordinates.
"""
function franke(x::Float64, y::Float64)::Float64
  term1 = 0.75 * exp(-(9 * x .- 2)^2 / 4 - (9 * y - 2)^2 / 4)
  term2 = 0.75 * exp(-(9 * x + 1)^2 / 49 - (9 * y + 1)^2 / 10)
  term3 = 0.5 * exp(-(9 * x - 7)^2 / 4 - (9 * y - 3)^2 / 4)
  term4 = -0.2 * exp(-(9 * x - 4)^2 - (9 * y - 7)^2)
  return term1 + term2 + term3 + term4
end
