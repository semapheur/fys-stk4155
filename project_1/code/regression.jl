using LinearAlgebra
using Random
using Statistics

"""
Transform the data to have zero mean and unit variance.

# Parameters
- `data::Matrix{Float64}`: The input data.

# Returns
- `data_scaled::Matrix{Float64}`: The scaled data.
"""
function standardize_data(data::Matrix{Float64})::Matrix{Float64}
  means = mean(data, dims=1)
  stds = std(data, dims=1)
  data_standardized = (data .- means) ./ stds
  return data_standardized
end

"""
Split the data into training and testing sets using random shuffling

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
function train_test_split(
  x::Matrix{Float64},
  y::Vector{Float64},
  train_ratio::Float64=0.8,
)::Tuple{Matrix{Float64},Matrix{Float64},Vector{Float64},Vector{Float64}}
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
Split the data into k folds for cross-validation.

# Parameters
- `n::Int`: The number of samples.
- `k::Int`: The number of folds.

# Returns
- `folds::Vector{Vector{Int}}`: The folds, each as a vector of indices.
"""
function kfold_split(n::Int, k::Int)
  # Ensure k is not larger than n
  k = min(k, n)

  # Create random permutation of indices
  indices = randperm(n)

  # Calculate fold sizes
  fold_sizes = fill(n ÷ k, k)
  remainder = n % k
  for i = 1:remainder
    fold_sizes[i] += 1
  end

  # Create folds
  folds = Vector{Vector{Int}}(undef, k)
  start_idx = 1
  for i = 1:k
    end_idx = start_idx + fold_sizes[i] - 1
    folds[i] = indices[start_idx:end_idx]
    start_idx = end_idx + 1
  end

  return folds
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
  term1 = 0.75 * exp(-(9 * x - 2)^2 / 4 - (9 * y - 2)^2 / 4)
  term2 = 0.75 * exp(-(9 * x + 1)^2 / 49 - (9 * y + 1)^2 / 10)
  term3 = 0.5 * exp(-(9 * x - 7)^2 / 4 - (9 * y - 3)^2 / 4)
  term4 = -0.2 * exp(-(9 * x - 4)^2 - (9 * y - 7)^2)
  return term1 + term2 + term3 + term4
end

function franke_training_data(
  n::Int,
  noise_amplitude::Float64=0.1,
  random_inputs::Bool=false,
)::Tuple{Matrix{Float64},Vector{Float64}}
  if random_inputs
    x_1 = rand(n)
    x_2 = rand(n)
  else
    x_1 = range(0, 1, length=n)
    x_2 = range(0, 1, length=n)
  end

  Y = franke.(x_1', x_2)

  if noise_amplitude != 0.0
    Y .+= noise_amplitude * randn(n, n)
  end

  y = vec(Y)

  # Create design matrix
  X_1 = repeat(x_1, n)
  X_2 = repeat(x_2', n)
  X = hcat(X_1, vec(X_2))

  return X, y
end

"""
Read a GeoTIFF file and sample `samples` random points.

# Parameters
- `file_path::String`: The path to the GeoTIFF file.
- `samples::Int`: The number of samples to take.

# Returns
- `X::Matrix{Float64}`: A matrix of size `(samples, 2)` containing the
  coordinates of the sampled points.
- `y::Vector{Float64}`: A vector of length `samples` containing the values
  of the sampled points.

# Throws
- `ArgumentError`: If the file is not found or if the requested samples exceed
  the available pixels.
"""
function tif_training_data(file_path::String, samples::Int)
  if !isfile(file_path)
    throw(ArgumentError("File not found: $file_path"))
  end

  dataset = ArchGDAL.read(file_path)
  band_data = ArchGDAL.read(dataset, 1)

  rows, cols = size(band_data)
  if samples > rows * cols
    throw(
      ArgumentError(
        "Requested samples ($samples) exceed available pixels ($(rows * cols))",
      ),
    )
  end

  y = Vector{Float64}(undef, samples)
  x_1 = Vector{Float64}(undef, samples)
  x_2 = Vector{Float64}(undef, samples)

  sampled_points = Set{Tuple{Int,Int}}()
  i = 1
  while i <= samples
    row = rand(1:rows)
    col = rand(1:cols)

    if (row, col) ∉ sampled_points
      push!(sampled_points, (row, col))
      x_1[i] = row
      x_2[i] = col
      y[i] = band_data[row, col]
      i += 1
    end
  end

  X = hcat(x_1, x_2)

  return X, y
end

"""
Generate all combinations of polynomial terms for a given number of variables
and degree.

# Parameters
- `variables::Int`: The number of variables.
- `degree::Int`: The degree of the polynomial.

# Returns
- `combinations::Vector{Vector{Int}}`: A vector of vectors, where each inner vector
  is a combination of powers of the variables, which sum up to the given degree.
"""
function polynomial_combinations(variables::Int, degree::Int)::Vector{Vector{Int}}
  function generate_combinations(
    remaining_degree::Int,
    current_pos::Int,
    current_combo::Vector{Int},
  )
    if current_pos > variables
      # Only keep the combination if we've used all the remaining degree
      if remaining_degree == 0
        return [copy(current_combo)]
      else
        return Vector{Int}[]
      end
    end

    combinations = Vector{Vector{Int}}()

    # Try all possible degrees for the current position
    for d = 0:remaining_degree
      current_combo[current_pos] = d
      append!(
        combinations,
        generate_combinations(remaining_degree - d, current_pos + 1, current_combo),
      )
    end

    return combinations
  end

  # Initialize storage for all combinations across all degrees
  all_combinations = Vector{Vector{Int}}()
  current_combo = zeros(Int, variables)

  # Generate combinations for each degree up to p
  for degree = 0:degree
    append!(all_combinations, generate_combinations(degree, 1, current_combo))
  end

  return all_combinations
end

"""
Generates a polynomial design matrix.

# Parameters
- `data::Matrix{Float64}`: The input data.
- `degree::Int`: The degree of the polynomial.

# Returns
- `design_matrix::Matrix{Float64}`: The polynomial design matrix.
"""
function polynomial_design_matrix(data::Matrix{Float64}, degree::Int)::Matrix{Float64}
  n, m = size(data) # n: number of samples, m: number of features

  # Create the design matrix
  combinations = polynomial_combinations(m, degree)
  num_terms = length(combinations)
  design_matrix = zeros(Float64, n, num_terms)

  # Fill the design matrix with polynomial features
  for (idx, combination) in enumerate(combinations)
    term = ones(n)  # Start with a vector of ones (constant term)
    for (feature_index, degree) in enumerate(combination)
      term .= term .* (data[:, feature_index] .^ degree)
    end
    design_matrix[:, idx] = term
  end

  return design_matrix
end
