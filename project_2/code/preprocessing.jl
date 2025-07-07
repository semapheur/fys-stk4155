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

struct KFoldSplit
  train_indices::Vector{Vector{Int}}
  test_indices::Vector{Vector{Int}}
  fold_sizes::Vector{Int}
  shuffle::Bool
  seed::Union{Nothing,Int}
end

"""
Split the data into k folds for cross-validation.

# Parameters
- `n::Int`: The number of samples.
- `k::Int`: The number of folds.
- `shuffle::Bool=true`: Whether to shuffle the data before splitting.
- `seed::Union{Nothing,Int}=nothing`: The random seed to use if shuffling.

# Returns
- `kfold_split::KFoldSplit`: A `KFoldSplit` object containing the train and test
  indices for each fold.
"""
function kfold_split(n::Int, k::Int, shuffle::Bool=true, seed::Union{Nothing,Int}=nothing)
  # Input validation
  if n < 2
    throw(ArgumentError("Number of samples must be at least 2, got $n"))
  end
  if k < 2
    throw(ArgumentError("Number of folds must be at least 2, got $k"))
  end
  if k > n
    throw(
      ArgumentError("Number of folds ($k) cannot be larger than number of samples ($n)"),
    )
  end

  # Set random seed if provided
  if !isnothing(seed)
    Random.seed!(seed)
  end

  # Create indices
  indices = collect(1:n)
  if shuffle
    shuffle!(indices)
  end

  # Calculate fold sizes
  fold_sizes = fill(n ÷ k, k)
  remainder = n % k
  for i = 1:remainder
    fold_sizes[i] += 1
  end

  # Create folds
  test_indices = Vector{Vector{Int}}(undef, k)
  train_indices = Vector{Vector{Int}}(undef, k)

  start_idx = 1
  for i = 1:k
    end_idx = start_idx + fold_sizes[i] - 1

    # Test indices for this fold
    test_indices[i] = indices[start_idx:end_idx]

    # Train indices for this fold (all other indices)
    train_indices[i] = vcat(indices[1:start_idx-1], indices[end_idx+1:end])

    start_idx = end_idx + 1
  end

  return KFoldSplit(train_indices, test_indices, fold_sizes, shuffle, seed)
end

function get_fold(split::KFoldSplit, fold::Int)
  if fold < 1 || fold > length(split.fold_sizes)
    throw(ArgumentError("Invalid fold number"))
  end
  return split.train_indices[fold], split.test_indices[fold]
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