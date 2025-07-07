include("../code/regression.jl")

function evaluate_ols(
  samples::Int,
  noise_amplitude::Float64,
  train_ratio::Float64,
  degrees::Int,
  random_inputs::Bool=false,
)
  x, y = franke_training_data(samples, noise_amplitude, random_inputs)
  x_scaled = standardize_data(x)

  degree_range = 0:degrees

  mse = Vector{Float64}(undef, degrees + 1)
  R2 = Vector{Float64}(undef, degrees + 1)
  coefficients = Vector{Vector{Float64}}()
  for p in degree_range
    x_poly = polynomial_design_matrix(x_scaled, p)
    x_train, x_test, y_train, y_test = train_test_split(x_poly, y, train_ratio)
    β = ordinary_least_squares(x_train, y_train)
    y_pred = x_test * β
    mse[p+1] = mean((y_test - y_pred) .^ 2)
    R2[p+1] = r_squared(y_test, y_pred)
    push!(coefficients, β)
  end

  return mse, R2, coefficients
end

# Helper function for evaluating ridge regression
function evaluate_ridge(
  samples::Int,
  noise_amplitude::Float64,
  train_ratio::Float64,
  degrees::Int,
  λ::Float64,
  random_inputs::Bool=false,
)
  x, y = franke_training_data(samples, noise_amplitude, random_inputs)
  x_scaled = standardize_data(x)

  degree_range = 0:degrees

  mse = Vector{Float64}(undef, degrees + 1)
  R2 = Vector{Float64}(undef, degrees + 1)
  coefficients = Vector{Vector{Float64}}()
  for p in degree_range
    x_poly = polynomial_design_matrix(x_scaled, p)
    x_train, x_test, y_train, y_test = train_test_split(x_poly, y, train_ratio)
    β = ridge_regression(x_train, y_train, λ)
    y_pred = x_test * β
    mse[p+1] = mean((y_test - y_pred) .^ 2)
    R2[p+1] = r_squared(y_test, y_pred)
    push!(coefficients, β)
  end

  return mse, R2, coefficients
end

# Helper function for evaluating LASSO regression
function evaluate_lasso(
  samples::Int,
  noise_amplitude::Float64,
  train_ratio::Float64,
  degrees::Int,
  λ::Float64,
  random_inputs::Bool=false,
)
  x, y = franke_training_data(samples, noise_amplitude, random_inputs)
  x_scaled = standardize_data(x)

  mse = Vector{Float64}(undef, degrees + 1)
  R2 = Vector{Float64}(undef, degrees + 1)
  coefficients = Vector{Vector{Float64}}()
  for p = 0:degrees
    x_poly = polynomial_design_matrix(x_scaled, p)
    x_train, x_test, y_train, y_test = train_test_split(x_poly, y, train_ratio)
    β = lasso_regression(x_train, y_train, λ)
    y_pred = x_test * β
    mse[p+1] = mean((y_test - y_pred) .^ 2)
    R2[p+1] = r_squared(y_test, y_pred)
    push!(coefficients, β)
  end

  return mse, R2, coefficients
end

function bootstrap_ols(
  trails::Int,
  samples::Int,
  noise_amplitude::Float64,
  train_ratio::Float64,
  degrees::Int,
  random_inputs::Bool=false,
)
  x, y = franke_training_data(samples, noise_amplitude, random_inputs)
  x_scaled = standardize_data(x)

  mse = zeros(degrees + 1)
  R2 = zeros(degrees + 1)
  coefficients = Vector{Vector{Float64}}()

  for p = 0:degrees
    x_poly = polynomial_design_matrix(x_scaled, p)
    coefficients_ = zeros(size(x_poly, 2))

    for _ = 1:trails
      x_train, x_test, y_train, y_test = train_test_split(x_poly, y, train_ratio)

      β = ordinary_least_squares(x_train, y_train)
      y_pred = x_test * β
      mse[p+1] += mean((y_test - y_pred) .^ 2)
      R2[p+1] += r_squared(y_test, y_pred)
      coefficients_ += β
    end

    push!(coefficients, coefficients_ ./ trails)
  end

  mse ./= trails
  R2 ./= trails

  return mse, R2, coefficients
end

function bootstrap_ridge(
  trails::Int,
  samples::Int,
  noise_amplitude::Float64,
  train_ratio::Float64,
  degrees::Int,
  λ::Float64,
  random_inputs::Bool=false,
)
  x, y = franke_training_data(samples, noise_amplitude, random_inputs)
  x_scaled = standardize_data(x)

  mse = zeros(degrees + 1)
  R2 = zeros(degrees + 1)
  coefficients = Vector{Vector{Float64}}()

  for p = 0:degrees
    x_poly = polynomial_design_matrix(x_scaled, p)
    coefficients_ = zeros(size(x_poly, 2))

    for _ = 1:trails
      x_train, x_test, y_train, y_test = train_test_split(x_poly, y, train_ratio)

      β = ridge_regression(x_train, y_train, λ)
      y_pred = x_test * β
      mse[p+1] += mean((y_test - y_pred) .^ 2)
      R2[p+1] += r_squared(y_test, y_pred)
      coefficients_ += β
    end

    push!(coefficients, coefficients_ ./ trails)
  end

  mse ./= trails
  R2 ./= trails

  return mse, R2, coefficients
end

function bootstrap_lasso(
  trails::Int,
  samples::Int,
  noise_amplitude::Float64,
  train_ratio::Float64,
  degrees::Int,
  λ::Float64,
  random_inputs::Bool=false,
)
  x, y = franke_training_data(samples, noise_amplitude, random_inputs)
  x_scaled = standardize_data(x)

  mse = zeros(degrees + 1)
  R2 = zeros(degrees + 1)
  coefficients = Vector{Vector{Float64}}()

  for p = 0:degrees
    x_poly = polynomial_design_matrix(x_scaled, p)
    coefficients_ = zeros(size(x_poly, 2))

    for _ = 1:trails
      x_train, x_test, y_train, y_test = train_test_split(x_poly, y, train_ratio)

      β = lasso_regression(x_train, y_train, λ)
      y_pred = x_test * β
      mse[p+1] += mean((y_test - y_pred) .^ 2)
      R2[p+1] += r_squared(y_test, y_pred)
      coefficients_ += β
    end

    push!(coefficients, coefficients_ ./ trails)
  end

  mse ./= trails
  R2 ./= trails

  return mse, R2, coefficients
end

function cross_validation_ols(
  folds::Int,
  samples::Int,
  degrees::Int,
  noise_amplitude::Float64,
  aggregate::Bool=true,
  random_inputs::Bool=false,
)
  x, y = franke_training_data(samples, noise_amplitude, random_inputs)
  x_scaled = standardize_data(x)

  n = size(x, 1)
  kfolds = kfold_split(n, folds)

  mse = zeros(degrees + 1, folds)
  r2 = zeros(degrees + 1, folds)

  for p = 0:degrees
    x_poly = polynomial_design_matrix(x_scaled, p)

    for (fold_idx, test_indices) in enumerate(kfolds)
      # Create train indices
      train_indices = setdiff(1:n, test_indices)

      # Split data into train and test
      x_train = x_poly[train_indices, :]
      y_train = y[train_indices]
      x_test = x_poly[test_indices, :]
      y_test = y[test_indices]

      # Train and test model 
      β = ordinary_least_squares(x_train, y_train)
      y_pred = x_test * β

      mse[p+1, fold_idx] = mean((y_test - y_pred) .^ 2)
      r2[p+1, fold_idx] = r_squared(y_test, y_pred)
    end
  end

  if aggregate
    return mean(mse, dims=2), mean(r2, dims=2)
  end

  return mse, r2
end

function cross_validation_ridge(
  folds::Int,
  samples::Int,
  degrees::Int,
  λ::Float64,
  noise_amplitude::Float64,
  aggregate::Bool=true,
  random_inputs::Bool=false,
)
  x, y = franke_training_data(samples, noise_amplitude, random_inputs)
  x_scaled = standardize_data(x)

  n = size(x, 1)
  kfolds = kfold_split(n, folds)

  mse = zeros(degrees + 1, folds)
  r2 = zeros(degrees + 1, folds)

  for p = 0:degrees
    x_poly = polynomial_design_matrix(x_scaled, p)

    for (fold_idx, test_indices) in enumerate(kfolds)
      # Create train indices
      train_indices = setdiff(1:n, test_indices)

      # Split data into train and test
      x_train = x_poly[train_indices, :]
      y_train = y[train_indices]
      x_test = x_poly[test_indices, :]
      y_test = y[test_indices]

      # Train and test model 
      β = ridge_regression(x_train, y_train, λ)
      y_pred = x_test * β

      mse[p+1, fold_idx] = mean((y_test - y_pred) .^ 2)
      r2[p+1, fold_idx] = r_squared(y_test, y_pred)
    end
  end

  if aggregate
    return mean(mse, dims=2), mean(r2, dims=2)
  end

  return mse, r2
end

function cross_validation_lasso(
  folds::Int,
  samples::Int,
  degrees::Int,
  λ::Float64,
  noise_amplitude::Float64,
  aggregate::Bool=true,
  random_inputs::Bool=false,
)
  x, y = franke_training_data(samples, noise_amplitude, random_inputs)
  x_scaled = standardize_data(x)

  n = size(x, 1)
  kfolds = kfold_split(n, folds)

  mse = zeros(degrees + 1, folds)
  r2 = zeros(degrees + 1, folds)

  for p = 0:degrees
    x_poly = polynomial_design_matrix(x_scaled, p)

    for (fold_idx, test_indices) in enumerate(kfolds)
      # Create train indices
      train_indices = setdiff(1:n, test_indices)

      # Split data into train and test
      x_train = x_poly[train_indices, :]
      y_train = y[train_indices]
      x_test = x_poly[test_indices, :]
      y_test = y[test_indices]

      # Train and test model 
      β = lasso_regression(x_train, y_train, λ)
      y_pred = x_test * β

      mse[p+1, fold_idx] = mean((y_test - y_pred) .^ 2)
      r2[p+1, fold_idx] = r_squared(y_test, y_pred)
    end
  end

  if aggregate
    return mean(mse, dims=2), mean(r2, dims=2)
  end

  return mse, r2
end

function ridge_optimize_lambda(
  lambdas::Vector{Float64},
  folds::Int,
  samples::Int,
  degree::Int,
  noise_amplitude::Float64,
  random_inputs::Bool=false,
)
  x, y = franke_training_data(samples, noise_amplitude, random_inputs)
  x_scaled = standardize_data(x)
  x_poly = polynomial_design_matrix(x_scaled, degree)

  n = size(x, 1)
  kfolds = kfold_split(n, folds)

  mse = zeros(length(lambdas), folds)
  r2 = zeros(length(lambdas), folds)

  for (i, λ) in enumerate(lambdas)
    for (fold_idx, test_indices) in enumerate(kfolds)
      train_indices = setdiff(1:n, test_indices)

      x_train = x_poly[train_indices, :]
      y_train = y[train_indices]
      x_test = x_poly[test_indices, :]
      y_test = y[test_indices]

      β = ridge_regression(x_train, y_train, λ)
      y_pred = x_test * β

      mse[i, fold_idx] = mean((y_test - y_pred) .^ 2)
      r2[i, fold_idx] = r_squared(y_test, y_pred)
    end
  end

  return mse, r2
end

function lasso_optimize_lambda(
  lambdas::Vector{Float64},
  folds::Int,
  samples::Int,
  degree::Int,
  noise_amplitude::Float64,
  random_inputs::Bool=false,
)
  x, y = franke_training_data(samples, noise_amplitude, random_inputs)
  x_scaled = standardize_data(x)
  x_poly = polynomial_design_matrix(x_scaled, degree)

  n = size(x, 1)
  kfolds = kfold_split(n, folds)

  mse = zeros(length(lambdas), folds)
  r2 = zeros(length(lambdas), folds)

  for (i, λ) in enumerate(lambdas)
    for (fold_idx, test_indices) in enumerate(kfolds)
      train_indices = setdiff(1:n, test_indices)

      x_train = x_poly[train_indices, :]
      y_train = y[train_indices]
      x_test = x_poly[test_indices, :]
      y_test = y[test_indices]

      β = lasso_regression(x_train, y_train, λ)
      y_pred = x_test * β

      mse[i, fold_idx] = mean((y_test - y_pred) .^ 2)
      r2[i, fold_idx] = r_squared(y_test, y_pred)
    end
  end

  return mse, r2
end

function plot_mse_r2(
  degrees::Int,
  mse::Vector{Float64},
  R2::Vector{Float64},
  title::String="",
)
  plot(
    0:degrees,
    mse,
    marker=:circle,
    label=L"\mathrm{MSE}",
    legend=(0.5, 0.6),
    title=title,
    xlabel="Polynomial degree",
    ylabel=L"\mathrm{MSE}",
  )
  plot!(
    twinx(),
    0:degrees,
    R2,
    marker=:diamond,
    color=:red,
    label=L"R^2",
    legend=(0.5, 0.4),
    ylabel=L"R^2",
  )
end

function plot_coefficients(
  coefficients::Vector{Vector{Float64}},
  padding_scale::Float64=0.1,
  title::String="",
  filename::String="",
)
  num_coefficients = [length(c) for c in coefficients]

  labels = permutedims(["p=$i" for i = 0:degrees])
  texts = vec([text(l, :bottom, :middle, 10) for l in labels])

  y_min = minimum(minimum(coefficients))
  y_max = maximum(maximum(coefficients))

  padding = padding_scale * (y_max - y_min)
  y_min -= padding
  y_max += padding

  vline(num_coefficients, label="")
  annotate!(num_coefficients, y_min + 0.1 * padding, texts)
  xticks = (1:num_coefficients[end], 0:num_coefficients[end]-1)
  plot!(
    coefficients,
    marker=:circle,
    labels=labels,
    xlabel=L"\beta_i",
    xticks=xticks,
    ylims=(y_min, y_max),
    title=title,
  )

  if !isempty(filename)
    savefig(filename)
  end
end

function create_markdown_table(variables::Int, degree::Int)::String
  # Generate the combinations
  combos = polynomial_combinations(variables, degree)

  # Create header row
  header = ["\$\\beta_{$(i)}\$" for i = 0:length(combos)-1]

  # Create separator row
  separator = ["---" for _ = 1:length(header)]

  # Create rows for the combinations
  rows = []
  for combo in combos
    row = [
      "\$" *
      join(["x_{$(i)}^{$(combo[i])}" for i = 1:variables if combo[i] > 0], " ") *
      "\$",
    ]
    push!(rows, row)
  end

  # Convert rows to markdown format
  markdown_table = "| " * join(header, " | ") * " |\n"
  markdown_table *= "| " * join(separator, " | ") * " |\n"

  for row in rows
    markdown_table *= "| " * join(row, " | ") * " "
  end

  markdown_table *= "|"

  return markdown_table
end