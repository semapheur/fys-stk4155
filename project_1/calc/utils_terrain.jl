include("../code/regression.jl")

function cross_validation(
  x::Matrix{Float64},
  y::Vector{Float64},
  folds::Int,
  degrees::Int,
  regression_model::Function,
  λ::Float64=0.0,
  aggregate::Bool=true,
)
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
      β =
        (λ == 0.0) ? ordinary_least_squares(x_train, y_train) :
        regression_model(x_train, y_train, λ)
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

function bootstrap(
  x::Matrix{Float64},
  y::Vector{Float64},
  trials::Int,
  train_ratio::Float64,
  degrees::Int,
  regression_model::Function,
  λ::Float64=0.0,
  aggregate::Bool=true,
)
  mse = zeros(degrees + 1, trials)
  r2 = zeros(degrees + 1, trials)

  for p = 0:degrees
    x_poly = polynomial_design_matrix(x, p)

    for t = 1:trials
      x_train, x_test, y_train, y_test = train_test_split(x_poly, y, train_ratio)

      # Train and test model 
      β =
        (λ == 0.0) ? ordinary_least_squares(x_train, y_train) :
        regression_model(x_train, y_train, λ)
      y_pred = x_test * β

      mse[p+1, t] = mean((y_test - y_pred) .^ 2)
      r2[p+1, t] = r_squared(y_test, y_pred)
    end
  end

  if aggregate
    return mean(mse, dims=2), mean(r2, dims=2)
  end

  return mse, r2
end

function bias_variance(
  x::Matrix{Float64},
  y::Vector{Float64},
  trials::Int,
  train_ratio::Float64,
  degrees::Int,
  regression_model::Function,
  λ::Float64=0.0,
)
  test_error = zeros(degrees + 1)
  train_error = zeros(degrees + 1)

  for p = 0:degrees
    x_poly = polynomial_design_matrix(x, p)

    for _ = 1:trials
      x_train, x_test, y_train, y_test = train_test_split(x_poly, y, train_ratio)
      β =
        (λ == 0.0) ? ordinary_least_squares(x_train, y_train) :
        regression_model(x_train, y_train, λ)
      y_pred_test = x_test * β
      y_pred_train = x_train * β
      test_error[p+1] += mean((y_test - y_pred_test) .^ 2)
      train_error[p+1] += mean((y_train - y_pred_train) .^ 2)
    end
  end

  train_error ./= trials
  test_error ./= trials

  return test_error, train_error
end