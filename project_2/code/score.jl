
"""
Calculates the R-squared value of the prediction.

# Parameters
- `target::Union{Matrix{Float64},Vector{Float64}}`: The target values.
- `prediction::Union{Matrix{Float64},Vector{Float64}}`: The predicted values.

# Returns
- `r2::Float64`: The R-squared value.
"""
function r_squared(
  target::Union{Matrix{Float64},Vector{Float64}},
  prediction::Union{Matrix{Float64},Vector{Float64}},
)::Float64
  if target isa Matrix && !matrix_is_vector(target)
    throw(
      DimensionMismatch(
        "Target must be a vector or Nx1/1xN matrix, but has size $dim_target",
      ),
    )
  end

  if prediction isa Matrix && !matrix_is_vector(target)
    throw(
      DimensionMismatch(
        "Prediction must be a vector or Nx1/1xN matrix, but has size $dim_prediction",
      ),
    )
  end

  rss = sum((target - prediction) .^ 2) # residual sum of squares
  tss = sum((target .- mean(target)) .^ 2) # total sum of squares

  return 1 - rss / tss
end

"""
Calculates the accuracy score for binary classification predictions.

# Parameters
- `prediction::Vector{Float64}`: The predicted probabilities or scores.
- `target::Vector{Float64}`: The true binary labels.
- `threshold::Float64`: The threshold to convert probabilities to binary labels. Defaults to 0.5.

# Returns
- `accuracy::Float64`: The accuracy score, representing the proportion of correct predictions.
"""
function accuracy_score(
  prediction::Union{Matrix{Float64},Vector{Float64}},
  target::Union{Matrix{Float64},Vector{Float64}},
  threshold::Float64=0.5,
)::Float64
  if target isa Matrix && !matrix_is_vector(target)
    throw(
      DimensionMismatch(
        "Target must be a vector or Nx1/1xN matrix, but has size $dim_target",
      ),
    )
  end

  if prediction isa Matrix && !matrix_is_vector(target)
    throw(
      DimensionMismatch(
        "Prediction must be a vector or Nx1/1xN matrix, but has size $dim_prediction",
      ),
    )
  end

  prediction = prediction .> threshold
  mean(prediction .== target)
end

function matrix_is_vector(x::Matrix{Float64})::Bool
  return size(x, 1) == 1 || size(x, 2) == 1
end
