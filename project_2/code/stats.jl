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
  if target isa Matrix
    dim_target = size(target)
    if dim_target[1] != 1 && dim_target[2] != 1
      throw(
        DimensionMismatch(
          "Target must be a vector or Nx1/1xN matrix, but has size $dim_target",
        ),
      )
    end
  end

  if prediction isa Matrix
    dim_prediction = size(prediction)
    if dim_prediction[1] != 1 && dim_prediction[2] != 1
      throw(
        DimensionMismatch(
          "Prediction must be a vector or Nx1/1xN matrix, but has size $dim_prediction",
        ),
      )
    end
  end

  rss = sum((target - prediction) .^ 2) # residual sum of squares
  tss = sum((target .- mean(target)) .^ 2) # total sum of squares

  return 1 - rss / tss
end

function r_squared(
  target::Union{Matrix{Float64},Vector{Float64}},
  prediction::Union{Matrix{Float64},Vector{Float64}},
)::Float64
  # Convert matrices to vectors if needed
  target_vec = if target isa Matrix
    dim_target = size(target)
    if dim_target[1] == 1 || dim_target[2] == 1
      vec(target)
    else
      throw(
        DimensionMismatch(
          "Target must be a vector or Nx1/1xN matrix, but has size $dim_target",
        ),
      )
    end
  else
    target
  end

  pred_vec = if prediction isa Matrix
    dim_prediction = size(prediction)
    if dim_prediction[1] == 1 || dim_prediction[2] == 1
      vec(prediction)
    else
      throw(
        DimensionMismatch(
          "Prediction must be a vector or Nx1/1xN matrix, but has size $dim_prediction",
        ),
      )
    end
  else
    prediction
  end

  # Check if lengths match
  if length(target_vec) != length(pred_vec)
    throw(
      DimensionMismatch(
        "Target and prediction must have the same length, got $(length(target_vec)) and $(length(pred_vec))",
      ),
    )
  end

  # Calculate R²
  rss = sum((target_vec - pred_vec) .^ 2)  # residual sum of squares
  tss = sum((target_vec .- mean(target_vec)) .^ 2)  # total sum of squares

  return 1 - rss / tss
end
