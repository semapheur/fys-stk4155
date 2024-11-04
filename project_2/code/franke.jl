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

"""
function franke_training_data(
  n::Int,
  noise_amplitude::Float64=0.1,
  random_inputs::Bool=false,
)::Tuple{Matrix{Float64},Vector{Float64}}
  if random_inputs
    X = rand(n, 2)
  else
    # Create grid
    x1_range = range(0.0, 1.0, length=Int(floor(sqrt(n))))
    x2_range = range(0.0, 1.0, length=Int(floor(sqrt(n))))

    X = [(x1, x2) for x1 in x1_range, x2 in x2_range]
    X = reduce(hcat, X)'
  end

  y = [franke(X[i, 1], X[i, 2]) for i in axes(X, 1)]

  if noise_amplitude != 0.0
    y .+= noise_amplitude * randn(n, n)
  end

  return X, y
end
"""

function franke_training_data(
  n::Int,
  noise_amplitude::Float64=0.1,
  random_inputs::Bool=false,
)::Tuple{Matrix{Float64},Vector{Float64}}
  if random_inputs
    x_1 = rand(n)
    x_2 = rand(n)
  else
    n = Int(floor(sqrt(n)))
    x_1 = range(0.0, 1.0, length=n)
    x_2 = range(0.0, 1.0, length=n)
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