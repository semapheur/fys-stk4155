using Distributed

function setup_workers(n_workers::Int)
  current_workers = nworkers()
  if current_workers < n_workers
    addprocs(n_workers - current_workers)
  end

  @everywhere begin
    using Statistics
    include("./hypertuning.jl")
    include("./preprocessing.jl")
  end
end

# Not working
function parallel_grid_search(
  x::Matrix{Float64},
  y::Matrix{Float64},
  hidden_layer_configs::Vector{Vector{Int}},
  learning_rates::Vector{Float64},
  l2_lambdas::Vector{Float64},
  model_functions::NetworkFunctions;
  k_folds::Int=5,
  epochs::Int=100,
  batch_size::Int=32,
  verbose::Bool=false,
  n_workers::Int=max(1, Sys.CPU_THREADS - 1),
)
  setup_workers(n_workers)

  # Generate all parameter combinations
  param_combinations = [
    ([size(x, 2), layer_sizes..., size(y, 2)], lr, l2_lambda) for
    layer_sizes in hidden_layer_configs for lr in learning_rates for
    l2_lambda in l2_lambdas
  ]

  # Create a channel for results
  results_channel = RemoteChannel(() -> Channel{Tuple}(length(param_combinations)))

  # Function to evaluate one parameter combination
  @everywhere function evaluate_params(
    params,
    x,
    y,
    model_functions,
    k_folds,
    epochs,
    batch_size,
  )
    mse_scores, r2_scores, training_times =
      evaluate_network(params, x, y, model_functions, k_folds, epochs, batch_size)
    return (
      params,
      mean(mse_scores),
      std(mse_scores),
      mean(r2_scores),
      std(r2_scores),
      mean(training_times),
      std(training_times),
    )
  end

  # Distribute work across workers
  @sync begin
    for params in param_combinations
      @async begin
        result = remotecall_fetch(
          evaluate_params,
          workers()[rand(1:nworkers())],
          params,
          x,
          y,
          model_functions,
          k_folds,
          epochs,
          batch_size,
        )
        put!(results_channel, result)

        if verbose
          print_score(result...)
        end
      end
    end
  end

  # Collect results
  results = []
  best_score = Inf
  best_params = nothing

  for _ = 1:length(param_combinations)
    result = take!(results_channel)
    push!(results, result)

    if result[2] < best_score
      best_score = result[2]
      best_params = result[1]
    end
  end

  return best_params, results
end
