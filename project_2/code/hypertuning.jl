include("./activation.jl")
include("./learning_rate.jl")
include("./neural_network.jl")
include("./preprocessing.jl")
include("./stats.jl")

using BenchmarkTools
using CategoricalArrays
using DataFrames
using DataStructures
using Flux
using LaTeXStrings
using Plots
using Printf
using Statistics
using StatsPlots

struct Hyperparams
  layer_sizes::Vector{Int}
  lr::Float64
  l2_lambda::Float64
end

struct MyTuneScore
  params::Hyperparams
  mse_mean_std::Tuple{Float64,Float64}
  r2_mean_std::Tuple{Float64,Float64}
  time_mean_std::Tuple{Float64,Float64}
end

struct MyModelScore
  model::String
  mse_mean_std::Tuple{Float64,Float64}
  r2_mean_std::Tuple{Float64,Float64}
  time_mean_std::Tuple{Float64,Float64}
end

struct NetworkFunctions
  hidden_activation::Function
  hidden_activation_prime::Function
  output_activation::Function
  output_activation_prime::Function
  cost::Function
  cost_prime::Function
end

function hidden_layer_configs(num_layers::Int, min_exp::Int, max_exp::Int)
  hidden_layer_configs = Vector{Vector{Int}}()
  power_range = min_exp:max_exp

  # Generate configurations for 1 to num_layers layers
  for n = 1:num_layers
    # Create all possible combinations of powers of 2 for n layers
    for combo in Iterators.product(fill(power_range, n)...)
      # Convert exponents to actual neuron counts
      layer_sizes = [2^exp for exp in combo]
      push!(hidden_layer_configs, layer_sizes)
    end
  end

  return unique(hidden_layer_configs)
end

function print_score(tune_score::MyTuneScore)
  text = """
  Layers: $(tune_score.params.layer_sizes), LR: $(@sprintf("%.2e", tune_score.params.lr)), L2: $(@sprintf("%.2e", tune_score.params.l2_lambda))
  MSE: $(@sprintf("%.2e", tune_score.mse_mean_std[1])) (±$(@sprintf("%.2e", tune_score.mse_mean_std[2])))
  R2: $(@sprintf("%.2e", tune_score.r2_mean_std[1])) (±$(@sprintf("%.2e", tune_score.r2_mean_std[2])))
  Time: $(@sprintf("%.2e", tune_score.time_mean_std[1])) (±$(@sprintf("%.2e", tune_score.time_mean_std[2])))
  """

  println(text)
end

function evaluate_network(
  x::Matrix{Float64},
  y::Matrix{Float64},
  model::NeuralNetwork,
  k_folds::Int=5,
  n_epochs::Int=100,
  batch_size::Int=32,
)
  training_times = zeros(k_folds)
  mse_scores = zeros(k_folds)
  r2_scores = zeros(k_folds)
  split = kfold_split(size(x, 1), k_folds, true)

  for i = 1:k_folds
    train_idx, val_idx = get_fold(split, i)

    x_train = x[train_idx, :]
    y_train = y[train_idx, :]

    x_val = x[val_idx, :]
    y_val = y[val_idx, :]

    training_start = time_ns()
    _ = train_network(model, x_train, y_train, n_epochs, batch_size, false)
    training_times[i] = (time_ns() - training_start) / 1e9

    y_pred = predict(model, x_val)
    mse_scores[i] = mean((y_val - y_pred) .^ 2)
    r2_scores[i] = r_squared(y_val, y_pred)
  end

  return skipmissing(mse_scores), skipmissing(r2_scores), training_times
end

# Not working, strugling with DimensionMismatch errors that I can't figure out
function evaluate_flux(
  x::Matrix{Float64},
  y::Matrix{Float64},
  layer_sizes::Vector{Int},
  hidden_activation::Function,
  output_activation::Union{Function,Nothing},
  lr::Float64,
  l2_lambda::Float64,
  k_folds::Int,
  epochs::Int,
  batch_size::Int,
)
  function create_model(
    layer_sizes::Vector{Int},
    hidden_activation::Function,
    output_activation::Union{Function,Nothing},
  )
    layers = []

    # Create the hidden layers
    for i = 1:(length(layer_sizes)-2)
      push!(layers, Dense(layer_sizes[i], layer_sizes[i+1], hidden_activation))
    end

    # Create the output layer
    if output_activation === nothing
      push!(layers, Dense(layer_sizes[end-1], layer_sizes[end]))
    else
      push!(layers, Dense(layer_sizes[end-1], layer_sizes[end], output_activation))
    end

    # Create the Chain with the layers
    return Chain(layers...)
  end

  training_times = zeros(k_folds)
  mse_scores = zeros(k_folds)
  r2_scores = zeros(k_folds)
  split = kfold_split(size(x, 1), k_folds, true)

  x = reshape(x, size(x, 2), size(x, 1))
  y = reshape(y, size(y, 2), size(y, 1))

  for i = 1:k_folds
    train_idx, val_idx = get_fold(split, i)

    x_train = x[:, train_idx]
    y_train = y[:, train_idx]

    x_val = x[:, val_idx]
    y_val = y[:, val_idx]

    # Define the model
    model = create_model(layer_sizes, hidden_activation, output_activation)

    # Define the loss function with L2 regularization
    loss(x, y) =
      Flux.Losses.mse(model(x), y) +
      l2_lambda * sum(sum(abs2, w) for w in Flux.params(model))
    #function loss(x, y)
    #  l2_reg = sum(sum(abs2, params(layer)) for layer in model)
    #  return Flux.Losses.mse(model(x), y) + l2_lambda * l2_reg
    #end

    train_data = Flux.DataLoader((x_train, y_train), batchsize=batch_size, shuffle=true)

    # Set up the optimizer
    optimizer = Descent(lr)

    # Training loop
    training_start = time_ns()
    for _ = 1:epochs
      for (x_batch, y_batch) in train_data
        Flux.train!(loss, Flux.params(model), [(x_batch, y_batch)], optimizer)
      end
    end
    training_times[i] = (time_ns() - training_start) / 1e9

    # Validate the model
    y_pred = model(x_val)

    mse_scores[i] = Flux.Losses.mse(y_pred, y_val)
    r2_scores[i] = r_squared(y_val, Float64.(y_pred))
  end

  return mse_scores, r2_scores, training_times
end

function network_scores(
  model::String,
  mse_scores::Union{Base.SkipMissing{Vector{Float64}},Vector{Float64}},
  r2_scores::Union{Base.SkipMissing{Vector{Float64}},Vector{Float64}},
  times::Vector{Float64},
)
  # Calculate means and standard deviations
  mean_mse = mean(mse_scores)
  std_mse = std(mse_scores)

  mean_r2 = mean(r2_scores)
  std_r2 = std(r2_scores)

  mean_time = mean(times)
  std_time = std(times)

  return MyModelScore(model, (mean_mse, std_mse), (mean_r2, std_r2), (mean_time, std_time))
end

function grid_search(
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
)::Tuple{Int,Vector{MyTuneScore}}
  best_score = Inf
  best_idx = nothing

  results = Vector{MyTuneScore}()

  for layer_sizes in hidden_layer_configs
    layer_sizes = [size(x, 2), layer_sizes..., size(y, 2)]
    for lr in learning_rates
      for l2_lambda in l2_lambdas
        model = initialize_network(
          layer_sizes,
          model_functions.hidden_activation,
          model_functions.hidden_activation_prime,
          model_functions.outputactivation,
          model_functions.outputactivation_prime,
          model_functions.cost,
          model_functions.cost_prime,
          ConstantLR(lr),
          l2_lambda,
        )

        mse_scores, r2_scores, time_scores =
          evaluate_network(x, y, model, k_folds, epochs, batch_size)
        mean_score = (mean(mse_scores), std(mse_scores))
        r2_score = (mean(r2_scores), std(r2_scores))
        time_score = (mean(time_scores), std(time_scores))
        scores = MyTuneScore(params, mean_score, r2_score, time_score)
        push!(results, scores)

        if verbose
          print_score(scores)
        end

        if mean_score[1] < best_score
          best_score = mean_score[1]
          best_idx = length(results)
        end
      end
    end
  end

  return best_idx, results
end

function random_search(
  x::Matrix{Float64},
  y::Matrix{Float64},
  model_functions::NetworkFunctions;
  trials::Int=50,
  max_hidden_layers::Int=3,
  min_max_neurons_power::Tuple{Int,Int}=(2, 7),
  min_max_lr::Tuple{Float64,Float64}=(0.0001, 0.1),
  min_max_l2_lambda::Tuple{Float64,Float64}=(1e-6, 0.1),
  k_folds::Int=5,
  epochs::Int=100,
  batch_size::Int=32,
  verbose::Bool=false,
)::Tuple{Int,Vector{MyTuneScore}}
  best_score = Inf
  best_idx = nothing

  results = Vector{MyTuneScore}()

  for _ = 1:trials
    n_hidden_layers = rand(1:max_hidden_layers)
    hidden_sizes = [
      2^round(
        Int,
        exp(uniform(log(min_max_neurons_power[1]), log(min_max_neurons_power[2]))),
      ) for _ = 1:n_hidden_layers
    ]
    layer_sizes = [size(x, 2), hidden_sizes..., size(y, 2)]
    lr = exp(uniform(log(min_max_lr[1]), log(min_max_lr[2])))
    l2_lambda = exp(uniform(log(min_max_l2_lambda[1]), log(min_max_l2_lambda[2])))

    model = initialize_network(
      layer_sizes,
      model_functions.hidden_activation,
      model_functions.hidden_activation_prime,
      model_functions.outputactivation,
      model_functions.outputactivation_prime,
      model_functions.cost,
      model_functions.cost_prime,
      ConstantLR(lr),
      l2_lambda,
    )
    mse_scores, r2_scores, time_scores =
      evaluate_network(x, y, model, k_folds, epochs, batch_size)
    mean_score = (mean(mse_scores), std(mse_scores))
    r2_score = (mean(r2_scores), std(r2_scores))
    time_score = (mean(time_scores), std(time_scores))
    scores = MyTuneScore(params, mean_score, r2_score, time_score)
    push!(results, scores)

    if mean_score[1] < best_score
      best_score = mean_score[1]
      best_idx = length(results)
      if verbose
        print_score(scores)
      end
    end
  end

  return best_idx, results
end

function uniform(a, b)
  return a + (b - a) * rand()
end

function visualize_architecture_tuning(results::Vector{MyTuneScore}; save_plots=true)
  results = sort(results, by=r -> (length(r.params.layer_sizes), sum(r.params.layer_sizes)))

  # Convert results to DataFrame for easier manipulation
  df = DataFrame(
    hidden_layers=[join(r.params.layer_sizes[2:end-1], "-") for r in results],
    learning_rate=[r.params.lr for r in results],
    l2_lambda=[r.params.l2_lambda for r in results],
    mse=[r.mse_mean_std[1] for r in results],
    mse_std=[r.mse_mean_std[2] for r in results],
    r2=[r.r2_mean_std[1] for r in results],
    r2_std=[r.r2_mean_std[2] for r in results],
    time=[r.time_mean_std[1] for r in results],
    time_std=[r.time_mean_std[2] for r in results],
  )

  df.layers = map(x -> length(split(x, "-")), df.hidden_layers)
  plots = []

  headers = OrderedDict{String,Union{Symbol,Tuple{Symbol,Symbol}}}(
    "Layers" => :hidden_layers,
    "η" => :learning_rate,
    "λ" => :l2_lambda,
    "MSE" => (:mse, :mse_std),
    "R^2" => (:r2, :r2_std),
    "Time [s]" => (:time, :time_std),
  )

  sorted_df = sort(df, :mse)
  markdown_table = dataframe_to_markdown_table(first(sorted_df, 5), headers)

  open("hyperparameter_tuning_table.txt", "w") do io
    write(io, markdown_table)
  end

  best_results_text = "Top 5 Configurations Overall\n"
  for i = 1:min(5, size(sorted_df, 1))
    best_results_text =
      best_results_text * """
$i. MSE: $(@sprintf("%.2e", sorted_df.mse[i])) ± $(@sprintf("%.2e", sorted_df.mse_std[i]))
    R2: $(@sprintf("%.2e", sorted_df.r2[i])) ± $(@sprintf("%.2e", sorted_df.r2_std[i]))
    Time: $(@sprintf("%.2e", sorted_df.time[i])) ± $(@sprintf("%.2e", sorted_df.time_std[i]))
    Layers: $(sorted_df.hidden_layers[i])
    LR: $(@sprintf("%.2e", sorted_df.learning_rate[i]))
    L2: $(@sprintf("%.2e", sorted_df.l2_lambda[i]))

"""
  end

  for l in unique(df.layers)
    filtered_df = df[df.layers.==l, :]
    filtered_df = sort(filtered_df, :mse)

    best_results_text =
      best_results_text * "Top 5 Configurations for $l $(l == 1 ? "Layer" : " Layers")\n"

    for i = 1:min(5, size(filtered_df, 1))
      best_results_text =
        best_results_text * """
$i. MSE: $(@sprintf("%.2e", filtered_df.mse[i])) ± $(@sprintf("%.2e", filtered_df.mse_std[i]))
    R2: $(@sprintf("%.2e", filtered_df.r2[i])) ± $(@sprintf("%.2e", filtered_df.r2_std[i]))
    Time: $(@sprintf("%.2e", filtered_df.time[i])) ± $(@sprintf("%.2e", filtered_df.time_std[i]))
    Layers: $(filtered_df.hidden_layers[i])
    LR: $(@sprintf("%.2e", filtered_df.learning_rate[i]))
    L2: $(@sprintf("%.2e", filtered_df.l2_lambda[i]))
"""
    end

    filtered_df.hidden_layers = pad_layers(filtered_df.hidden_layers)
    filtered_df = sort(filtered_df, :hidden_layers)
    tick_labels = sort(pad_layers(unique(filtered_df.hidden_layers)))

    if length(tick_labels) > 40
      step = ceil(Int, length(tick_labels) / 40)
      xticks = (0.5:step:(length(tick_labels)-0.5), tick_labels[1:step:end])
    else
      xticks = (0.5:1:(length(tick_labels)-0.5), tick_labels)
    end

    plot_ = @df filtered_df boxplot(
      :hidden_layers,
      :mse,
      fillalpha=0.75,
      label="",
      title="$(l) Hidden $(l == 1 ? "Layer" : " Layers")",
      xlabel="Hidden Layer Sizes",
      ylabel="Mean Squared Error",
      xticks=xticks,
      x_rotation=90,
      #color=:blue,
      whisker_width=0.5,
      size=(800, 500),
    )

    push!(plots, plot_)
  end

  layout = @layout [grid(length(plots), 1)]
  final_plot = plot(
    plots...,
    layout=layout,
    plot_title="MSE of Layer Architectures",
    size=(800, 1200),
    margin=5Plots.mm,
  )

  if save_plots
    savefig(final_plot, "hyperparameter_tuning_architecture.pdf")
    # Save best results to text file
    open("best_configurations.txt", "w") do io
      write(io, best_results_text)
    end
  end

  return final_plot, best_results_text
end

function pad_layers(hidden_layers::Vector{String})
  # Extract all numbers and determine the maximum length
  all_numbers = [parse(Int, num) for arch in hidden_layers for num in split(arch, "-")]
  max_length = maximum(map(length, string.(all_numbers)))

  # Pad each architecture
  padded_layers =
    [join(lpad.(string.(split(arch, "-")), max_length), "-") for arch in hidden_layers]

  return padded_layers
end

function dataframe_to_markdown_table(
  df::DataFrame,
  headers::OrderedDict{String,Union{Symbol,Tuple{Symbol,Symbol}}},
)
  # Create the header of the Markdown table
  header_row = "| " * join(keys(headers), " | ") * " |"
  separator_row = "| " * join([":-:" for _ = 1:length(headers)], " | ") * " |"

  # Initialize the Markdown table with the header
  markdown_table = [header_row, separator_row]

  # Iterate over each row in the DataFrame to format it
  for row in eachrow(df)
    markdown_row = "|"

    for (_, column) in headers
      if column isa Symbol
        value = row[column]
        formatted_value = value isa Number ? "\\num{$(@sprintf("%.2e", value))}" : value

      elseif column isa Tuple{Symbol,Symbol}
        value1, value2 = row[column[1]], row[column[2]]
        formatted_value = "\\num{$(@sprintf("%.2e", value1))} \\pm \\num{$(@sprintf("%.2e", value2))}"
      end

      markdown_row = markdown_row * " $formatted_value |"
    end

    # Append the formatted row to the Markdown table
    push!(markdown_table, markdown_row)
  end

  # Join all parts into a single string
  return join(markdown_table, "\n")
end