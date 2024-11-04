include("./learning_rate.jl")

using Random

"""
Layer(weights, biases, activation, activation_derivative)

A struct representing a layer in a neural network.

# Fields
- `weights::Matrix{Float64}`: A matrix of weights for the layer.
- `biases::Vector{Float64}`: A vector of biases for the layer.
- `activation::Function`: The activation function for the layer.
- `activation_prime::Function`: The derivative of the activation function for the layer.
"""
struct Layer
  weights::Matrix{Float64}
  biases::Vector{Float64}
  activation::Function
  activation_prime::Function
end

"""
NeuralNetwork

A struct representing a neural network.

# Fields
- `layers::Vector{Layer}`: A vector of layers in the network.
- `cost::Function`: The cost function used to evaluate the network.
- `cost_prime::Function`: The derivative of the cost function used to evaluate the network.
- `lr_scheduler::LearningRateScheduler`: The learning rate scheduler used to update the network.
- `l2_lambda::Float64`: The regularization strength for L2 regularization.
"""
struct NeuralNetwork
  layers::Vector{Layer}
  cost::Function
  cost_prime::Function
  lr_scheduler::LearningRateScheduler
  l2_lambda::Float64
end

"""
Calculate the initialization scale factor based on the activation function.

# Parameters
- `activation::Function`: The activation function used in the layer.
- `fan_in::Int`: The number of input units in the layer.

# Returns
- `scale::Float64`: The calculated scale factor for weight initialization.
"""
function initialization_scale(activation::Function, fan_in::Int)::Float64
  if activation == relu
    return sqrt(2.0 / fan_in)  # He initialization
  elseif activation == tanh || activation == sigmoid
    return sqrt(1.0 / fan_in)  # Xavier initialization
  else
    # Default to He initialization for unknown activation functions
    return sqrt(2.0 / fan_in)
  end
end

"""
Initialize a neural network with specified architecture and parameters. The neural network is initialized using the input convention (samples, features).

# Parameters
- `layer_sizes::Vector{Int}`: A vector specifying the number of neurons in each layer.
- `activation::Function`: The activation function for the hidden layers.
- `activation_prime::Function`: The derivative of the activation function.
- `cost::Function`: The cost function for evaluating the network.
- `cost_prime::Function`: The derivative of the cost function.
- `lr_scheduler::LearningRateScheduler`: Scheduler for adjusting the learning rate (default: ConstantLR(0.01)).
- `l2_lambda::Float64`: L2 regularization strength (default: 0.0).

# Returns
- `NeuralNetwork`: An initialized neural network with the given specifications.
"""
function initialize_network(
  layer_sizes::Vector{Int},
  hidden_activation::Function,
  hidden_activation_prime::Function,
  output_activation::Function,
  output_activation_prime::Function,
  cost::Function,
  cost_prime::Function,
  lr_scheduler::LearningRateScheduler,
  l2_lambda::Float64=0.0,
)::NeuralNetwork
  layers = Vector{Layer}()

  # Add hidden layers
  for i = 1:(length(layer_sizes)-2)
    fan_in = layer_sizes[i] # Number of inputs to the layer
    fan_out = layer_sizes[i+1] # Number of outputs from the layer

    scale = initialization_scale(hidden_activation, fan_in)

    weights = randn(fan_in, fan_out) * scale
    biases = zeros(fan_out) .+ 0.01
    push!(layers, Layer(weights, biases, hidden_activation, hidden_activation_prime))
  end

  # Add output layer
  fan_in = layer_sizes[end-1]
  fan_out = layer_sizes[end]

  scale = initialization_scale(output_activation, fan_in)

  weights = randn(fan_in, fan_out) * scale
  biases = zeros(fan_out) .+ 0.01
  push!(layers, Layer(weights, biases, output_activation, output_activation_prime))

  return NeuralNetwork(layers, cost, cost_prime, lr_scheduler, l2_lambda)
end

"""
Performs a forward pass through the neural network.

# Parameters
- `network::NeuralNetwork`: The neural network to perform the forward pass on.
- `input::Matrix{Float64}`: The input matrix to the network.

# Returns
- `Tuple{Vector{Matrix{Float64}}, Vector{Matrix{Float64}}}`: A tuple containing activations and weighted inputs at each layer.
"""
function feedforward(
  network::NeuralNetwork,
  input::Matrix{Float64},
)::Tuple{Vector{Matrix{Float64}},Vector{Matrix{Float64}}}
  activations = Vector{Matrix{Float64}}(undef, length(network.layers) + 1)
  weighted_inputs = Vector{Matrix{Float64}}(undef, length(network.layers))

  activations[1] = input

  for (i, layer) in enumerate(network.layers)
    weighted_input = activations[i] * layer.weights .+ layer.biases'
    weighted_inputs[i] = weighted_input
    activations[i+1] = layer.activation.(weighted_input)
  end

  return activations, weighted_inputs
end

"""
Performs backpropagation on a neural network to update its weights and biases.

# Parameters
- `network::NeuralNetwork`: The neural network to update.
- `input::VectorMatrix{Float64}`: The input to the network.
- `target_output::Vector{Float64}`: The target output of the network.
- `learning_rate::Float64`: The learning rate to use for the update.

# Returns
- `Float64`: The cost of the network for the given input and target output.
"""
function backpropagate(
  network::NeuralNetwork,
  input::Matrix{Float64},
  target_output::Matrix{Float64},
  learning_rate::Float64,
)
  # Forward pass
  activations, weighted_inputs = feedforward(network, input)

  # Output layer error
  error =
    network.cost_prime(activations[end], target_output) .*
    network.layers[end].activation_prime.(weighted_inputs[end])

  # Initialize gradients
  weight_gradients = [zeros(size(layer.weights)) for layer in network.layers]
  bias_gradients = [zeros(size(layer.biases)) for layer in network.layers]

  # Backpropagate errors
  for l in reverse(1:(length(network.layers)))
    bias_gradients[l] = vec(sum(error, dims=1))
    weight_gradients[l] =
      activations[l]' * error + (network.l2_lambda .* network.layers[l].weights)

    if l > 1
      error =
        (error * network.layers[l].weights') .*
        network.layers[l-1].activation_prime.(weighted_inputs[l-1])
    end
  end

  # Update weights and biases
  for l = 1:(length(network.layers))
    network.layers[l].weights .-= learning_rate * weight_gradients[l]
    network.layers[l].biases .-= learning_rate * bias_gradients[l]
  end

  return network.cost(activations[end], target_output)
end

"""
Get batches of data for training a neural network.

# Parameters
- `X::Matrix{Float64}`: The design matrix containing the inputs to the network.
- `y::Vector{Float64}`: The target outputs of the network.
- `batch_size::Int`: The size of each batch.

# Returns
- `Vector{Tuple{Matrix{Float64},Vector{Float64}}}`: The batches, each as a tuple of the design matrix and target output.
"""
function get_batches(x::Matrix{Float64}, y::Matrix{Float64}, batch_size::Int)
  n_samples = size(x, 1)
  indices = shuffle(1:n_samples)

  n_batches = ceil(Int, n_samples / batch_size)
  batches = Vector{Tuple{Matrix{Float64},Matrix{Float64}}}()

  for i = 1:n_batches
    start_idx = (i - 1) * batch_size + 1
    end_idx = min(i * batch_size, n_samples)
    batch_indices = indices[start_idx:end_idx]
    push!(batches, (x[batch_indices, :], y[batch_indices, :]))
  end

  return batches
end

"""
Train a neural network using backpropagation and stochastic gradient descent.

# Parameters
- `network::NeuralNetwork`: The neural network to train.
- `X::Matrix{Float64}`: The design matrix containing the inputs to the network.
- `y::Matrix{Float64}`: The target outputs of the network.
- `epochs::Int`: The number of epochs to train the network for.
- `learning_rate::Float64`: The learning rate to use for stochastic gradient descent.

# Returns
- `Vector{Float64}`: A vector of the loss at each epoch.
"""
function train_network(
  network::NeuralNetwork,
  x::Matrix{Float64},
  y::Matrix{Float64},
  epochs::Int,
  batch_size::Int,
  verbose::Bool=true,
)::Vector{Float64}
  losses = Vector{Float64}()

  for epoch = 1:epochs
    current_lr = network.lr_scheduler(epoch - 1)

    epoch_loss = 0.0

    batches = get_batches(x, y, batch_size)

    for (batch_x, batch_y) in batches
      batch_loss = backpropagate(network, batch_x, batch_y, current_lr)
      epoch_loss += batch_loss * size(batch_x, 1)
    end

    push!(losses, epoch_loss / n_samples)

    if verbose && (epoch % 100 == 0 || epoch == 1)
      println("Epoch $epoch/$epochs: loss = $(losses[end])")
    end
  end

  return losses
end

function predict(network::NeuralNetwork, input::Matrix{Float64})::Matrix{Float64}
  activation = input

  for layer in network.layers
    weighted_input = activation * layer.weights .+ layer.biases'
    activation = layer.activation.(weighted_input)
  end

  return activation
end
