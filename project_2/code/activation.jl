lin(x) = x
lin_prime(x) = 1

sigmoid(x) = 1 / (1 + exp(-x))
sigmoid_prime(x) = sigmoid(x) * (1 - sigmoid(x))

relu(x) = max(0, x)
relu_prime(x) = x > 0 ? 1.0 : 0.0

leaky_relu(x, α=0.01) = max(α * x, x)
leaky_relu_prime(x, α=0.01) = x > 0 ? 1.0 : α
