mse(y_pred, y_true) = mean((y_pred .- y_true) .^ 2)
mse_prime(y_pred, y_true) = y_pred .- y_true

function binary_cross_entropy(logits, y_true)
  probs = 1 ./ (1 .+ exp.(-logits))  # sigmoid
  -mean(y_true .* log.(probs .+ 1e-10) .+ (1 .- y_true) .* log.(1 .- probs .+ 1e-10))
end

function binary_cross_entropy_prime(logits, y_true)
  probs = 1 ./ (1 .+ exp.(-logits))
  return (probs .- y_true) ./ length(y_true)
end
