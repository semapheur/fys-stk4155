mse(y_pred, y_true) = mean((y_pred .- y_true) .^ 2)
mse_prime(y_pred, y_true) = y_pred .- y_true