abstract type LearningRateScheduler end

struct ConstantLR <: LearningRateScheduler
  learning_rate::Float64
end
(s::ConstantLR)(epoch::Int) = s.learning_rate

struct StepLR <: LearningRateScheduler
  initial_learning_rate::Float64
  step_size::Int
  gamma::Float64
end
function (s::StepLR)(epoch::Int)
  decay = s.gamma^(epoch ÷ s.step_size)
  return s.initial_learning_rate * decay
end

struct ExponentialLR <: LearningRateScheduler
  initial_learning_rate::Float64
  gamma::Float64
end
(s::ExponentialLR)(epoch::Int) = s.initial_learning_rate * s.gamma^epoch
