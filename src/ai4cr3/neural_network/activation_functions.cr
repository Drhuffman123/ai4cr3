module Ai4cr3
  module NeuralNetwork
    # Collection of common activation functions and their derivatives.
    module ActivationFunctions
      FUNCTIONS = {
        sigmoid: ->(x) { 1.0 / (1.0 + Math.exp(-x)) },
        tanh:    ->(x) { Math.tanh(x) },
        relu:    ->(x) { [x, 0].max },

        softmax: lambda do |arr|
          max = arr.max
          exps = arr.map { |v| Math.exp(v - max) }
          sum = exps.inject(:+)
          exps.map { |e| e / sum }
        end,
      }.freeze

      DERIVATIVES = {
        sigmoid: ->(y) { y * (1 - y) },
        tanh:    ->(y) { 1.0 - (y**2) },
        relu:    ->(y) { y.positive? ? 1.0 : 0.0 },
        softmax: ->(y) { y * (1 - y) },
      }.freeze
    end
  end
end
