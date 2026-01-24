# frozen_string_literal: true

# Ported from: https://github.com/SergioFierens/ai4r
# Ported by: Daniel Huffman
#
# You can redistribute it and/or modify it under the terms of
# the Mozilla Public License version 1.1  as published by the
# Mozilla Foundation at http://www.mozilla.org/MPL/MPL-1.1.txt

module Ai4r3
  module NeuralNetwork
    # Collection of common activation functions and their derivatives.
    class ActivationFunctions
      def sigmoid(x)
        1.0 / (1.0 + Math.exp(-x))
      end

      def tanh(x)
        Math.tanh(x)
      end

      def relu(x)
        [x, 0].max
      end

      # def softmax(arr)
      #   lambda do |arr|
      #     max = arr.max
      #     exps = arr.map { |v| Math.exp(v - max) }
      #     sum = exps.inject(:+)
      #     exps.map { |e| e / sum }
      #   end
      # end

      # def sigmoid(y)
      #   y * (1 - y)
      # end

      # def tanh(y)
      #   1.0 - (y**2)
      # end

      # def relu(y)
      #   y.positive? ? 1.0 : 0.0
      # end

      def softmax(y)
        y * (1 - y)
      end
    end
  end
end
