module Ai4r
  module NeuralNetwork
    # Collection of common weight initialization strategies.
    module WeightInitializations
      # Uniform distribution in [-1, 1)
      def uniform
        ->(_n, _i, _j) { (rand * 2) - 1 }
      end

      # Xavier/Glorot initialization based on layer dimensions
      def xavier(structure)
        lambda do |layer, _i, _j|
          limit = Math.sqrt(6.0 / (structure[layer] + structure[layer + 1]))
          (rand * 2 * limit) - limit
        end
      end

      # He initialization suitable for ReLU activations
      def he(structure)
        lambda do |layer, _i, _j|
          limit = Math.sqrt(6.0 / structure[layer])
          (rand * 2 * limit) - limit
        end
      end

      # module_function :uniform, :xavier, :he
    end
  end
end
