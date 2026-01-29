require "../../spec_helper"
require "../../../src/ai4cr3/neural_network/backpropagation.cr"

STRUCTURE = [3, 120]
TESTER    = Ai4cr3::NeuralNetwork::Backpropagation.new(STRUCTURE)

describe Ai4cr3::NeuralNetwork::Backpropagation do
  # TODO: Write tests
  it "works" do
    # structure = [3, 120]
    # TESTER = Ai4cr3::NeuralNetwork::Backpropagation.new(structure)
    puts "TESTER" + TESTER.to_s
    (TESTER).should be_a(Ai4cr3::NeuralNetwork::Backpropagation)
  end

  context "initialize" do
    it "sets main var types" do
      # structure = [3, 120]
      # TESTER = Ai4cr3::NeuralNetwork::Backpropagation.new(structure)
      (TESTER.structure).should be_a(Array(Int32))

      (TESTER.weights).should be_a(Array(Array(Float64)))
      (TESTER.activation_nodes).should be_a(Array(Array(Float64)))
      (TESTER.last_changes).should be_a(Array(Array(Float64)))
      (TESTER.weight_init).should be_a(Symbol)
      (TESTER.activation).should be_a(Array(Symbol))
    end

    it "sets main var values" do
      # structure = [3, 120]
      # TESTER = Ai4cr3::NeuralNetwork::Backpropagation.new(structure)
      (TESTER.structure).should be_a(Array(Int32))

      (TESTER.weights).should eq(Array(Array(Float64)).new)
      (TESTER.activation_nodes).should eq(Array(Array(Float64)).new)
      (TESTER.last_changes).should eq(Array(Array(Float64)).new)
      (TESTER.weight_init).should eq(:uniform)
      (TESTER.activation).should eq([:sigmoid]) # [:sigmoid, :tanh, :relu, :softmax])
    end
  end

  context "returns a Float64" do
    it "returns a Float64" do
      # TESTER = Ai4cr3::NeuralNetwork::Backpropagation.new(structure)
      TESTER.initial_weight_function.should be_a(Float64)
    end

    it "returns a Float64 in range -1.0 to 1.0" do
      example = TESTER.initial_weight_function
      (example <= 1.0).should eq(true)
      (example >= -1.0).should eq(true)
    end
  end

  context "activation_param_change" do
    # TODO
  end

  # context "activation=" do
  #   context "when symbol = :sigmoid" do
  #     it "sets @activation" do
  #       syms = :sigmoid
  #       TESTER.activation=(syms).should eq("FOO")
  #     end
  #   end
  # 
  #   context "when symbol = [:sigmoid]" do
  #     it "sets @activation" do
  #       syms = [:sigmoid]
  #       TESTER.activation=(syms).should eq("FOO")
  #     end
  #   end
  # end

  context "propagation_functions" do
    context "sigmoid" do
      # SYM : sigmoid, y = -1, deriv = 0.2689414213699951
      # SYM : sigmoid, y = -0.75, deriv = 0.320821300824607
      # SYM : sigmoid, y = -0.5, deriv = 0.3775406687981454
      # SYM : sigmoid, y = -0.25, deriv = 0.43782349911420193
      # SYM : sigmoid, y = -0.1, deriv = 0.47502081252106
      # SYM : sigmoid, y = -0.01, deriv = 0.49750002083312506
      # SYM : sigmoid, y = 0, deriv = 0.5
      # SYM : sigmoid, y = 0.01, deriv = 0.5024999791668749
      # SYM : sigmoid, y = 0.1, deriv = 0.52497918747894
      # SYM : sigmoid, y = 0.25, deriv = 0.5621765008857981
      # SYM : sigmoid, y = 0.5, deriv = 0.6224593312018546
      # SYM : sigmoid, y = 0.75, deriv = 0.679178699175393
      # SYM : sigmoid, y = 1, deriv = 0.7310585786300049

      # SYM : sigmoid, y = -1, deriv = 0.2689414213699951
      it "case -1.0" do
        TESTER.activation == [:sigmoid]
        given_y = -1.0
        expected_value = 0.2689414213699951
        (TESTER).should be_a(Ai4cr3::NeuralNetwork::Backpropagation)
        (TESTER.propagation_functions(given_y)).should be_a(Float64)
        (TESTER.propagation_functions(given_y)).should eq(expected_value)
      end

      # SYM : sigmoid, y = -0.5, deriv = 0.3775406687981454
      it "case -0.5" do
        TESTER.activation == [:sigmoid]
        given_y = -0.5
        expected_value = 0.3775406687981454
        (TESTER).should be_a(Ai4cr3::NeuralNetwork::Backpropagation)
        (TESTER.propagation_functions(given_y)).should be_a(Float64)
        (TESTER.propagation_functions(given_y)).should eq(expected_value)
      end

      # SYM : sigmoid, y = 0, deriv = 0.5
      it "case 0.0" do
        TESTER.activation == [:sigmoid]
        given_y = 0.0
        expected_value = 0.5
        (TESTER).should be_a(Ai4cr3::NeuralNetwork::Backpropagation)
        (TESTER.propagation_functions(given_y)).should be_a(Float64)
        (TESTER.propagation_functions(given_y)).should eq(expected_value)
      end

      # SYM : sigmoid, y = 0.5, deriv = 0.6224593312018546
      it "case 0.5" do
        TESTER.activation == [:sigmoid]
        given_y = 0.5
        expected_value = 0.6224593312018546
        (TESTER).should be_a(Ai4cr3::NeuralNetwork::Backpropagation)
        (TESTER.propagation_functions(given_y)).should be_a(Float64)
        (TESTER.propagation_functions(given_y)).should eq(expected_value)
      end

      # SYM : sigmoid, y = 1, deriv = 0.7310585786300049
      it "case 1.0" do
        TESTER.activation == [:sigmoid]
        given_y = 1.0
        expected_value = 0.7310585786300049
        (TESTER).should be_a(Ai4cr3::NeuralNetwork::Backpropagation)
        (TESTER.propagation_functions(given_y)).should be_a(Float64)
        (TESTER.propagation_functions(given_y)).should eq(expected_value)
      end
    end

    # context "tanh" do
    # end

    # context "relu" do
    # end

    # context "softmax" do
    # end
  end

  context "derivative_functions" do
    context "sigmoid" do
      # SYM : sigmoid, y = -1, deriv = -2
      it "case -1.0" do
        TESTER.activation == [:sigmoid]
        given_y = -1.0
        expected_value = -2.0
        (TESTER).should be_a(Ai4cr3::NeuralNetwork::Backpropagation)
        (TESTER.derivative_functions(given_y)).should be_a(Float64)
        (TESTER.derivative_functions(given_y)).should eq(expected_value)
      end

      # SYM : sigmoid, y = -0.5, deriv = -0.75
      it "case -0.5" do
        TESTER.activation == [:sigmoid]
        given_y = -0.5
        expected_value = -0.75
        (TESTER).should be_a(Ai4cr3::NeuralNetwork::Backpropagation)
        (TESTER.derivative_functions(given_y)).should be_a(Float64)
        (TESTER.derivative_functions(given_y)).should eq(expected_value)
      end

      # SYM : sigmoid, y = 0, deriv = 0
      it "case 0.0" do
        TESTER.activation == [:sigmoid]
        given_y = 0.0
        expected_value = 0.0
        (TESTER).should be_a(Ai4cr3::NeuralNetwork::Backpropagation)
        (TESTER.derivative_functions(given_y)).should be_a(Float64)
        (TESTER.derivative_functions(given_y)).should eq(expected_value)
      end

      # SYM : sigmoid, y = 0.5, deriv = 0.25
      it "case 0.5" do
        TESTER.activation == [:sigmoid]
        given_y = 0.5
        expected_value = 0.25
        (TESTER).should be_a(Ai4cr3::NeuralNetwork::Backpropagation)
        (TESTER.derivative_functions(given_y)).should be_a(Float64)
        (TESTER.derivative_functions(given_y)).should eq(expected_value)
      end

      # SYM : sigmoid, y = 1, deriv = 0
      it "case 1.0" do
        TESTER.activation == [:sigmoid]
        given_y = 1.0
        expected_value = 0.0
        (TESTER).should be_a(Ai4cr3::NeuralNetwork::Backpropagation)
        (TESTER.derivative_functions(given_y)).should be_a(Float64)
        (TESTER.derivative_functions(given_y)).should eq(expected_value)
      end
    end

    context "tanh" do
      # # SYM : tanh, y = -1, deriv = 0.0
      # it "case -1.0" do
      #   TESTER.activation == [:tanh]
      #   given_y = -1.0
      #   expected_value = -2.0 # -0.0
      #   (TESTER).should be_a(Ai4cr3::NeuralNetwork::Backpropagation)
      #   (TESTER.derivative_functions(given_y)).should be_a(Float64)
      #   (TESTER.derivative_functions(given_y)).should eq(expected_value)
      # end

      # # SYM : tanh, y = -0.5, deriv = 0.75
      # it "case -0.5" do
      #   TESTER.activation == [:tanh]
      #   given_y = -0.5
      #   expected_value = -0.75
      #   (TESTER).should be_a(Ai4cr3::NeuralNetwork::Backpropagation)
      #   (TESTER.derivative_functions(given_y)).should be_a(Float64)
      #   (TESTER.derivative_functions(given_y)).should eq(expected_value)
      # end

      # # SYM : tanh, y = 0, deriv = 1.0
      # it "case 0.0" do
      #   TESTER.activation == [:tanh]
      #   given_y = 0.0
      #   expected_value = 0.0
      #   (TESTER).should be_a(Ai4cr3::NeuralNetwork::Backpropagation)
      #   (TESTER.derivative_functions(given_y)).should be_a(Float64)
      #   (TESTER.derivative_functions(given_y)).should eq(expected_value)
      # end

      # # SYM : tanh, y = 0.5, deriv = 0.75
      # it "case 0.5" do
      #   TESTER.activation == [:tanh]
      #   given_y = 0.5
      #   expected_value = 0.25 # 0.75
      #   (TESTER).should be_a(Ai4cr3::NeuralNetwork::Backpropagation)
      #   (TESTER.derivative_functions(given_y)).should be_a(Float64)
      #   (TESTER.derivative_functions(given_y)).should eq(expected_value)
      # end

      # # SYM : tanh, y = 1, deriv = 0.0
      # it "case 1.0" do
      #   TESTER.activation == [:tanh]
      #   given_y = 1.0
      #   expected_value = 0.0
      #   (TESTER).should be_a(Ai4cr3::NeuralNetwork::Backpropagation)
      #   (TESTER.derivative_functions(given_y)).should be_a(Float64)
      #   (TESTER.derivative_functions(given_y)).should eq(expected_value)
      # end
    end

    context "relu" do
      # # SYM : relu, y = -1, deriv = 0.0
      # it "case -1.0" do
      #   TESTER.activation == [:relu]
      #   given_y = -1.0
      #   expected_value = -2.0 # 0.0
      #   (TESTER).should be_a(Ai4cr3::NeuralNetwork::Backpropagation)
      #   (TESTER.derivative_functions(given_y)).should be_a(Float64)
      #   (TESTER.derivative_functions(given_y)).should eq(expected_value)
      # end

      # # SYM : relu, y = -0.5, deriv = 0.0
      # it "case -0.5" do
      #   TESTER.activation == [:relu]
      #   given_y = -0.5
      #   expected_value = -0.75 # 0.0
      #   (TESTER).should be_a(Ai4cr3::NeuralNetwork::Backpropagation)
      #   (TESTER.derivative_functions(given_y)).should be_a(Float64)
      #   (TESTER.derivative_functions(given_y)).should eq(expected_value)
      # end

      # # SYM : relu, y = 0, deriv = 0.0
      # it "case 0.0" do
      #   TESTER.activation == [:relu]
      #   given_y = 0.0
      #   expected_value = 0.0
      #   (TESTER).should be_a(Ai4cr3::NeuralNetwork::Backpropagation)
      #   (TESTER.derivative_functions(given_y)).should be_a(Float64)
      #   (TESTER.derivative_functions(given_y)).should eq(expected_value)
      # end

      # # # SYM : relu, y = 0.5, deriv = 1.0
      # it "case 0.5" do
      #   TESTER.activation == [:relu]
      #   given_y = 0.5
      #   expected_value = 0.25 # 1.0 
      #   (TESTER).should be_a(Ai4cr3::NeuralNetwork::Backpropagation)
      #   (TESTER.derivative_functions(given_y)).should be_a(Float64)
      #   (TESTER.derivative_functions(given_y)).should eq(expected_value)
      # end

      # # SYM : relu, y = 1, deriv = 1.0
      # it "case 1.0" do
      #   TESTER.activation == [:relu]
      #   given_y = 1.0
      #   expected_value = 0.0 # 1.0
      #   (TESTER).should be_a(Ai4cr3::NeuralNetwork::Backpropagation)
      #   (TESTER.derivative_functions(given_y)).should be_a(Float64)
      #   (TESTER.derivative_functions(given_y)).should eq(expected_value)
      # end
    end

    # it "softmax" do
    #   TODO
    # end
  end

  context "activation_method" do
    # TODO
  end

  context "activation" do
    it "if @activation.is_a?(Array)" do
      TESTER.activation == [:sigmoid]
    end

    it "if @set_by_loss || (@loss_function == :cross_entropy && @activation_overridden)" do
      TESTER.activation == [:sigmoid]
    end

    it "else" do
      
    end
  end

  context "weight_init" do
    # TODO
  end

  context "initial_weight_function" do
    # TODO
  end

  context "loss_function" do
    # TODO
  end

  context "eval" do
    # TODO
  end

  context "eval_result" do
    # TODO
  end

  context "train" do
    # TODO
  end

  context "train_batch" do
    # TODO
  end

  context "train_epochs" do
    # TODO
  end

  context "init_network" do
    # TODO
  end

  context "backpropagate" do
    # TODO
  end

  context "feedforward" do
    # TODO
  end

  context "init_activation_nodes" do
    # TODO
  end

  context "init_weights" do
    # TODO
  end

  context "init_last_changes" do
    # TODO
  end

  context "init_weights" do
    # TODO
  end

  context "calculate_output_deltas" do
    # TODO
  end

  context "calculate_internal_deltas" do
    # TODO
  end

  context "update_weights" do
    # TODO
  end

  context "calculate_error" do
    # TODO
  end

  context "calculate_loss" do
    # TODO
  end

  context "check_input_dimension" do
    # TODO
  end

  context "check_output_dimension" do
    # TODO
  end
end

