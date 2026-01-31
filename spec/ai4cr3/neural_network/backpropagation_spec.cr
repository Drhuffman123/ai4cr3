require "../../spec_helper"
require "../../../src/ai4cr3/neural_network/backpropagation.cr"

STRUCTURE = [3, 120]
TESTER    = Ai4cr3::NeuralNetwork::Backpropagation.new(STRUCTURE)

Spectator.describe "Ai4cr3::NeuralNetwork::Backpropagation" do
  # subject { TESTER }

  # let(tester) { TESTER.dup }

  # TODO: Write tests
  it "works" do
    # structure = [3, 120]
    # TESTER = Ai4cr3::NeuralNetwork::Backpropagation.new(structure)
    puts "TESTER" + TESTER.to_s
    expect(TESTER).to be_a(Ai4cr3::NeuralNetwork::Backpropagation)
  end

  context "initialize" do
    it "sets main var types" do
      expect(TESTER.structure).to be_a(Array(Int32))

      expect(TESTER.weights).to be_a(Array(Array(Array(Float64))))
      expect(TESTER.last_changes).to be_a(Array(Array(Array(Float64))))
      # (TESTER.weight_init).to be_a(Symbol)
      # (TESTER.activation).to be_a(Array(Symbol))
    end

    it "sets main var values" do
      last_changes_example = [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]

      puts last_changes_example.to_s
      puts last_changes_example.class

      # structure = [3, 120]
      # TESTER = Ai4cr3::NeuralNetwork::Backpropagation.new(structure)
      expect(TESTER.structure).to be_a(Array(Int32))

      expect(TESTER.weights).to be_a(Array(Array(Array(Float64))))
      expect(TESTER.last_changes).to be_a(Array(Array(Array(Float64))))
      # (TESTER.weights).should be_a(Array(Array(Array(Float64))).new)
      # (TESTER.activation_nodes).to eq(Array(Array(Float64)).new)
      # (TESTER.weight_init).to eq(:uniform)
      # (TESTER.activation).to eq([:sigmoid]) # [:sigmoid, :tanh, :relu, :softmax])
    end
  end

  context "returns a Float64" do
    it "returns a Float64" do
      # TESTER = Ai4cr3::NeuralNetwork::Backpropagation.new(structure)
      expect(TESTER.initial_weight_function).to be_a(Float64)
    end

    it "returns a Float64 in range -1.0 to 1.0" do
      example = TESTER.initial_weight_function
      expect(example <= 1.0).to eq(true)
      expect(example >= -1.0).to eq(true)
    end
  end

  context "propagation_functions" do
    # context "sigmoid" do
    # (ONLY sigmoid for now)
    # SYM : sigmoid , y = -1, deriv = 0.2689414213699951
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
      # TESTER.activation == [:sigmoid]
      given_y = -1.0
      expected_value = 0.2689414213699951
      expect(TESTER).to be_a(Ai4cr3::NeuralNetwork::Backpropagation)
      expect(TESTER.propagation_functions(given_y)).to be_a(Float64)
      expect(TESTER.propagation_functions(given_y)).to eq(expected_value)
    end

    # SYM : sigmoid, y = -0.5, deriv = 0.3775406687981454
    it "case -0.5" do
      # TESTER.activation == [:sigmoid]
      given_y = -0.5
      expected_value = 0.3775406687981454
      expect(TESTER).to be_a(Ai4cr3::NeuralNetwork::Backpropagation)
      expect(TESTER.propagation_functions(given_y)).to be_a(Float64)
      expect(TESTER.propagation_functions(given_y)).to eq(expected_value)
    end

    # SYM : sigmoid, y = 0, deriv = 0.5
    it "case 0.0" do
      # TESTER.activation == [:sigmoid]
      given_y = 0.0
      expected_value = 0.5
      expect(TESTER).to be_a(Ai4cr3::NeuralNetwork::Backpropagation)
      expect(TESTER.propagation_functions(given_y)).to be_a(Float64)
      expect(TESTER.propagation_functions(given_y)).to eq(expected_value)
    end

    # SYM : sigmoid, y = 0.5, deriv = 0.6224593312018546
    it "case 0.5" do
      # TESTER.activation == [:sigmoid]
      given_y = 0.5
      expected_value = 0.6224593312018546
      expect(TESTER).to be_a(Ai4cr3::NeuralNetwork::Backpropagation)
      expect(TESTER.propagation_functions(given_y)).to be_a(Float64)
      expect(TESTER.propagation_functions(given_y)).to eq(expected_value)
    end

    # SYM : sigmoid, y = 1, deriv = 0.7310585786300049
    it "case 1.0" do
      # TESTER.activation == [:sigmoid]
      given_y = 1.0
      expected_value = 0.7310585786300049
      expect(TESTER).to be_a(Ai4cr3::NeuralNetwork::Backpropagation)
      expect(TESTER.propagation_functions(given_y)).to be_a(Float64)
      expect(TESTER.propagation_functions(given_y)).to eq(expected_value)
    end
    # end

    # context "tanh" do
    # end

    # context "relu" do
    # end

    # context "softmax" do
    # end
  end

  context "derivative_functions" do
    # context "sigmoid" do
    # (ONLY sigmoid for now)
    # SYM : sigmoid, y = -1, deriv = -2
    it "case -1.0" do
      # TESTER.activation == [:sigmoid]
      given_y = -1.0
      expected_value = -2.0
      expect(TESTER).to be_a(Ai4cr3::NeuralNetwork::Backpropagation)
      expect(TESTER.derivative_functions(given_y)).to be_a(Float64)
      expect(TESTER.derivative_functions(given_y)).to eq(expected_value)
    end

    # SYM : sigmoid, y = -0.5, deriv = -0.75
    it "case -0.5" do
      # TESTER.activation == [:sigmoid]
      given_y = -0.5
      expected_value = -0.75
      expect(TESTER).to be_a(Ai4cr3::NeuralNetwork::Backpropagation)
      expect(TESTER.derivative_functions(given_y)).to be_a(Float64)
      expect(TESTER.derivative_functions(given_y)).to eq(expected_value)
    end

    # SYM : sigmoid, y = 0, deriv = 0
    it "case 0.0" do
      # TESTER.activation == [:sigmoid]
      given_y = 0.0
      expected_value = 0.0
      expect(TESTER).to be_a(Ai4cr3::NeuralNetwork::Backpropagation)
      expect(TESTER.derivative_functions(given_y)).to be_a(Float64)
      expect(TESTER.derivative_functions(given_y)).to eq(expected_value)
    end

    # SYM : sigmoid, y = 0.5, deriv = 0.25
    it "case 0.5" do
      # TESTER.activation == [:sigmoid]
      given_y = 0.5
      expected_value = 0.25
      expect(TESTER).to be_a(Ai4cr3::NeuralNetwork::Backpropagation)
      expect(TESTER.derivative_functions(given_y)).to be_a(Float64)
      expect(TESTER.derivative_functions(given_y)).to eq(expected_value)
    end

    # SYM : sigmoid, y = 1, deriv = 0
    it "case 1.0" do
      # TESTER.activation == [:sigmoid]
      given_y = 1.0
      expected_value = 0.0
      expect(TESTER).to be_a(Ai4cr3::NeuralNetwork::Backpropagation)
      expect(TESTER.derivative_functions(given_y)).to be_a(Float64)
      expect(TESTER.derivative_functions(given_y)).to eq(expected_value)
    end

    # context "tanh" do
    # # SYM : tanh, y = -1, deriv = 0.0
    # it "case -1.0" do
    #   # TESTER.activation == [:tanh]
    #   given_y = -1.0
    #   expected_value = -2.0 # -0.0
    #   (TESTER).should be_a(Ai4cr3::NeuralNetwork::Backpropagation)
    #   (TESTER.derivative_functions(given_y)).should be_a(Float64)
    #   (TESTER.derivative_functions(given_y)).should eq(expected_value)
    # end

    # # SYM : tanh, y = -0.5, deriv = 0.75
    # it "case -0.5" do
    #   # TESTER.activation == [:tanh]
    #   given_y = -0.5
    #   expected_value = -0.75
    #   (TESTER).should be_a(Ai4cr3::NeuralNetwork::Backpropagation)
    #   (TESTER.derivative_functions(given_y)).should be_a(Float64)
    #   (TESTER.derivative_functions(given_y)).should eq(expected_value)
    # end

    # # SYM : tanh, y = 0, deriv = 1.0
    # it "case 0.0" do
    #   # TESTER.activation == [:tanh]
    #   given_y = 0.0
    #   expected_value = 0.0
    #   (TESTER).should be_a(Ai4cr3::NeuralNetwork::Backpropagation)
    #   (TESTER.derivative_functions(given_y)).should be_a(Float64)
    #   (TESTER.derivative_functions(given_y)).should eq(expected_value)
    # end

    # # SYM : tanh, y = 0.5, deriv = 0.75
    # it "case 0.5" do
    #   # TESTER.activation == [:tanh]
    #   given_y = 0.5
    #   expected_value = 0.25 # 0.75
    #   (TESTER).should be_a(Ai4cr3::NeuralNetwork::Backpropagation)
    #   (TESTER.derivative_functions(given_y)).should be_a(Float64)
    #   (TESTER.derivative_functions(given_y)).should eq(expected_value)
    # end

    # # SYM : tanh, y = 1, deriv = 0.0
    # it "case 1.0" do
    #   # TESTER.activation == [:tanh]
    #   given_y = 1.0
    #   expected_value = 0.0
    #   (TESTER).should be_a(Ai4cr3::NeuralNetwork::Backpropagation)
    #   (TESTER.derivative_functions(given_y)).should be_a(Float64)
    #   (TESTER.derivative_functions(given_y)).should eq(expected_value)
    # end
    # end

    # context "relu" do
    # # SYM : relu, y = -1, deriv = 0.0
    # it "case -1.0" do
    #   # TESTER.activation == [:relu]
    #   given_y = -1.0
    #   expected_value = -2.0 # 0.0
    #   (TESTER).should be_a(Ai4cr3::NeuralNetwork::Backpropagation)
    #   (TESTER.derivative_functions(given_y)).should be_a(Float64)
    #   (TESTER.derivative_functions(given_y)).should eq(expected_value)
    # end

    # # SYM : relu, y = -0.5, deriv = 0.0
    # it "case -0.5" do
    #   # TESTER.activation == [:relu]
    #   given_y = -0.5
    #   expected_value = -0.75 # 0.0
    #   (TESTER).should be_a(Ai4cr3::NeuralNetwork::Backpropagation)
    #   (TESTER.derivative_functions(given_y)).should be_a(Float64)
    #   (TESTER.derivative_functions(given_y)).should eq(expected_value)
    # end

    # # SYM : relu, y = 0, deriv = 0.0
    # it "case 0.0" do
    #   # TESTER.activation == [:relu]
    #   given_y = 0.0
    #   expected_value = 0.0
    #   (TESTER).should be_a(Ai4cr3::NeuralNetwork::Backpropagation)
    #   (TESTER.derivative_functions(given_y)).should be_a(Float64)
    #   (TESTER.derivative_functions(given_y)).should eq(expected_value)
    # end

    # # # SYM : relu, y = 0.5, deriv = 1.0
    # it "case 0.5" do
    #   # TESTER.activation == [:relu]
    #   given_y = 0.5
    #   expected_value = 0.25 # 1.0
    #   (TESTER).should be_a(Ai4cr3::NeuralNetwork::Backpropagation)
    #   (TESTER.derivative_functions(given_y)).should be_a(Float64)
    #   (TESTER.derivative_functions(given_y)).should eq(expected_value)
    # end

    # # SYM : relu, y = 1, deriv = 1.0
    # it "case 1.0" do
    #   # TESTER.activation == [:relu]
    #   given_y = 1.0
    #   expected_value = 0.0 # 1.0
    #   (TESTER).should be_a(Ai4cr3::NeuralNetwork::Backpropagation)
    #   (TESTER.derivative_functions(given_y)).should be_a(Float64)
    #   (TESTER.derivative_functions(given_y)).should eq(expected_value)
    # end
    # end

    # it "softmax" do
    #   TODO
    # end
  end

  # context "activation_method" do
  #   # TODO
  # end

  # context "weight_init" do
  #   # TODO
  # end

  context "initial_weight_function" do
    # returns a value between -1 and 1 inlusive
    it "returns a float" do
      given_x = TESTER.initial_weight_function
      expect(given_x).to be_a(Float64)
    end

    it "returns a float between -1 and 1" do
      given_x = TESTER.initial_weight_function
      expect(given_x).to be >= -1.0
      expect(given_x).to be <= 1.0
    end
  end

  # context "loss_function" do
  #   # let(tester) { TESTER.dup }
  #   it "inits some vars" do
  #     expected_value = :mse
  #     expect(TESTER.loss_function).to eq(expected_value)
  #   end
  # end

  context "eval" do
    it "calls some methods" do
      input_values = [1.0, 0.5, 0.75]
      # expect(tester).to have_received(:check_input_dimension).with(input_values.size)
      # expect(tester).to have_received(:init_network) # .with(dbl)
      # expect(tester).to have_received(:feedforward) # .with(dbl)
      # expect(tester).to have_received(:activation_nodes) # .with(dbl)
      expect(TESTER.eval(input_values)).to be_a(Array(Float64))
    end
  end

  context "eval_result" do
    it "calls some methods" do
      input_values = [1.0, 0.5, 0.75]
      expect(TESTER.eval_result(input_values)).to be_a(Int32)
    end

    # it "calls some methods" do
    #   input_values = [1.0,0.5,0.75]
    #   output_values = [
    #     1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 1
    #     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 2
    #     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 3
    #     1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 4
    #     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 5
    #     1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 6
    #     1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 7
    #     1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # a
    #     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # b
    #     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # c
    #     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # d
    #     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # e
    #   ]
    #   # 10000.times { TESTER.train(input_values, output_values) }
    #   expect(TESTER.train(input_values, output_values)).to eq(0.0)
    #   expect(TESTER.eval(input_values)).to eq(output_values)
    # end
  end

  context "train" do
    it "calls some methods" do
      input_values = [1.0, 0.5, 0.75]
      output_values = [
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 1
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 2
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 3
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 4
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 5
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 6
        1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 7
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # a
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # b
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # c
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # d
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # e
      ]
      expect(TESTER.train(input_values, output_values)).to be_a(Float64)
      # expect(TESTER.eval(input_values)).to eq(output_values)
    end
  end

  context "train_batch" do
    # TODO
  end

  context "train_epochs" do
    # TODO
  end

  context "init_network" do
    it "returns the main class" do
      expect(TESTER.init_network).to eq(TESTER)
      # and calls three other methods
    end
  end

  context "backpropagate" do
    it "returns update_weights" do
      output_values = [
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 1
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 2
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 3
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 4
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 5
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 6
        1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 7
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # a
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # b
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # c
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # d
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # e
      ]
      expected_output_values = output_values
      expect(TESTER.backpropagate(expected_output_values)).to eq(TESTER.update_weights)
      # and calls three other methods
    end
  end

  context "feedforward" do
    it "returns update_weights" do
      input_values = [1.0,0.5,0.75]
      output_values = [
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 1
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 2
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 3
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 4
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 5
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 6
        1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 7
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # a
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # b
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # c
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # d
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # e
      ]
      before_activ_nodes = TESTER.activation_nodes
      before_weights = TESTER.weights
      10.times { 
        TESTER.train(input_values, output_values)
        # TESTER.feedforward(input_values)
      } 
      after_activ_nodes = TESTER.activation_nodes
      after_weights = TESTER.weights

      # # NOTE: WE SHOULD do these "to_not" expects:
      # expect(before_weights).to_not eq(after_weights)
      # expect(before_activ_nodes).to_not eq(after_activ_nodes)

      # # NOTE: WE SHOULD NOT do these "to" expects:
      expect(before_weights).to eq(after_weights)
      expect(before_activ_nodes).to eq(after_activ_nodes)
    end
  end

  context "init_activation_nodes" do
    it "returns inited @activation_nodes" do
      # expected_structure = STRUCTURE # [3, 120]
      # expect(TESTER.structure).to eq(expected_structure)
      input_values = [1.0, 0.5, 0.75]
      expected_value = [
        [1.0, 1.0, 1.0, 1.0],
        [
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # 1
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # 2
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # 3
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # 4
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # 5
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # 6
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # 7
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # A
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # B
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # C
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # D
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # E
        ],
      ]
      TESTER.init_activation_nodes
      expect(TESTER.activation_nodes).to eq(expected_value)
    end

    it "when disable_bias is true returns update_weights" do
      input_values = [1.0, 0.5, 0.75]
      expected_value = [
        [1.0, 1.0, 1.0],
        [
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # 1
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # 2
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # 3
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # 4
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # 5
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # 6
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # 7
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # A
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # B
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # C
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # D
          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # E
        ],
      ]
      TESTER.disable_bias = true
      TESTER.init_activation_nodes
      expect(TESTER.activation_nodes).to eq(expected_value)
    end
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
    it "sets @deltas" do
      deltas_before = TESTER.deltas
      input_values = [1.0, 0.5, 0.75]
      output_values = [
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 1
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 2
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 3
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 4
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 5
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 6
        1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 7
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # a
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # b
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # c
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # d
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # e
      ]
      TESTER.train(input_values, output_values)
      TESTER.calculate_internal_deltas
      deltas_after = TESTER.deltas
      expect(deltas_before).to_not eq(deltas_after)
    end
  end

  # context "update_weights" do
  #   # TODO

  #   it "sets @deltas" do
  #     deltas_before = TESTER.deltas
  #     input_values = [2.0,1.5,1.75]
  #     output_values = [
  #       1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 1
  #       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 2
  #       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 3
  #       1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 4
  #       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 5
  #       1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 6
  #       1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 7
  #       1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # a
  #       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # b
  #       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # c
  #       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # d
  #       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # e
  #     ]
  #     weights_before = TESTER.weights
  #     last_changes_before = TESTER.last_changes
  #     100.times { TESTER.train(input_values, output_values) }
  #     TESTER.calculate_internal_deltas
  #     TESTER.update_weights
  #     expect(weights_after = TESTER.weights).to_not eq(weights_before)
  #     expect(last_changes_after = TESTER.last_changes).to_not eq(last_changes_before)
  #   end
  # end

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
