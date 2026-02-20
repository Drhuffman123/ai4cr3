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

      # TESTER.should receive("check_input_dimension", input_values)
      # TESTER.should receive("init_network")
      # TESTER.should receive("feedforward")

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
      # expect(TESTER.eval(input_values)).to be_a(Array(Float64))
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
      before_activ_nodes = TESTER.activation_nodes
      before_weights = TESTER.weights
      10.times {
        TESTER.train(input_values, output_values)
      # TESTER.feedforward(input_values)
      }
      after_activ_nodes = TESTER.activation_nodes
      after_weights = TESTER.weights

      # # TODO: WE SHOULD do these "to_not" expects:
      # expect(before_weights).to_not eq(after_weights)
      # expect(before_activ_nodes).to_not eq(after_activ_nodes)

      # # TODO: WE SHOULD NOT do these "to" expects:
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
    it "resets weights" do
      input_values = [1.0, 0.5, 0.75]
      output_values = [
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
      ]
      TESTER.train(input_values, output_values)
      weights_after_training = TESTER.weights
      TESTER.init_weights
      weights_after_reset = TESTER.weights
      expect(weights_after_training).to_not eq(weights_after_reset)
    end
  end

  context "init_last_changes" do
    it "..." do
      input_values = [1.0, 0.5, 0.75]
      output_values = [
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
      ]
      expected_last_changes = [
        [
          [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          ],
          [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          ],
          [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          ],
          [
            0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
            -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
            -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
            0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
            -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
            0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
            0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
            0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
            -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
            -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
            -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
            -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
          ],
        ],
      ]
      TESTER.train(input_values, output_values)
      last_changes_before_initing = TESTER.last_changes
      TESTER.init_last_changes
      last_changes_after_initing = TESTER.last_changes
      expect(last_changes_before_initing).to_not eq(last_changes_after_initing)
    end
  end

  context "init_weights" do
    it "..." do
      input_values = [1.0, 0.5, 0.75]
      output_values = [
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
      ]
      expected_last_changes = [
        [
          [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          ],
          [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          ],
          [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          ],
          [
            0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
            -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
            -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
            0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
            -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
            0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
            0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
            0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
            -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
            -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
            -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
            -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
          ],
        ],
      ]
      TESTER.train(input_values, output_values)
      last_changes_before_initing = TESTER.weights
      TESTER.init_weights
      last_changes_after_initing = TESTER.weights
      expect(last_changes_before_initing).to_not eq(last_changes_after_initing)
    end
  end

  context "calculate_output_deltas" do
    it "changes the output_values" do
      input_values = [1.0, 0.5, 0.75]
      expected_output_values = [
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
      deltas_before = TESTER.deltas
      TESTER.calculate_output_deltas(expected_output_values)
      deltas_after = TESTER.deltas
      activation_nodes_after = TESTER.activation_nodes.last
      expect(deltas_before).to_not eq(deltas_after)
    end
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

  # context "calc_a_single_weight" do
  #   it "WHAT IS UP WITH THAT?" do
  #     weights_before = TESTER.weights
  #     w_before = TESTER.weights[0][1][2]
  #     TESTER.weights[0][1][2] += -0.5
  #     TESTER.update_weights
  #     TESTER.calc_a_single_weight(0,1,2)
  #     w_after = TESTER.weights[0][1][2]
  #     weights_after = TESTER.weights
  #     expect(w_before).to_not eq(w_after)
  #     expect(weights_before[0][1][2]).to_not eq(weights_after[0][1][2]) # WHY is this not working???!!!
  #   end
  #
  #   it "updates @weights" do
  #     weights_before = TESTER.weights
  #     input_values = [1.0, 0.5, 0.75]
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
  #       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # e
  #     ]
  #     10.times { TESTER.train(input_values, output_values) }
  #     TESTER.calculate_internal_deltas
  #     weights_after = TESTER.weights
  #     expect(weights_before).to_not eq(weights_after)
  #   end
  #
  #   it "updates @last_changes" do
  #     last_changes_before = TESTER.last_changes
  #     input_values = [1.0, 0.5, 0.75]
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
  #       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # e
  #     ]
  #     10.times {
  #       TESTER.train(input_values, output_values)
  #       TESTER.calculate_internal_deltas
  #     }
  #     last_changes_after = TESTER.last_changes
  #     expect(last_changes_before).to_not eq(last_changes_after)
  #   end
  # end

  # context "update_weights" do
  #   # TODO
  #
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
    it "returns the error" do
      input_values = [2.0, 1.5, 1.75]
      expected_output = [
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
      actual_output = TESTER.calculate_error(expected_output)
      expect(actual_output).to_not eq(0.0)
      expect(actual_output).to be_a(Float64)
      is_positive = (actual_output > 0.0)
      expect(is_positive).to eq(true)
    end
  end

  context "calculate_loss" do
    it "returns the error" do
      input_values = [2.0, 1.5, 1.75]
      expected_output = [
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
      actual_output = [
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 1
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
      expected_error = 1.0
      actual_error = TESTER.calculate_loss(expected_output, actual_output)
      expect(actual_error).to eq(expected_error)
    end

    it "returns the error" do
      input_values = [2.0, 1.5, 1.75]
      expected_output = [
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
      actual_output = [
        0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 1
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
      expected_error = 1.5
      actual_error = TESTER.calculate_loss(expected_output, actual_output)
      # expect(actual_output).to_not eq(0.0)
      expect(actual_error).to eq(expected_error)
    end

    it "returns the error" do
      input_values = [2.0, 1.5, 1.75]
      expected_output = [
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
      actual_output = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 1
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 2
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 3
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 4
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 5
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 6
        1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 7
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # a
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # b
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # c
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # d
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # e
      ]
      expected_error = 3.0
      actual_error = TESTER.calculate_loss(expected_output, actual_output)
      expect(actual_error).to eq(expected_error)
    end
  end

  context "check_input_dimension" do
    context "correctly sized inputs" do
      it "should not error (just returns a nil)" do
        input_values = [2.0, 1.5, 1.75]
        result = TESTER.check_input_dimension(input_values)
        expect(result).to eq(nil)
      end
    end

    context "INcorrectly sized inputs" do
      it "should error (when too few)" do
        input_values = [2.0]
        expect_raises(InputsException) do
          TESTER.check_input_dimension(input_values)
        end
      end

      it "should error (when too many)" do
        input_values = [2.0, 1.5, 1.75, 1.0]
        expect_raises(InputsException) do
          TESTER.check_input_dimension(input_values)
        end
      end
    end
  end

  context "check_output_dimension" do
    # TODO
    OutputsException
    context "correctly sized inputs" do
      it "should not error (just returns a nil)" do
        outputs = [
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 1
          1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 2
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 3
          0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 4
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 5
          0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 6
          1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 7
          1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # a
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # b
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # c
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # d
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # e
        ]
        result = TESTER.check_output_dimension(outputs)
        expect(result).to eq(nil)
      end
    end

    context "INcorrectly sized inputs" do
      it "should error (when too few)" do
        outputs = [
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 1
          1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 2
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 3
          0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 4
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 5
          0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 6
          1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 7
          1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # a
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # b
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # c
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # d
        ]
        expect_raises(OutputsException) do
          TESTER.check_output_dimension(outputs)
        end
      end

      it "should error (when too many)" do
        outputs = [
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 1
          1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 2
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 3
          0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 4
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 5
          0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 6
          1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 7
          1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # a
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # b
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # c
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # d
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # e
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]
        expect_raises(OutputsException) do
          TESTER.check_output_dimension(outputs)
        end
      end
    end
  end

  context "from_yaml" do
    it "loads the file" do
      file_path = "a_backprop.yml"
      TESTER.save(file_path)
      yml_content = File.read(file_path)
      new_backprop = Ai4cr3::NeuralNetwork::Backpropagation.from_yaml(yml_content)
      expect(new_backprop).to be_a(Ai4cr3::NeuralNetwork::Backpropagation)
      # puts "yml_content: #{yml_content}"
      # puts "new_backprop: #{new_backprop}"
    end
  end

  context "save" do
    it "saves the file" do
      file_path = "a_backprop.yml"
      File.delete(file_path)
      is_file = File.exists?(file_path)
      expect(is_file).to be(false)
      TESTER.save(file_path)
      is_file = File.exists?(file_path)
      expect(is_file).to be(true)
    end
  end
end
