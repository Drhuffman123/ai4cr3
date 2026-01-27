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
end

# In spec/ai4cr3/neural_network/backpropagation_spec.cr:2:1
#
#  2 | require "../../../../src/ai4cr3/neural_network/backpropagation.cr"
#      ^
# Error: can't find file '../../../../src/ai4cr3/neural_network/backpropagation.cr'
# relative to '/home/drhuffman/dev-github-redo/drhuffman123/ai4cr3/
#   spec/ai4cr3/neural_network/backpropagation_spec.cr'
