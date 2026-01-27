require "../../spec_helper"
require "../../../src/ai4cr3/neural_network/backpropagation.cr"

describe Ai4cr3::NeuralNetwork::Backpropagation do
  # TODO: Write tests

  it "works" do
    structure = [3, 120]
    tester = Ai4cr3::NeuralNetwork::Backpropagation.new(structure)
    puts "tester" + tester.to_s
    (tester).should be_a(Ai4cr3::NeuralNetwork::Backpropagation)
  end

  context "initialize" do
    it "sets main var types" do
      structure = [3, 120]
      tester = Ai4cr3::NeuralNetwork::Backpropagation.new(structure)
      (tester.structure).should be_a(Array(Int32))

      (tester.weights).should be_a(Array(Array(Float64)))
      (tester.activation_nodes).should be_a(Array(Array(Float64)))
      (tester.last_changes).should be_a(Array(Array(Float64)))
      (tester.weight_init).should be_a(Symbol)
      (tester.activation).should be_a(Array(Symbol))
    end

    it "sets main var values" do
      structure = [3, 120]
      tester = Ai4cr3::NeuralNetwork::Backpropagation.new(structure)
      (tester.structure).should be_a(Array(Int32))

      (tester.weights).should eq(Array(Array(Float64)).new)
      (tester.activation_nodes).should eq(Array(Array(Float64)).new)
      (tester.last_changes).should eq(Array(Array(Float64)).new)
      (tester.weight_init).should eq(:uniform)
      (tester.activation).should eq([:sigmoid]) # [:sigmoid, :tanh, :relu, :softmax])
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
