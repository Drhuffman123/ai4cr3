require "../../spec_helper"
require "../../../src/ai4cr3/neural_network/backpropagation.cr"

describe Ai4cr3::NeuralNetwork::Backpropagation do
  # TODO: Write tests

  it "works" do
    structure = [3, 120]
    tester = Ai4cr3::NeuralNetwork::Backpropagation.new(structure)
    tester.should be_a(Ai4cr3::NeuralNetwork::Backpropagation)
  end
end

# In spec/ai4cr3/neural_network/backpropagation_spec.cr:2:1
#
#  2 | require "../../../../src/ai4cr3/neural_network/backpropagation.cr"
#      ^
# Error: can't find file '../../../../src/ai4cr3/neural_network/backpropagation.cr'
# relative to '/home/drhuffman/dev-github-redo/drhuffman123/ai4cr3/
#   spec/ai4cr3/neural_network/backpropagation_spec.cr'
