require "../../spec_helper"
require "../../../src/neural_network/backpropagation.cr"

describe "Ai4cr3::NeuralNetwork::Backpropagation" do
  it "the module" do
    network_structure = [3,120]
    tester = Ai4r3::NeuralNetwork::Backpropagation.new(network_structure)
    tester.should be_a(Ai4r3::NeuralNetwork::Backpropagation)
  end

  context "eval" do
    it "Test set 1" do
      tester = Ai4r3::NeuralNetwork::Backpropagation.new([3, 2])
      y = tester.eval([3, 2, 3])
      y.length.should eq 2
    end

    it "Test set 2" do
      tester = Ai4r3::NeuralNetwork::Backpropagation.new([2, 4, 8, 10, 7])
      y = tester.eval([2, 3])
      y.length.should eq 7
    end
  end

end
