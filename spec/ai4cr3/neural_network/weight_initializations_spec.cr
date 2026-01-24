require "../../spec_helper"
require "../../../src/neural_network/weight_initializations.cr"

TESTER = Ai4r3::NeuralNetwork::WeightInitializations.new

describe "Ai4r3::NeuralNetwork::WeightInitializations" do
  it "is a Ai4r3::NeuralNetwork::WeightInitializations" do
    TESTER.should be_a(Ai4r3::NeuralNetwork::WeightInitializations)
  end

  context "uniform" do
    it "tbd" do
      TESTER.uniform.should be >= -1.0
      TESTER.uniform.should be <= 1.0
    end
  end

  # context "xavier" do
  #   it "tbd" do
  #     TESTER.xavier.should be >= -1.0
  #     TESTER.xavier.should be <= 1.0
  #   end
  # end

  # context "he" do
  #   it "tbd" do
  #   end
  # end
end
