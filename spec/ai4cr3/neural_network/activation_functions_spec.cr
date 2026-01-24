require "../../spec_helper"
require "../../../src/neural_network/activation_functions.cr"

describe "Ai4cr3::NeuralNetwork::ActivationFunctions" do
  it "the module" do
    tester = Ai4r3::NeuralNetwork::ActivationFunctions
    tester.should eq(Ai4r3::NeuralNetwork::ActivationFunctions)
  end

  context "FUNCTIONS" do
    context "sigmoid" do
      it "case 1.0" do
        given = 1.0
        expected = 0.7310585786300049
        tester = Ai4r3::NeuralNetwork::ActivationFunctions.new
        tester.sigmoid(given).should eq(expected)
      end

      it "case 0.5" do
        given = 0.5
        expected = 0.6224593312018546
        tester = Ai4r3::NeuralNetwork::ActivationFunctions.new
        tester.sigmoid(given).should eq(expected)
      end

      it "case 0.0" do
        given = 0.0
        expected = 0.5
        tester = Ai4r3::NeuralNetwork::ActivationFunctions.new
        tester.sigmoid(given).should eq(expected)
      end
    end

    context "tanh" do
      it "case 1.0" do
        given = 1.0
        expected = 0.7615941559557649
        tester = Ai4r3::NeuralNetwork::ActivationFunctions.new
        tester.tanh(given).should eq(expected)
      end

      it "case 0.5" do
        given = 0.5
        expected = 0.46211715726000974
        tester = Ai4r3::NeuralNetwork::ActivationFunctions.new
        tester.tanh(given).should eq(expected)
      end

      it "case 0.0" do
        given = 0.0
        expected = 0.0
        tester = Ai4r3::NeuralNetwork::ActivationFunctions.new
        tester.tanh(given).should eq(expected)
      end
    end

    context "relu" do
      it "case 1.0" do
        given = 1.0
        expected = 1.0
        tester = Ai4r3::NeuralNetwork::ActivationFunctions.new
        tester.relu(given).should eq(expected)
      end

      it "case 0.5" do
        given = 0.5
        expected = 0.5
        tester = Ai4r3::NeuralNetwork::ActivationFunctions.new
        tester.relu(given).should eq(expected)
      end

      it "case 0.0" do
        given = 0.0
        expected = 0.0
        tester = Ai4r3::NeuralNetwork::ActivationFunctions.new
        tester.relu(given).should eq(expected)
      end
    end


    context "softmax" do
      it "case 1.0" do
        given = 1.0
        expected = 0.0
        tester = Ai4r3::NeuralNetwork::ActivationFunctions.new
        tester.softmax(given).should eq(expected)
      end

      it "case 0.5" do
        given = 0.5
        expected = 0.25
        tester = Ai4r3::NeuralNetwork::ActivationFunctions.new
        tester.softmax(given).should eq(expected)
      end

      it "case 0.0" do
        given = 0.0
        expected = 0.0
        tester = Ai4r3::NeuralNetwork::ActivationFunctions.new
        tester.softmax(given).should eq(expected)
      end
    end
  end
end

