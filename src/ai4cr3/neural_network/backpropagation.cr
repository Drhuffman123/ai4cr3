# (re-)Ported from:       https://github.com/SergioFierens/ai4r

class IOException < Exception
  def initialize(@structure : Array(Int32), @inputs : Array(Float64), @outputs : Array(Float64))
  end

  def to_s(io : IO) : Nil
    io << "Wrong number of inputs and outputs. Expected (inputs): #{@structure.first}, received (inputs): #{@inputs}, Expected (outputs): #{@structure.last}, received (outputs): #{@outputs}."
  end
end

class InputsException < Exception
  def initialize(@structure : Array(Int32), @inputs : Array(Float64))
  end

  def to_s(io : IO) : Nil
    io << "Wrong number of inputs. Expected: #{@structure.first}, received: #{@inputs}."
  end
end

class OutputsException < Exception
  def initialize(@structure : Array(Int32), @inputs : Array(Float64))
  end

  def to_s(io : IO) : Nil
    io << "Wrong number of outputs. Expected: #{@structure.last}, received: #{@inputs}."
  end
end

require "yaml"

module Ai4cr3
  module NeuralNetwork
    class Backpropagation
      include YAML::Serializable
      # include Ai4r::Data::Parameterizable

      property structure : Array(Int32)
      # property activation : Array(Symbol) | Symbol # one per structure layer
      property weights : Array(Array(Array(Float64))) = [[[0.0]]]
      property activation_nodes : Array(Array(Float64)) = [[0.0]]
      property last_changes : Array(Array(Array(Float64))) = [[[0.0]]]
      property deltas : Array(Array(Float64)) = [[0.0]]
      property disable_bias = false
      property learning_rate = 0.25
      property momentum = 0.1

      # property weight_init : Symbol

      # Creates a new network specifying the its architecture.
      # E.g.
      #
      #   net = Backpropagation.new([4, 3, 2])  # 4 inputs
      #                                         # 1 hidden layer with 3 neurons,
      #                                         # 2 outputs
      #   net = Backpropagation.new([2, 3, 3, 4])   # 2 inputs
      #                                             # 2 hidden layer with 3 neurons each,
      #                                             # 4 outputs
      #   net = Backpropagation.new([2, 1])   # 2 inputs
      #                                       # No hidden layer
      #                                       # 1 output
      # @param network_structure [Object]
      # @param activation [Object]
      # # @param weight_init [Object]
      # @return [Object]
      def initialize(network_structure : Array(Int32), activation = [:sigmoid]) # , weight_init = :uniform)
        # @activation = :sigmoid # DEFAULT for now

        @structure = network_structure
        @weights = Array(Array(Array(Float64))).new
        @activation_nodes = Array(Array(Float64)).new
        @last_changes = Array(Array(Array(Float64))).new

        # @weight_init = weight_init
        @custom_propagation = false
        # @set_by_loss = true
        # @activation = activation

        # @activation_overridden = (activation != :sigmoid)
        # @set_by_loss = false
        @disable_bias = false
        # @learning_rate = 0.25
        # @momentum = 0.1
        # @loss_function = :mse
        init_network
      end

      # When the activation parameter changes, update internal lambdas for each
      # layer. Accepts a single symbol or an array of symbols (one for each
      # layer except the input layer).
      # @param symbols [Object]
      # @return [Object]
      # def activation=(symbols)
      # Other than 'sigmoid', the others are getting errors, so we'll just use 'sigmoid' (as default and only option for now)
      # def activation_param_change(symbols)
      # symbols = [symbols] unless symbols.is_a?(Array)
      # layer_count = @structure.size - 1
      # if symbols.size == 1
      #   symbols = Array.new(layer_count, symbols.first)
      # elsif symbols.size != layer_count
      #   raise ArgumentError.new("Activation array size must match number of layers (#{layer_count})")
      # end
      # @activation = symbols
      # @propagation_functions = @activation.map do |a|
      #   # Ai4r::NeuralNetwork::ActivationFunctions::FUNCTIONS[a] ||
      #   #  Ai4r::NeuralNetwork::ActivationFunctions::FUNCTIONS[:sigmoid]
      # end
      # @derivative_functions = @activation.map do |a|
      #   # Ai4r::NeuralNetwork::ActivationFunctions::DERIVATIVES[a] ||
      #   #   Ai4r::NeuralNetwork::ActivationFunctions::DERIVATIVES[:sigmoid]
      # end
      # end

      # Other than 'sigmoid', the others are getting errors, so we'll just use 'sigmoid' (as default and only option for now)
      def propagation_functions(x : Float64)
        # if @activation.first == :sigmoid
        #   1.0 / (1.0 + Math.exp(-x))
        #   # elsif @activation.first == :tanh
        #   #   Math.tanh(x)
        #   # elsif @activation.first == :relu
        #   #   [x, 0].max
        # else
        #   raise ":tanh and :relu and :softmax not supported yet, particularly #{@activation} (TODO)"
        #   # else # :softmax
        #   #   max = arr.max
        #   #   exps = arr.map { |v| Math.exp(v - max) }
        #   #   sum = exps.inject(:+)
        #   #   exps.map { |e| e / sum }
        # end
        1.0 / (1.0 + Math.exp(-x))
      end

      # Other than 'sigmoid', the others are getting errors, so we'll just use 'sigmoid' (as default and only option for now)
      def derivative_functions(y)
        # if @activation.first == :sigmoid
        #   y * (1 - y)
        #   # elsif @activation.first == :tanh
        #   #   1.0 - (y**2)
        #   # elsif @activation.first == :relu
        #   #   y.positive? ? 1.0 : 0.0
        # else
        #   raise ":tanh and :relu and :softmax not supported yet (TODO)"
        #   # else # :softmax
        #   #   y * (1 - y)
        # end
        y * (1 - y)
      end

      # Other than 'sigmoid', the others are getting errors, so we'll just use 'sigmoid' (as default and only option for now)
      # # @return [Object]
      # def activation_method
      #   if @activation.is_a?(Array)
      #     # if @activation.as(Array(Symbol)).nil?
      #     #   :sigmoid
      #     # else
      #     activ = @activation.as(Array(Symbol)).first
      #     if activ.nil?
      #       :sigmoid
      #     else
      #       activ
      #     end
      #     # end
      #     # if !(@set_by_loss || (@loss_function == :cross_entropy)) # && @activation_overridden))
      #     #   @activation
      #     # else
      #     #   @activation.first
      #     # end
      #   else
      #     @activation
      #   end
      # end

      # @param symbol [Object]
      # @return [Object]
      # def weight_init=(symbol)
      #   @weight_init = symbol
      #   # @initial_weight_function = initial_weight_function(n, i, j)
      # end

      def initial_weight_function # (n, i, j)
        # case n
        # when :xavier
        #   Ai4r::NeuralNetwork::WeightInitializations.xavier(i,j)
        # when :he
        #   Ai4r::NeuralNetwork::WeightInitializations.he(i,j)
        # else
        #   Ai4r::NeuralNetwork::WeightInitializations.uniform
        # end
        (rand * 2) - 1
      end

      # @param symbol [Object]
      # @return [Object]
      # def loss_function=(symbol)
      #   @loss_function = :mse                   # Use 'mse' as default for now. # symbol
      #   # return unless symbol == :cross_entropy # && !@activation_overridden && !@custom_propagation
      #
      #   # @set_by_loss = true
      #
      #   # @activation = :softmax # DEFAULT for now
      #   # @activation_overridden = false
      # end

      #     net.eval([25, 32.3, 12.8, 1.5])
      #         # =>  [0.83, 0.03]
      # @param input_values [Object]
      # @return [Object]
      def eval(input_values : Array(Float64))
        check_input_dimension(input_values)
        init_network unless @weights
        feedforward(input_values)
        @activation_nodes.last.clone
      end

      # Evaluates the input and returns most active node
      # E.g.
      #     net = Backpropagation.new([4, 3, 2])
      #     net.eval_result([25, 32.3, 12.8, 1.5])
      #         # eval gives [0.83, 0.03]
      #         # =>  0
      # @param input_values [Object]
      # @return [Object]
      def eval_result(input_values : Array(Float64))
        result = eval(input_values)
        result.index(result.max)
      end

      # This method trains the network using the backpropagation algorithm.
      #
      # input: Networks input
      #
      # output: Expected output for the given input.
      #
      # This method returns the training loss according to +loss_function+.
      # @param inputs [Object]
      # @param outputs [Object]
      # @return [Object]
      def train(inputs : Array(Float64), outputs : Array(Float64))
        eval(inputs)
        backpropagate(outputs)
        calculate_loss(outputs, @activation_nodes.last)
      end

      # Train a list of input/output pairs and return average loss.
      # @param batch_inputs [Object]
      # @param batch_outputs [Object]
      # @return [Object]
      def train_batch(batch_inputs : Array(Array(Float64)), batch_outputs : Array(Array(Float64)))
        if batch_inputs.size != batch_outputs.size
          # raise ArgumentError.new("Inputs and outputs size mismatch")
          raise IOException.new(@structure, batch_inputs, batch_outputs)
        end

        batch_size = batch_inputs.size
        init_network unless @weights

        accumulated_changes = Array.new(@weights.size) do |w|
          Array.new(@weights[w].size) do |i|
            Array.new(@weights[w][i].size, 0.0)
          end
        end

        sum_error = 0.0
        batch_inputs.each_index do |idx|
          inputs = batch_inputs[idx]
          outputs = batch_outputs[idx]
          eval(inputs)
          calculate_output_deltas(outputs)
          calculate_internal_deltas

          (@weights.size - 1).downto(0) do |n|
            @weights[n].each_index do |i|
              @weights[n][i].each_index do |j|
                change = @deltas[n][j] * @activation_nodes[n][i]
                accumulated_changes[n][i][j] += change
              end
            end
          end

          sum_error += calculate_loss(outputs, @activation_nodes.last)
        end

        (@weights.size - 1).downto(0) do |n|
          @weights[n].each_index do |i|
            @weights[n][i].each_index do |j|
              avg_change = accumulated_changes[n][i][j] / batch_size.to_f
              @weights[n][i][j] += (@learning_rate * avg_change) + (@momentum * @last_changes[n][i][j])
              @last_changes[n][i][j] = avg_change
            end
          end
        end

        sum_error / batch_size.to_f
      end

      # Train for a number of epochs over the dataset. Optionally define a batch size.
      # Data can be shuffled between epochs passing +shuffle: true+ (default).
      # Use +random_seed+ to make shuffling deterministic.
      # Returns an array with the average loss of each epoch.
      # @return [Object]
      def train_epochs(data_inputs : Array(Float64), data_outputs : Array(Float64), epochs : Int32, batch_size = 1,
                       early_stopping_patience = nil, min_delta = 0.0,
                       shuffle = true, random_seed : Random? = nil, &block)
        if data_inputs.size != data_outputs.size
          # raise ArgumentError.new("Inputs and outputs size mismatch")
          raise IOException.new(@structure, data_inputs, data_outputs)
        end

        losses = Array(Array(Float64)).new
        best_loss = Float::INFINITY
        patience = early_stopping_patience
        patience_counter = 0
        rng = random_seed.nil? ? Random.rand : Random.rand(random_seed)
        epochs.times do |epoch|
          epoch_error = 0.0
          epoch_inputs = data_inputs
          epoch_outputs = data_outputs
          if shuffle
            indices = (0...data_inputs.size).to_a.shuffle(random: rng)
            epoch_inputs = data_inputs.values_at(*indices)
            epoch_outputs = data_outputs.values_at(*indices)
          end
          index = 0
          while index < epoch_inputs.size
            batch_in = epoch_inputs[index, batch_size]
            batch_out = epoch_outputs[index, batch_size]
            batch_error = train_batch(batch_in, batch_out)
            epoch_error += batch_error * batch_in.size
            index += batch_size
          end
          epoch_loss = epoch_error / data_inputs.size.to_f
          losses << epoch_loss
          if block
            if block.arity >= 3
              correct = 0
              data_inputs.each_index do |i|
                output = eval(data_inputs[i])
                predicted = output.index(output.max)
                expected = data_outputs[i].index(data_outputs[i].max)
                correct += 1 if predicted == expected
              end
              accuracy = correct.to_f / data_inputs.size
              block.call(epoch, epoch_loss, accuracy)
            else
              block.call(epoch, epoch_loss)
            end
          end
          if patience
            if best_loss - epoch_loss > min_delta
              best_loss = epoch_loss
              patience_counter = 0
            else
              patience_counter += 1
              break if patience_counter >= patience
            end
          end
        end
        losses
      end

      # Initialize (or reset) activation nodes and weights, with the
      # provided net structure and parameters.
      # @return [Object]
      def init_network
        init_activation_nodes
        init_weights
        init_last_changes
        self
      end

      # # Custom serialization. It used to fail trying to serialize because
      # # it uses lambda functions internally, and they cannot be serialized.
      # # Now it does not fail, but if you customize the values of
      # # * initial_weight_function
      # # * propagation_function
      # # * derivative_propagation_function
      # # you must restore their values manually after loading the instance.
      # # @return [Object]
      # protected def marshal_dump
      #   [
      #     @structure,
      #     @disable_bias,
      #     @learning_rate,
      #     @momentum,
      #     @weights,
      #     @last_changes,
      #     @activation_nodes,
      #     @activation,
      #   ]
      # end

      # # @param ary [Object]
      # # @return [Object]
      # protected def marshal_load(ary)
      #   @structure, @disable_bias, @learning_rate, @momentum, @weights, @last_changes, @activation_nodes, @activation = ary
      #   @weight_init = :uniform
      #   @activation = @activation || :sigmoid
      # end

      # Propagate error backwards
      # @param expected_output_values [Object]
      # @return [Object]
      # protected
      def backpropagate(expected_output_values : Array(Float64))
        check_output_dimension(expected_output_values)
        calculate_output_deltas(expected_output_values)
        calculate_internal_deltas
        update_weights
      end

      # Propagate values forward
      # @param input_values [Object]
      # @return [Object]
      # protected
      def feedforward(input_values : Array(Float64))
        input_values.each_index do |input_index|
          # @activation_nodes.first[input_index] = input_values[input_index]
          @activation_nodes[1][input_index] = input_values[input_index]
        end
        @weights.each_index do |n|
          sums = Array.new(@structure[n + 1], 0.0)
          @structure[n + 1].times do |j|
            @activation_nodes[n].each_index do |i|
              sums[j] += (@activation_nodes[n][i] * @weights[n][i][j])
            end
          end
          # if @activation[n] == :softmax
          #   # TODO: values = @propagation_functions[n].call(sums)
          #   values.each_index { |j| @activation_nodes[n + 1][j] = values[j] }
          # else
          sums.each_index do |j|
            @activation_nodes[n + 1][j] = propagation_functions(sums[j]) # TODO: propagation_functions(n).call(sums[j])
          end
          # end
        end
      end

      # Initialize neurons structure.
      # @return [Object]
      # protected
      def init_activation_nodes
        @activation_nodes = Array.new(@structure.size) do |n|
          Array.new(@structure[n], 1.0)
        end
        return if @disable_bias

        @activation_nodes[0...-1].each { |layer| layer << 1.0 }
      end

      # Initialize the weight arrays using function specified with the
      # initial_weight_function parameter
      # @return [Object]
      # protected
      def init_weights
        @weights = Array.new(@structure.size - 1) do |i|
          nodes_origin = @activation_nodes[i].size
          nodes_target = @structure[i + 1]
          Array.new(nodes_origin) do |j|
            Array.new(nodes_target) do |k|
              # Ai4cr3::NeuralNetwork::WeightInitializations.initial_weight_function(i, j, k)
              initial_weight_function
            end
          end
        end
      end

      # Momentum usage need to know how much a weight changed in the
      # previous training. This method initialize the @last_changes
      # structure with 0 values.
      # @return [Object]
      # protected
      def init_last_changes
        @last_changes = Array.new(@weights.size) do |w|
          Array.new(@weights[w].size) do |i|
            Array.new(@weights[w][i].size, 0.0)
          end
        end
      end

      # Calculate deltas for output layer
      # @param expected_values [Object]
      # @return [Object]
      # protected
      def calculate_output_deltas(expected_values : Array(Float64))
        output_values = @activation_nodes.last
        output_deltas = Array(Float64).new
        # func = @derivative_functions.last
        # func = :sigmoid # derivative_functions.last
        output_values.each_index do |output_index|
          # if @loss_function == :cross_entropy && @activation == :softmax
          #   output_deltas << (output_values[output_index] - expected_values[output_index])
          # else
          error = expected_values[output_index] - output_values[output_index]
          # output_deltas << (func.call(output_values[output_index]) * error)
          output_deltas << (derivative_functions(output_values[output_index]) * error)
          # end
        end
        @deltas = [output_deltas]
      end

      # protected
      def calculate_internal_deltas_factor(weights, layer_index, j, k) : Float64
        weights[layer_index.round.to_i][j.round.to_i][k.round.to_i]
      end

      # protected
      def calculate_internal_deltas_previous(prev_deltas, k) : Float64
        prev_deltas[k.round.to_i]
      end

      def calculate_internal_deltas_structure(j, layer_index, prev_deltas)
        error = 0.0
        @structure[layer_index + 1].times do |k|
          # func = @derivative_functions[layer_index - 1]
          # layer_deltas[j] = func.call(@activation_nodes[layer_index][j]) * error
          # raise "#{derivative_functions(@activation_nodes[layer_index][j]) * error}"
          error += prev_deltas[k] * @weights[layer_index][j][k]
        end
        error
      end

      # # Calculate deltas for hidden layers
      # # @return [Object]
      # def calculate_internal_deltas
      #   prev_deltas = @deltas.last
      #   (@activation_nodes.length - 2).downto(1) do |layer_index|
      #     layer_deltas = []
      #     @activation_nodes[layer_index].each_index do |j|
      #       error = 0.0
      #       @structure[layer_index + 1].times do |k|
      #         error += prev_deltas[k] * @weights[layer_index][j][k]
      #       end
      #       func = @derivative_functions[layer_index - 1]
      #       layer_deltas[j] = func.call(@activation_nodes[layer_index][j]) * error
      #     end
      #     prev_deltas = layer_deltas
      #     @deltas.unshift(layer_deltas)
      #   end
      # end

      # Calculate deltas for hidden layers
      # @return [Object]
      def calculate_internal_deltas : Array(Array(Float64))
        prev_deltas = @deltas.last
        (@activation_nodes.size - 2).downto(1) do |layer_index|
          # layer_deltas = Array(Array(Float64)).new
          layer_deltas = Array(Float64).new
          @activation_nodes[layer_index].each_index do |j|
            error = calculate_internal_deltas_structure(j, layer_index, prev_deltas)
            # puts "error == #{error}"

            # layer_deltas[j] = derivative_functions(@activation_nodes[layer_index][j]) * error
            layer_deltas << derivative_functions(@activation_nodes[layer_index][j]) * error
            # puts "layer_deltas == #{layer_deltas}"
            # TODO: Above!!!
          end
          prev_deltas = layer_deltas
          @deltas.unshift(layer_deltas)
        end
        @deltas
      end

      # # Calculate deltas for hidden layers
      # # @return [Object]
      # protected def calculate_internal_deltas
      #   prev_deltas = @deltas.last
      #   (@activation_nodes.size - 2).downto(1) do |layer_index|
      #     # vvv TODO: This method needs MUCH review and checks!
      #     layer_deltas = Array(Array(Float64)).new
      #     @activation_nodes[layer_index].each_index do |j|
      #       error = 0.0.to_f64
      #       @structure[layer_index + 1].times do |k|
      #         factor = calculate_internal_deltas_factor(@weights, layer_index, j, k)
      #         previous = calculate_internal_deltas_previous(prev_deltas, k)
      #         error += previous * factor
      #       end
      #       # func = @derivative_functions[layer_index - 1]
      #       # func = derivative_functions(layer_index - 1)
      #       # layer_deltas[j] = func.call(@activation_nodes[layer_index][j]) * error
      #       layer_deltas[j] << derivative_functions(@activation_nodes[layer_index][j]) * error
      #     end
      #     prev_deltas = layer_deltas
      #     @deltas.unshift(layer_deltas[0])
      #     # ^^^ TODO: This method needs MUCH review and checks!
      #   end
      # end

      def calc_a_single_weight(n, i, j) # : Float64
        wb = @weights[n][i][j]
        d = 0.0
        a = 0.0
        change = 0.0
        wd = 0
        # # j: 0..119
        # j_min = j if j < j_min
        # j_max = j if j > j_max
        # raise "check type mismatch... deltas: #{@deltas.class}, activation_nodes: #{activation_nodes.class}"
        # raise "check type mismatch... deltas: #{@deltas.last.size}, activation_nodes: #{activation_nodes.last.size}"
        d = @deltas[n][j]
        if (@activation_nodes[n].size) - 1 < i
          # raise "TODO: WHY!!! Attempted to go out of bounds at i #{i} on @activation_nodes[n]: #{@activation_nodes[n]}"
          a = 0
        else
          a = @activation_nodes[n][i]
        end
        change = d * a
        # @weights[n][i][j] += ((@learning_rate * change) +
        #                       (@momentum * @last_changes[n][i][j]))
        wd = ((@learning_rate * change) +
              (@momentum * @last_changes[n][i][j]))
        # puts "WAIT! .. a: #{a}, d: #{d}, wd: #{wd}, @learning_rate: #{@learning_rate}, change: #{change}, @momentum: #{@momentum}, @last_changes[n][i][j]: #{@last_changes[n][i][j]}"
        @weights[n][i][j] += wd.to_f
        wa = @weights[n][i][j]
        # TODO: raise "WEIGHTS! wd: #{wd}; wa: #{wa}; wb: #{wb}" # if wb == wa && wd != 0.0
        @last_changes[n][i][j] = change.to_f
      rescue ex
        # puts "#{ex} #{ex.message}, n: #{n}, i: #{i}, j: #{j}, d: #{d}, a: #{a}, change: #{change}, wd: #{wd}, weights[0][3][0]: #{weights[0][3][0]}, @deltas[0][119]: #{@deltas[0][119]}; n_min: #{n_min}; n_max: #{n_max}"
        # puts "#{ex}; backtrace: #{ex.backtrace}; n: #{n}; i: #{i}; j: #{j}; a: #{a}; d: #{d}; wd: #{wd}; change: #{change}; @weights[n][i][j]: #{@weights[n][i][j]}; @last_changes[n][i][j]: #{@last_changes[n][i][j]}"
        puts "#{ex}" # TODO: WHY is this going out of bounds?
      end

      # Update weights after @deltas have been calculated.
      # @return [Object]
      # protected
      def update_weights : Array(Array(Array(Float64)))
        # n_min = 10; n_max = 0
        (@weights.size - 1).downto(0) do |n|
          # # n: 0..0
          # n_min = n if n < n_min
          # n_max = n if n > n_max
          # i_min = 10; i_max = 0
          @weights[n].each_index do |i|
            # # i: 0..3
            # i_min = i if i < i_min
            # i_max = i if i > i_max
            # j_min = 10; j_max = 0
            @weights[n][i].each_index do |j|
              calc_a_single_weight(n, i, j)
            end
          end
        end
        @weights
      end

      # Calculate quadratic error for an expected output value
      # Error = 0.5 * sum( (expected_value[i] - output_value[i])**2 )
      # @param expected_output [Object]
      # @return [Object]
      # protected
      def calculate_error(expected_output : Array(Float64))
        output_values = @activation_nodes.last
        error = 0.0
        expected_output.each_index do |output_index|
          error +=
            0.5 * ((output_values[output_index] - expected_output[output_index])**2)
        end
        error
      end

      # Calculate loss for expected/actual vectors according to selected
      # loss_function (:mse or :cross_entropy).
      # @param expected [Object]
      # @param actual [Object]
      # @return [Object]
      # protected
      def calculate_loss(expected : Array(Float64), actual : Array(Float64))
        # case @loss_function
        # when :cross_entropy
        #   epsilon = 1e-12
        #   loss = 0.0
        #   if @activation == :softmax
        #     expected.each_index do |i|
        #       p = [[actual[i], epsilon].max, 1 - epsilon].min
        #       loss -= expected[i] * Math.log(p)
        #     end
        #   else
        #     expected.each_index do |i|
        #       p = [[actual[i], epsilon].max, 1 - epsilon].min
        #       loss -= (expected[i] * Math.log(p)) + ((1 - expected[i]) * Math.log(1 - p))
        #     end
        #   end
        #   loss
        # else
        # Mean squared error
        error = 0.0
        expected.each_index do |i|
          error += 0.5 * ((expected[i] - actual[i])**2)
        end
        error
        # end
      end

      # @param inputs [Object]
      # @return [Object]
      # protected
      def check_input_dimension(inputs : Array(Float64))
        return if inputs.size == @structure.first
        raise InputsException.new(@structure, inputs)
      end

      # @param outputs [Object]
      # @return [Object]
      # protected
      def check_output_dimension(outputs : Array(Float64))
        return unless outputs.size != @structure.last
        raise OutputsException.new(@structure, outputs)
      end

      def from_file(file_path)
        yml_content = File.read(file_path)
        Ai4cr3::NeuralNetwork::Backpropagation.from_yaml(yml_content)
      end

      def save(file_path)
        File.write(file_path, self.to_yaml)
      end

      # def new(ctx : YAML::PullParser, node : YAML::Nodes::Node)
      # end

      # def to_yaml(builder : YAML::Nodes::Builder)
      # end

      # parameters_info disable_bias: 'If true, the algorithm will not use ' \
      #                              'bias nodes. False by default.',
      #                 initial_weight_function: 'f(n, i, j) must return the initial ' \
      #                                          'weight for the conection between the node i in layer n, and ' \
      #                                          'node j in layer n+1. By default a random number in [-1, 1) range.',
      #                 weight_init: 'Built-in weight initialization strategy (:uniform, :xavier or :he). Default: :uniform',
      #                 propagation_function: 'By default: ' \
      #                                       'lambda { |x| 1/(1+Math.exp(-1*(x))) }',
      #                 derivative_propagation_function: 'Derivative of the propagation ' \
      #                                                  'function, based on propagation function output. By default: ' \
      #                                                  'lambda { |y| y*(1-y) }, where y=propagation_function(x)',
      #                 activation: 'Activation function per layer. Provide a symbol or an array of symbols (:sigmoid, :tanh, :relu or :softmax). Default: :sigmoid',
      #                 learning_rate: 'By default 0.25',
      #                 momentum: 'By default 0.1. Set this parameter to 0 to disable ' \
      #                           'momentum.',
      #                 loss_function: 'Loss function used when training (:mse or ' \
      #                                ':cross_entropy). Default: :mse'
    end
  end
end
