import 'dart:math';

class NeuralNetwork {
  late List<int> layers;
  late List<List<List<double>>> weights;
  late List<List<double>> biases;
  double learningRate;

  NeuralNetwork(this.layers, {this.learningRate = 0.1}) {
    _initializeWeightsAndBiases();
  }

  // Initialize the weights and biases
  // The weights and biases are initialized with random values between -1 and 1
  void _initializeWeightsAndBiases() {
    weights = [];
    biases = [];
    Random random = Random();

    for (int i = 1; i < layers.length; i++) {
      List<List<double>> layerWeights = [];
      List<double> layerBiases = [];

      for (int j = 0; j < layers[i]; j++) {
        List<double> neuronWeights = [];
        for (int k = 0; k < layers[i - 1]; k++) {
          neuronWeights.add(random.nextDouble() * 2 - 1);
        }
        layerWeights.add(neuronWeights);
        layerBiases.add(random.nextDouble() * 2 - 1);
      }

      weights.add(layerWeights);
      biases.add(layerBiases);
    }
  }

  // Feedforward
  // The feedforward algorithm is used to calculate the output of the neural network
  List<double> feedForward(List<double> inputs) {
    List<double> activations = inputs;

    for (int i = 0; i < weights.length; i++) {
      List<double> newActivations = [];
      for (int j = 0; j < weights[i].length; j++) {
        double sum = biases[i][j];
        for (int k = 0; k < weights[i][j].length; k++) {
          sum += weights[i][j][k] * activations[k];
        }
        newActivations.add(_sigmoid(sum));
      }
      activations = newActivations;
    }

    return activations;
  }

  void train(
      List<List<double>> inputs, List<List<double>> targets, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
      for (int i = 0; i < inputs.length; i++) {
        List<double> output = feedForward(inputs[i]);
        List<List<double>> deltas = _backpropagate(output, targets[i]);
        _updateWeightsAndBiases(inputs[i], deltas);
      }
    }
  }

  // Backpropagation
  // The backpropagation algorithm is used to calculate the error and update the weights and biases
  List<List<double>> _backpropagate(List<double> output, List<double> target) {
    List<List<double>> deltas = [];
    List<double> error = [];

    for (int i = 0; i < output.length; i++) {
      error.add(target[i] - output[i]);
    }

    for (int i = weights.length - 1; i >= 0; i--) {
      List<double> layerDelta = [];

      if (i == weights.length - 1) {
        for (int j = 0; j < weights[i].length; j++) {
          layerDelta.add(error[j] * _sigmoidDerivative(output[j]));
        }
      } else {
        for (int j = 0; j < weights[i].length; j++) {
          double sum = 0;
          for (int k = 0; k < weights[i + 1].length; k++) {
            sum += weights[i + 1][k][j] * deltas[0][k];
          }
          layerDelta.add(sum * _sigmoidDerivative(output[j]));
        }
      }

      deltas.insert(0, layerDelta);
      output = feedForward(List<double>.filled(weights[i][0].length, 0));
    }

    return deltas;
  }

  // Update the weights and biases
  // The weights and biases are updated using the deltas calculated in the backpropagation step
  void _updateWeightsAndBiases(List<double> input, List<List<double>> deltas) {
    List<double> activations = input;

    for (int i = 0; i < weights.length; i++) {
      for (int j = 0; j < weights[i].length; j++) {
        for (int k = 0; k < weights[i][j].length; k++) {
          weights[i][j][k] += learningRate * deltas[i][j] * activations[k];
        }
        biases[i][j] += learningRate * deltas[i][j];
      }
      activations = feedForward(activations);
    }
  }

  // Sigmoid function
  // The sigmoid function squashes the input value between 0 and 1
  double _sigmoid(double x) {
    return 1 / (1 + exp(-x));
  }

  // Derivative of the sigmoid function
  // The derivative of the sigmoid function is used to calculate the error
  double _sigmoidDerivative(double x) {
    return x * (1 - x);
  }
}
