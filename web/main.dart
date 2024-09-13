import 'dart:html';
import 'package:dartml/dartml.dart';

void main() {
  var outputDiv = querySelector('#output');

  // Create a neural network with 2 input neurons, 3 hidden neurons, and 1 output neuron
  var nn = NeuralNetwork([2, 3, 1]);

  // Training data for XOR problem
  var trainingData = [
    {
      'input': [0, 0],
      'output': [0]
    },
    {
      'input': [0, 1],
      'output': [1]
    },
    {
      'input': [1, 0],
      'output': [1]
    },
    {
      'input': [1, 1],
      'output': [0]
    },
  ];

  // Train the network
  outputDiv!.appendText('Training...\n');
  for (int i = 0; i < 10000; i++) {
    for (var data in trainingData) {
      nn.train(
          // TODO: Fix this line of code to match the new API of the train method in the NeuralNetwork class
          [data['input'] as List<double>],
          [data['output'] as List<double>],
          1);
    }
  }

  // Test the network
  outputDiv.appendText('Testing...\n');
  for (var data in trainingData) {
    var input = data['input'] as List<double>;
    var output = nn.feedForward(input);
    outputDiv.appendText(
        'Input: $input, Predicted Output: ${output[0].toStringAsFixed(4)}, Expected Output: ${data['output']![0]}\n');
  }
}
