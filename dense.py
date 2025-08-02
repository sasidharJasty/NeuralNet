from layer import Layer
import numpy as np


class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        # Sum across samples (axis=1) and keep dimensions for proper broadcasting
        bias_gradient = np.sum(output_gradient, axis=1, keepdims=True)
        self.bias -= learning_rate * bias_gradient
        return input_gradient
    
