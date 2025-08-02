from layer import Layer
import numpy as np


class DenseAdam(Layer):
    def __init__(self, input_size, output_size, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        
        # Adam optimizer parameters
        self.alpha = alpha      # Learning rate
        self.beta1 = beta1      # Decay rate for first moment
        self.beta2 = beta2      # Decay rate for second moment
        self.epsilon = epsilon  # Small constant to prevent division by zero
        
        # Initialize moment estimates
        self.m_w = np.zeros_like(self.weights)  # First moment for weights
        self.v_w = np.zeros_like(self.weights)  # Second moment for weights
        self.m_b = np.zeros_like(self.bias)     # First moment for bias
        self.v_b = np.zeros_like(self.bias)     # Second moment for bias
        
        # Time step counter
        self.t = 0

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate=None):
        # Note: learning_rate parameter is ignored since Adam uses its own alpha
        self.t += 1  # Increment time step
        
        # Compute gradients
        weights_gradient = np.dot(output_gradient, self.input.T)
        bias_gradient = np.sum(output_gradient, axis=1, keepdims=True)
        
        # Update first moment estimates (momentum)
        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * weights_gradient
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * bias_gradient
        
        # Update second moment estimates (squared gradients)
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (weights_gradient ** 2)
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (bias_gradient ** 2)
        
        # Bias correction
        m_w_corrected = self.m_w / (1 - self.beta1 ** self.t)
        m_b_corrected = self.m_b / (1 - self.beta1 ** self.t)
        v_w_corrected = self.v_w / (1 - self.beta2 ** self.t)
        v_b_corrected = self.v_b / (1 - self.beta2 ** self.t)
        
        # Update weights and bias using Adam
        self.weights -= self.alpha * m_w_corrected / (np.sqrt(v_w_corrected) + self.epsilon)
        self.bias -= self.alpha * m_b_corrected / (np.sqrt(v_b_corrected) + self.epsilon)
        
        # Return input gradient for backpropagation
        input_gradient = np.dot(self.weights.T, output_gradient)
        return input_gradient