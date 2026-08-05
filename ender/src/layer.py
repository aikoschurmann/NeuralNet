from ender.src.activations import *
from ender.src.initializers import *
import numpy as np

class Layer:
    """A single layer in a feedforward neural network."""
    def forward(self, inputs):
        raise NotImplementedError("Layer must implement forward pass")
        
    def backward(self, output_gradient):
        raise NotImplementedError("Layer must implement backward pass")

class DenseLayer(Layer):
    def __init__(self, inputs, outputs, initializer: Initializer = None):
        """Initialize the Dense layer.
        
        Args:
            inputs (int): Number of input units.
            outputs (int): Number of output units.
            initializer (Initializer): The weight initializer object.
        """
        if initializer is None:
            initializer = XavierInitializer()
        self.W = initializer.initialize_weights(inputs, outputs)
        self.b = np.zeros(outputs, dtype=np.float32)
        
    def forward(self, inputs):
        """Perform linear transformation."""
        self.inputs = inputs
        return np.dot(inputs, self.W) + self.b
        
    def backward(self, output_gradient):
        """Compute gradients for weights and biases, and return input gradient."""
        self.dW = np.dot(self.inputs.T, output_gradient)
        self.db = np.sum(output_gradient, axis=0)
        return np.dot(output_gradient, self.W.T)

class ActivationLayer(Layer):
    """Wrapper layer that applies an ActivationFunction."""
    def __init__(self, activation: ActivationFunction):
        self.activation = activation
        
    def forward(self, inputs):
        self.inputs = inputs
        return self.activation.f(inputs)
        
    def backward(self, output_gradient):
        return self.activation.derivative(self.inputs) * output_gradient
