from ender.src.activations import *
from ender.src.initializers import *
import numpy as np

class Layer:
    """A single layer in a feedforward neural network."""
    def forward(self, inputs, **kwargs):
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
        
    def forward(self, inputs, **kwargs):
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
        
    def forward(self, inputs, **kwargs):
        self.inputs = inputs
        return self.activation.f(inputs)
        
    def backward(self, output_gradient):
        return self.activation.derivative(self.inputs) * output_gradient

class DropoutLayer(Layer):
    def __init__(self, rate=0.5):
        self.rate = rate
        self.mask = None
        
    def forward(self, inputs, training=False, **kwargs):
        if training:
            self.mask = np.random.binomial(1, 1.0 - self.rate, size=inputs.shape) / (1.0 - self.rate)
            return inputs * self.mask
        else:
            return inputs
            
    def backward(self, output_gradient):
        return output_gradient * self.mask

class BatchNormalizationLayer(Layer):
    def __init__(self, input_dim, momentum=0.9, epsilon=1e-5):
        self.W = np.ones(input_dim)
        self.b = np.zeros(input_dim)
        self.running_mean = np.zeros(input_dim)
        self.running_var = np.ones(input_dim)
        self.momentum = momentum
        self.epsilon = epsilon
        
    def forward(self, inputs, training=False, **kwargs):
        if training:
            mean = np.mean(inputs, axis=0)
            var = np.var(inputs, axis=0)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            
            self.x_centered = inputs - mean
            self.stddev_inv = 1.0 / np.sqrt(var + self.epsilon)
            self.x_norm = self.x_centered * self.stddev_inv
            return self.W * self.x_norm + self.b
        else:
            x_norm = (inputs - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            return self.W * x_norm + self.b
            
    def backward(self, output_gradient):
        N = output_gradient.shape[0]
        
        self.dW = np.sum(output_gradient * self.x_norm, axis=0)
        self.db = np.sum(output_gradient, axis=0)
        
        dx_norm = output_gradient * self.W
        dvar = np.sum(dx_norm * self.x_centered * -0.5 * (self.stddev_inv ** 3), axis=0)
        dmean = np.sum(dx_norm * -self.stddev_inv, axis=0) + dvar * np.mean(-2.0 * self.x_centered, axis=0)
        
        dx = (dx_norm * self.stddev_inv) + (dvar * 2.0 * self.x_centered / N) + (dmean / N)
        return dx
