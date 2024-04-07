from ender.src.activations import *
from ender.src.initializers import *
import numpy as np

class Layer:
    """A single layer in a feedforward neural network."""

    def __init__(self, inputs : int, outputs : int, activation : ActivationFunction , initializer: Initializer = XavierInitializer):
        """Initialize the layer.

        Args:
            inputs (int): Number of input units.
            outputs (int): Number of output units.
            activation (Activation): The activation function of the layer.
            initializer (Initializer): The weight initializer object.
        """
        self.W = initializer.initialize_weights(inputs, outputs)
        self.b = np.zeros(outputs, dtype=np.float32)
        self.activation = activation
 
    def linear_forward(self, inputs):
        """Perform linear transformation in the forward pass.

        Args:
            inputs (ndarray): Input data.

        Returns:
            ndarray: Output of the linear transformation.
        """
        self.A_prev = inputs
        Z = np.dot(inputs, self.W) + self.b
        self.Z = Z
        return Z
    
    def activation_forward(self, inputs):
        """Perform activation function in the forward pass.

        Args:
            inputs (ndarray): Input data.

        Returns:
            ndarray: Output of the activation function.
        """
        return self.activation.f(self.linear_forward(inputs))
    
class DenseLayer(Layer):
    def __init__(self, inputs : int, outputs : int, activation : ActivationFunction = ReLU(), initializer: Initializer = XavierInitializer):
        super().__init__(inputs, outputs, activation, initializer)

class OutputLayer(Layer):
    def __init__(self, inputs, outputs, activation : ActivationFunction = Sigmoid(), initializer: Initializer = XavierInitializer):
        super().__init__(inputs, outputs, activation, initializer)