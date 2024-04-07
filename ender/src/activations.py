import numpy as np

class ActivationFunction:
    def f(self, z):
        raise NotImplementedError("Subclasses must implement f method.")

    def derivative(self, z):
        raise NotImplementedError("Subclasses must implement derivative method.")

class ReLU(ActivationFunction):
    def f(self, z):
        return np.maximum(0, z)

    def derivative(self, z):
        return np.where(z <= 0, 0, 1)

class LeakyReLU(ActivationFunction):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def f(self, z):
        return np.where(z < 0, self.alpha * z, z)

    def derivative(self, z):
        return np.where(z < 0, self.alpha, 1)

class ELU(ActivationFunction):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def f(self, z):
        return np.where(z > 0, z, self.alpha * (np.exp(z) - 1))

    def derivative(self, z):
        return np.where(z > 0, 1, self.alpha * np.exp(z))

class Sigmoid(ActivationFunction):
    def f(self, z):
        return 1 / (1 + np.exp(-z))

    def derivative(self, z):
        return self.f(z) * (1 - self.f(z))

class SoftPlus(ActivationFunction):
    def f(self, z):
        return np.log(1 + np.exp(z))

    def derivative(self, z):
        return 1 / (1 + np.exp(-z))

class TanH(ActivationFunction):
    def f(self, z):
        return np.tanh(z)

    def derivative(self, z):
        return 1 - np.tanh(z) ** 2

class Arctan(ActivationFunction):
    def f(self, z):
        return np.arctan(z)

    def derivative(self, z):
        return 1 / (1 + z ** 2)

class Swish(ActivationFunction):
    def f(self, z):
        return z / (1 + np.exp(-z))

    def derivative(self, z):
        sigmoid = 1 / (1 + np.exp(-z))
        return z * sigmoid * (1 - sigmoid)
