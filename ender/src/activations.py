from ender.src.backend import np

class ActivationFunction:
    def f(self, z):
        raise NotImplementedError("Subclasses must implement f method.")

    def derivative(self, z):
        raise NotImplementedError("Subclasses must implement derivative method.")

class ReLU(ActivationFunction):
    def f(self, z):
        return np.maximum(0, z)

    def derivative(self, z):
        return np.where(z < 0, 0, 1)

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
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def derivative(self, z):
        s = self.f(z)
        return s * (1 - s)

class SoftPlus(ActivationFunction):
    def f(self, z):
        return np.where(z > 20, z, np.log(1 + np.exp(np.clip(z, -500, 20))))

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


class Softmax(ActivationFunction):
    def f(self, z):
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    def derivative(self, z):
        s = self.f(z)
        # For a vector input, the Jacobian matrix is returned
        return s * (1 - s)

class Swish(ActivationFunction):
    def f(self, z):
        return z / (1 + np.exp(-z))

    def derivative(self, z):
        sigmoid = 1 / (1 + np.exp(-z))
        swish_val = z * sigmoid
        return sigmoid + swish_val * (1 - sigmoid) 
