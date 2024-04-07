import numpy as np

class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update_parameters(self, layers):
        raise NotImplementedError("Subclasses must implement the update method.")


class SGD(Optimizer):
    def update_parameters(self, layers):
        for layer in layers:
            layer.W -= self.learning_rate * layer.dW
            layer.b -= self.learning_rate * layer.db


class Momentum(Optimizer):
    def __init__(self, learning_rate, momentum=0.4):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.v = None

    def update_parameters(self, layers):
        if self.v is None:
            self.v = [np.zeros_like(layer.W) for layer in layers]

        for i, layer in enumerate(layers):
            self.v[i] = self.momentum * self.v[i] - self.learning_rate * layer.dW
            layer.W += self.v[i]
            layer.b -= self.learning_rate * layer.db


class Adagrad(Optimizer):
    def __init__(self, learning_rate, epsilon=1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.cache = None

    def update_parameters(self, layers):
        if self.cache is None:
            self.cache = [np.zeros_like(layer.W) for layer in layers]

        for i, layer in enumerate(layers):
            self.cache[i] += layer.dW ** 2
            layer.W -= (self.learning_rate / (np.sqrt(self.cache[i]) + self.epsilon)) * layer.dW
            layer.b -= self.learning_rate * layer.db


class RMSprop(Optimizer):
    def __init__(self, learning_rate, decay_rate=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = None

    def update_parameters(self, layers):
        if self.cache is None:
            self.cache = [np.zeros_like(layer.W) for layer in layers]

        for i, layer in enumerate(layers):
            self.cache[i] = self.decay_rate * self.cache[i] + (1 - self.decay_rate) * (layer.dW ** 2)
            layer.W -= (self.learning_rate / (np.sqrt(self.cache[i]) + self.epsilon)) * layer.dW
            layer.b -= self.learning_rate * layer.db


class Adam(Optimizer):
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update_parameters(self, layers):
        if self.m is None:
            self.m = [np.zeros_like(layer.W) for layer in layers]
            self.v = [np.zeros_like(layer.W) for layer in layers]

        self.t += 1
        for i, layer in enumerate(layers):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * layer.dW
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (layer.dW ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            layer.W -= (self.learning_rate / (np.sqrt(v_hat) + self.epsilon)) * m_hat
            layer.b -= self.learning_rate * layer.db
