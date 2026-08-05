from ender.src.backend import np

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
        self.v_w = None
        self.v_b = None

    def update_parameters(self, layers):
        if self.v_w is None:
            self.v_w = [np.zeros_like(layer.W) for layer in layers]
            self.v_b = [np.zeros_like(layer.b) for layer in layers]

        for i, layer in enumerate(layers):
            self.v_w[i] = self.momentum * self.v_w[i] - self.learning_rate * layer.dW
            layer.W += self.v_w[i]
            
            self.v_b[i] = self.momentum * self.v_b[i] - self.learning_rate * layer.db
            layer.b += self.v_b[i]


class Adagrad(Optimizer):
    def __init__(self, learning_rate, epsilon=1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.cache_w = None
        self.cache_b = None

    def update_parameters(self, layers):
        if self.cache_w is None:
            self.cache_w = [np.zeros_like(layer.W) for layer in layers]
            self.cache_b = [np.zeros_like(layer.b) for layer in layers]

        for i, layer in enumerate(layers):
            self.cache_w[i] += layer.dW ** 2
            layer.W -= (self.learning_rate / (np.sqrt(self.cache_w[i]) + self.epsilon)) * layer.dW
            
            self.cache_b[i] += layer.db ** 2
            layer.b -= (self.learning_rate / (np.sqrt(self.cache_b[i]) + self.epsilon)) * layer.db


class RMSprop(Optimizer):
    def __init__(self, learning_rate, decay_rate=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache_w = None
        self.cache_b = None

    def update_parameters(self, layers):
        if self.cache_w is None:
            self.cache_w = [np.zeros_like(layer.W) for layer in layers]
            self.cache_b = [np.zeros_like(layer.b) for layer in layers]

        for i, layer in enumerate(layers):
            self.cache_w[i] = self.decay_rate * self.cache_w[i] + (1 - self.decay_rate) * (layer.dW ** 2)
            layer.W -= (self.learning_rate / (np.sqrt(self.cache_w[i]) + self.epsilon)) * layer.dW
            
            self.cache_b[i] = self.decay_rate * self.cache_b[i] + (1 - self.decay_rate) * (layer.db ** 2)
            layer.b -= (self.learning_rate / (np.sqrt(self.cache_b[i]) + self.epsilon)) * layer.db


class Adam(Optimizer):
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_w = None
        self.v_w = None
        self.m_b = None
        self.v_b = None
        self.t = 0

    def update_parameters(self, layers):
        if self.m_w is None:
            self.m_w = [np.zeros_like(layer.W) for layer in layers]
            self.v_w = [np.zeros_like(layer.W) for layer in layers]
            self.m_b = [np.zeros_like(layer.b) for layer in layers]
            self.v_b = [np.zeros_like(layer.b) for layer in layers]

        self.t += 1
        for i, layer in enumerate(layers):
            # Weights
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * layer.dW
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (layer.dW ** 2)
            m_hat_w = self.m_w[i] / (1 - self.beta1 ** self.t)
            v_hat_w = self.v_w[i] / (1 - self.beta2 ** self.t)
            layer.W -= (self.learning_rate / (np.sqrt(v_hat_w) + self.epsilon)) * m_hat_w
            
            # Biases
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * layer.db
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (layer.db ** 2)
            m_hat_b = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_hat_b = self.v_b[i] / (1 - self.beta2 ** self.t)
            layer.b -= (self.learning_rate / (np.sqrt(v_hat_b) + self.epsilon)) * m_hat_b
