import numpy as np

class Initializer:
    def initialize_weights(self, inputs: int, outputs: int):
        ...

class XavierInitializer(Initializer):
    @staticmethod
    def initialize_weights(inputs: int, outputs: int):
        stddev = np.sqrt(2.0 / (inputs + outputs))
        return np.random.normal(0, stddev, size=(inputs, outputs)).astype(np.float32)
    
class HeInitializer:
    @staticmethod
    def initialize_weights(inputs: int, outputs: int):
        stddev = np.sqrt(2.0 / inputs)
        return np.random.normal(0, stddev, size=(inputs, outputs)).astype(np.float32)

class UniformInitializer:
    @staticmethod
    def initialize_weights(inputs: int, outputs: int, scale=0.05):
        return np.random.uniform(-scale, scale, size=(inputs, outputs)).astype(np.float32)

class OrthogonalInitializer:
    @staticmethod
    def initialize_weights(inputs: int, outputs: int):
        flat_shape = (inputs, outputs) if inputs < outputs else (outputs, inputs)
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape((inputs, outputs))
        return q.astype(np.float32)

class RandomNormalInitializer:
    @staticmethod
    def initialize_weights(inputs: int, outputs: int, stddev=0.05):
        return np.random.normal(0, stddev, size=(inputs, outputs)).astype(np.float32)

class RandomUniformInitializer:
    @staticmethod
    def initialize_weights(inputs: int, outputs: int, scale=0.05):
        return np.random.uniform(-scale, scale, size=(inputs, outputs)).astype(np.float32)
