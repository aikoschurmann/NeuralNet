import numpy as np

class Initializer:
    def initialize_weights(self, inputs: int, outputs: int):
        ...

class XavierInitializer(Initializer):
    @staticmethod
    def initialize_weights(inputs: int, outputs: int):
        stddev = np.sqrt(2.0 / (inputs + outputs))
        return np.random.normal(0, stddev, size=(inputs, outputs)).astype(np.float32)
    
class HeInitializer(Initializer):
    @staticmethod
    def initialize_weights(inputs: int, outputs: int):
        stddev = np.sqrt(2.0 / inputs)
        return np.random.normal(0, stddev, size=(inputs, outputs)).astype(np.float32)

class UniformInitializer(Initializer):
    @staticmethod
    def initialize_weights(inputs: int, outputs: int, scale=0.05):
        return np.random.uniform(-scale, scale, size=(inputs, outputs)).astype(np.float32)

class OrthogonalInitializer(Initializer):
    @staticmethod
    def initialize_weights(inputs: int, outputs: int):
        flat_shape = (inputs, outputs)
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, vh = np.linalg.svd(a, full_matrices=False)
        
        # Pick the one that has the correct shape logic, usually u unless inputs < outputs
        q = u if u.shape == flat_shape else vh
        
        # Resizing just in case SVD shapes vary by implementation
        q = q[:inputs, :outputs] 
        return q.astype(np.float32)

class RandomNormalInitializer(Initializer):
    @staticmethod
    def initialize_weights(inputs: int, outputs: int, stddev=0.05):
        return np.random.normal(0, stddev, size=(inputs, outputs)).astype(np.float32)

class RandomUniformInitializer(Initializer):
    @staticmethod
    def initialize_weights(inputs: int, outputs: int, scale=0.05):
        return np.random.uniform(-scale, scale, size=(inputs, outputs)).astype(np.float32)
