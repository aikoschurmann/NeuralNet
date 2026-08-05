from ender.src.backend import np
from ender.src.layer import Layer
from ender.src.initializers import XavierInitializer

class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initializer=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        if initializer is None:
            initializer = XavierInitializer()
            
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros(out_channels)
        
    def forward(self, inputs, **kwargs):
        self.inputs = inputs
        batch_size, in_c, in_h, in_w = inputs.shape
        out_h = (in_h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (in_w + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        self.inputs_padded = np.pad(inputs, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        
        out = np.zeros((batch_size, self.out_channels, out_h, out_w))
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                
                slice_ = self.inputs_padded[:, :, h_start:h_end, w_start:w_end]
                out[:, :, i, j] = np.tensordot(slice_, self.W, axes=([1, 2, 3], [1, 2, 3])) + self.b
                
        return out
        
    def backward(self, output_gradient):
        batch_size, out_c, out_h, out_w = output_gradient.shape
        
        self.dW = np.zeros_like(self.W)
        self.db = np.sum(output_gradient, axis=(0, 2, 3))
        
        dinputs_padded = np.zeros_like(self.inputs_padded)
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                
                slice_ = self.inputs_padded[:, :, h_start:h_end, w_start:w_end]
                grad_pixel = output_gradient[:, :, i, j] 
                
                self.dW += np.tensordot(grad_pixel, slice_, axes=([0], [0]))
                dinputs_padded[:, :, h_start:h_end, w_start:w_end] += np.tensordot(grad_pixel, self.W, axes=([1], [0]))
                
        if self.padding > 0:
            dinputs = dinputs_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dinputs = dinputs_padded
            
        return dinputs

class MaxPooling2D(Layer):
    def __init__(self, pool_size, stride=None):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        
    def forward(self, inputs, **kwargs):
        self.inputs = inputs
        batch_size, in_c, in_h, in_w = inputs.shape
        out_h = (in_h - self.pool_size) // self.stride + 1
        out_w = (in_w - self.pool_size) // self.stride + 1
        
        out = np.zeros((batch_size, in_c, out_h, out_w))
        self.max_indices = np.zeros((batch_size, in_c, out_h, out_w, 2), dtype=int)
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                
                slice_ = inputs[:, :, h_start:h_end, w_start:w_end]
                max_val = np.max(slice_, axis=(2, 3))
                out[:, :, i, j] = max_val
                
                slice_flat = slice_.reshape(batch_size, in_c, -1)
                argmax = np.argmax(slice_flat, axis=2)
                
                max_h = argmax // self.pool_size
                max_w = argmax % self.pool_size
                
                self.max_indices[:, :, i, j, 0] = h_start + max_h
                self.max_indices[:, :, i, j, 1] = w_start + max_w
                
        return out
        
    def backward(self, output_gradient):
        batch_size, in_c, out_h, out_w = output_gradient.shape
        dinputs = np.zeros_like(self.inputs)
        
        B = np.arange(batch_size)[:, None, None, None]
        C = np.arange(in_c)[None, :, None, None]
        H = self.max_indices[:, :, :, :, 0]
        W = self.max_indices[:, :, :, :, 1]
        
        np.add.at(dinputs, (B, C, H, W), output_gradient)
        return dinputs

class FlattenLayer(Layer):
    def forward(self, inputs, **kwargs):
        self.input_shape = inputs.shape
        batch_size = inputs.shape[0]
        return inputs.reshape(batch_size, -1)
        
    def backward(self, output_gradient):
        return output_gradient.reshape(self.input_shape)
