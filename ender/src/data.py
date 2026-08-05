from ender.src.backend import np
import math

class DataLoader:
    def __init__(self, X, y, batch_size, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        n_samples = len(self.X)
        indices = np.arange(n_samples)
        
        if self.shuffle:
            np.random.shuffle(indices)
            
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            yield self.X[batch_indices], self.y[batch_indices]

    def __len__(self):
        return math.ceil(len(self.X) / self.batch_size)
