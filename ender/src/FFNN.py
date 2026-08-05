import numpy as np
import math
from enum import Enum
from ender.src.layer import Layer
from ender.src.optimizers import Optimizer, Adam
from ender.src.losses import LossFunction, MeanSquaredErrorLoss
from tqdm import tqdm
import os

class RegularizationType(Enum):
    L1 = 'l1'
    L2 = 'l2'

class FFNN:
    def __init__(self, layers: list[Layer] = None, regularization: RegularizationType = RegularizationType.L2, lmbda=0.01, optimizer: Optimizer = Adam, learning_rate=0.01, loss_function: LossFunction = MeanSquaredErrorLoss):
        self.layers = layers if layers is not None else []
        self.learning_rate = learning_rate
        self.optimizer = optimizer(learning_rate=learning_rate)
        self.regularization = regularization
        self.lmbda = lmbda
        self.loss_function = loss_function

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def add_layer(self, layer: Layer):
        self.layers.append(layer)
    
    def backward(self, loss_gradient):
        for layer in reversed(self.layers):
            loss_gradient = layer.backward(loss_gradient)
            
            # Apply regularization if the layer has weights
            if hasattr(layer, 'W') and hasattr(layer, 'dW'):
                if self.regularization == RegularizationType.L1:
                    layer.dW += self.lmbda * np.sign(layer.W)
                elif self.regularization == RegularizationType.L2:
                    layer.dW += self.lmbda * layer.W

    def update_parameters(self):
        # Pass only layers that have weights to the optimizer
        trainable_layers = [layer for layer in self.layers if hasattr(layer, 'W')]
        self.optimizer.update_parameters(trainable_layers)

    def summary(self):
        print("Neural Network Summary:")
        print("======================")
        print("Architecture:")
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'W'):
                print(f"Layer {i+1}: {layer.__class__.__name__} - Input: {layer.W.shape[0]}, Output: {layer.W.shape[1]}")
            elif hasattr(layer, 'activation'):
                print(f"Layer {i+1}: {layer.__class__.__name__} - {layer.activation.__class__.__name__}")
            else:
                print(f"Layer {i+1}: {layer.__class__.__name__}")
        print("----------------------")
        print(f"Total Parameters: {self.total_parameters()}")
        print(f"Optimizer: {self.optimizer.__class__.__name__}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Loss Function: {self.loss_function.__class__.__name__}")
        print(f"Regularization: {self.regularization}")
        print(f"Regularization Parameter (lambda): {self.lmbda}")
        print("======================")
    
    def total_parameters(self):
        return sum(layer.W.size + layer.b.size for layer in self.layers if hasattr(layer, 'W'))

    def train(self, X_train, y_train, epochs: int = 25, batch_size: int = 32, validation_data=None, callbacks=None):
        if callbacks is None:
            callbacks = []

        train_losses = []
        val_losses = []
        val_accuracies = []

        X_val, y_val = validation_data if validation_data is not None else (None, None)

        for cb in callbacks:
            cb.on_train_begin()

        for epoch in tqdm(range(epochs), desc='Epochs'):
            for cb in callbacks:
                cb.on_epoch_begin(epoch)

            # Shuffle training data
            indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            total_loss = 0.0

            for i in tqdm(range(0, len(X_train), batch_size), desc='Batches', leave=False):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]

                output = self.forward(X_batch)
                loss = self.loss_function.loss(y_batch, output)
                total_loss += loss

                loss_gradient = self.loss_function.derivative(y_batch, output)
                self.backward(loss_gradient)
                self.update_parameters()

            num_batches = math.ceil(len(X_train) / batch_size)
            train_loss = total_loss / num_batches
            train_losses.append(train_loss)

            logs = {'loss': train_loss}

            if X_val is not None and y_val is not None:
                total_val_loss = 0.0
                correct_predictions = 0

                for i in range(0, len(X_val), batch_size):
                    X_val_batch = X_val[i:i+batch_size]
                    y_val_batch = y_val[i:i+batch_size]

                    val_output = self.forward(X_val_batch)
                    val_loss = self.loss_function.loss(y_val_batch, val_output)
                    total_val_loss += val_loss

                    predicted_labels = np.argmax(val_output, axis=1)
                    true_labels = np.argmax(y_val_batch, axis=1)
                    correct_predictions += np.sum(predicted_labels == true_labels)

                num_val_batches = math.ceil(len(X_val) / batch_size)
                val_loss = total_val_loss / num_val_batches
                val_accuracy = correct_predictions / len(X_val)

                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
                logs['val_loss'] = val_loss
                logs['val_accuracy'] = val_accuracy

                tqdm.write(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2%}")
            else:
                tqdm.write(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")

            for cb in callbacks:
                cb.on_epoch_end(epoch, logs)
            
            if any(getattr(cb, 'stop_training', False) for cb in callbacks):
                break

        for cb in callbacks:
            cb.on_train_end()

        return train_losses, val_losses, val_accuracies

    def test(self, X_test, y_test, batch_size):
        total_loss = 0.0
        correct_predictions = 0

        for i in range(0, len(X_test), batch_size):
            output = self.forward(X_test[i:i+batch_size])
            loss = self.loss_function.loss(y_test[i:i+batch_size], output)
            total_loss += loss

            predicted_labels = np.argmax(output, axis=1)
            true_labels = np.argmax(y_test[i:i+batch_size], axis=1)
            correct_predictions += np.sum(predicted_labels == true_labels)

        num_test_batches = math.ceil(len(X_test) / batch_size)
        test_loss = total_loss / num_test_batches
        test_accuracy = correct_predictions / len(X_test)
        return test_loss, test_accuracy

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)
    
    def save(self, file_path):
        weights = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'W'):
                weights[f"W_{i}"] = layer.W
                weights[f"b_{i}"] = layer.b
        np.savez(file_path, **weights)

    def load(self, file_path):
        data = np.load(file_path if file_path.endswith('.npz') else file_path + '.npz')
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'W'):
                layer.W = data[f"W_{i}"]
                layer.b = data[f"b_{i}"]
