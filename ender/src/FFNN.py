import numpy as np
from enum import Enum
from ender.src.layer import *
from ender.src.optimizers import *
from ender.src.losses import *
from ender.src.schedulers import *
import pickle
from tqdm import tqdm



class regularizationType(Enum):
    """Enum class for different types of regularization."""
    L1 = 'l1'
    L2 = 'l2'

class FFNN:
    """Feedforward Neural Network."""

    def __init__(self, layers : list[Layer] = [], regularization : regularizationType = regularizationType.L2, lmbda=0.01, optimizer : Optimizer = Adam, learning_rate=0.01, loss_function: LossFunction = MeanSquaredErrorLoss):
        """Initialize the feedforward neural network.

        Args:
            layers (list[Layer]): List of layers in the network.
            regularization (str): The type of regularization to use.
            lmbda (float): Regularization parameter.
            optimizer (OptimizerType): The type of optimizer to use.
            learning_rate (float): Learning rate for optimization.
        """
        self.layers = layers
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer
        self.regularization = regularization
        self.lmbda = lmbda
        self.optimizer = optimizer
        self.lossFunction = loss_function

    def forward(self, X):
        """Perform forward pass through the network.

        Args:
            X (ndarray): Input data.

        Returns:
            ndarray: Output of the network.
        """
        output = X
        for layer in self.layers:            
            output = layer.activation_forward(output)
        return output
    
    def add_layer(self, layer : Layer):
        """Add a layer to the network.

        Args:
            layer (Layer): Layer to be added.
        """
        self.layers.append(layer)
    
    def backward(self, loss_gradient):
        """Perform backward pass through the network.

        Args:
            loss_gradient (ndarray): Gradient of the loss with respect to the output.
        """
        for i in range(len(self.layers) - 1, -1, -1):
            loss_gradient = self.layers[i].activation.derivative(self.layers[i].Z) * loss_gradient
            dW = np.dot(self.layers[i].A_prev.T, loss_gradient)
            db = np.sum(loss_gradient, axis=0)
            loss_gradient = np.dot(loss_gradient, self.layers[i].W.T)

            if self.regularization == regularizationType.L1:
                dW += self.lmbda * np.sign(self.layers[i].W)
            elif self.regularization == regularizationType.L2:
                dW += self.lmbda * self.layers[i].W 
  
            self.layers[i].dW = dW
            self.layers[i].db = db

    def update_parameters(self):
        """Update parameters using the optimizer."""
        self.optimizer.update_parameters(self.layers)

    def summary(self):
        """Print a summary of the neural network."""
        print("Neural Network Summary:")
        print("======================")
        print("Architecture:")
        for i, layer in enumerate(self.layers):
            print(f"Layer {i+1}: {layer.__class__.__name__} - Input Shape: {layer.W.shape[0]}, Output Shape: {layer.W.shape[1]}")
        print("----------------------")
        print(f"Total Parameters: {self.total_parameters()}")
        print(f"Optimizer: {self.optimizer.__class__.__name__}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Loss Function: {self.lossFunction.__class__.__name__}")
        print(f"Regularization: {self.regularization}")
        print(f"Regularization Parameter (lambda): {self.lmbda}")
        print("======================")
    
    def total_parameters(self):
        """Compute the total number of parameters in the network."""
        total_parameters = 0
        for layer in self.layers:
            layer_parameters = layer.W.size + layer.b.size
            total_parameters += layer_parameters
        return total_parameters

    
    def train(self, X_train, y_train, epochs: int = 25, batch_size: int = 32, validation_data=None, lr_scheduler: LearningRateSchedulerBase = ReduceLROnPlateauScheduler(initial_lr=0.01, factor=0.9, patience=5)):
        """Train the neural network.

        Args:
            X_train (ndarray): Training data.
            y_train (ndarray): Training labels.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            validation_data (tuple): Validation data as a tuple (X_val, y_val).
            lr_scheduler (LearningRateSchedulerBase): Learning rate scheduler.

        Returns:
            tuple: Training losses, validation losses, validation accuracies.
        """
        train_losses = []
        val_losses = []
        val_accuracies = []

        X_val, y_val = validation_data if validation_data is not None else (None, None)

        for epoch in tqdm(range(epochs), desc='Epochs'):
            # Shuffle training data
            indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            total_loss = 0.0

            # Mini-batch training
            for i in tqdm(range(0, len(X_train), batch_size), desc='Batches', leave=False):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]

                # Forward pass
                output = self.forward(X_batch)

                # Calculate loss
                loss = self.lossFunction.loss(y_batch, output)
                total_loss += loss

                # Backward pass
                loss_gradient = self.lossFunction.derivative(y_batch, output)
                self.backward(loss_gradient)

                # Update parameters using optimizer
                self.update_parameters()

            # Calculate average training loss for the epoch
            train_loss = total_loss / len(X_train)
            train_losses.append(train_loss)

            # Validation loss and accuracy
            if X_val is not None and y_val is not None:
                total_val_loss = 0.0
                correct_predictions = 0

                # Evaluate on validation data
                for i in range(0, len(X_val), batch_size):
                    X_val_batch = X_val[i:i+batch_size]
                    y_val_batch = y_val[i:i+batch_size]

                    val_output = self.forward(X_val_batch)
                    val_loss = self.lossFunction.loss(y_val_batch, val_output)
                    total_val_loss += val_loss

                    # Calculate accuracy
                    predicted_labels = np.argmax(val_output, axis=1)
                    true_labels = np.argmax(y_val_batch, axis=1)
                    correct_predictions += np.sum(predicted_labels == true_labels)

                val_loss = total_val_loss / len(X_val)
                val_accuracy = correct_predictions / len(X_val)

                new_lr = lr_scheduler.schedule(val_loss)
                if new_lr is not None:
                    self.learning_rate = new_lr

                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)

                tqdm.write(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy:.2%}, Learning Rate: {self.learning_rate}")
            else:
                tqdm.write(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss}, Learning Rate: {self.learning_rate}")

        return train_losses, val_losses, val_accuracies

    
    def test(self, X_test, y_test, batch_size):
        """Test the neural network.

        Args:
            X_test (ndarray): Test data.
            y_test (ndarray): Test labels.
            batch_size (int): Batch size for testing.

        Returns:
            tuple: Test loss, test accuracy.
        """
        total_loss = 0.0
        correct_predictions = 0

        for i in range(0, len(X_test), batch_size):
            output = self.forward(X_test[i:i+batch_size])
            loss = self.lossFunction.loss(y_test[i:i+batch_size], output)
            total_loss += loss

            predicted_labels = np.argmax(output, axis=1)
            true_labels = np.argmax(y_test[i:i+batch_size], axis=1)
            correct_predictions += np.sum(predicted_labels == true_labels)

        test_loss = total_loss / len(X_test)
        test_accuracy = correct_predictions / len(X_test)
        return test_loss, test_accuracy

    def predict(self, X):
        """Predict labels for input data.

        Args:
            X (ndarray): Input data.

        Returns:
            ndarray: Predicted labels.
        """
        output = self.forward(X)
        predictions = np.argmax(output, axis=1)
        return predictions
    
    def save(self, file_path):
        """Save the neural network to a file.

        Args:
            file_path (str): File path to save the model.
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path):
        """Load a neural network from a file.

        Args:
            file_path (str): File path to load the model from.

        Returns:
            FFNN: Loaded neural network.
        """
        with open(file_path, 'rb') as f:
            return pickle.load(f)
