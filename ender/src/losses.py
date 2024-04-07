import numpy as np

class LossFunction:
    """Base class for loss functions."""

    def loss(self, y, predicted):
        """Calculate the loss.

        Args:
            y (ndarray): Ground truth labels.
            predicted (ndarray): Predicted labels.

        Returns:
            float: The loss value.
        """
        raise NotImplementedError("loss method must be implemented in subclass.")

    def derivative(self, y, predicted):
        """Calculate the derivative of the loss.

        Args:
            y (ndarray): Ground truth labels.
            predicted (ndarray): Predicted labels.

        Returns:
            ndarray: The derivative of the loss.
        """
        raise NotImplementedError("derivative method must be implemented in subclass.")

class MeanSquaredErrorLoss(LossFunction):
    """Class for Mean Squared Error loss function."""

    def loss(self, y, predicted):
        return np.mean(np.square(predicted - y))

    def derivative(self, y, predicted):
        return 2 * (predicted - y) / len(y)

class MeanAbsoluteError(LossFunction):
    """Class for Mean Absolute Error loss function."""

    def loss(self, y, predicted):
        return np.mean(np.abs(predicted - y))

    def derivative(self, y, predicted):
        return np.sign(predicted - y) / len(y)

class BinaryCrossEntropy(LossFunction):
    """Class for Binary Cross Entropy loss function."""

    def loss(self, y, predicted):
        epsilon = 1e-9
        return -np.mean(y * np.log(predicted + epsilon) + (1 - y) * np.log(1 - predicted + epsilon))

    def derivative(self, y, predicted):
        epsilon = 1e-9
        return (predicted - y) / ((predicted + epsilon) * (1 - predicted + epsilon))

class HuberLoss(LossFunction):
    """Class for Huber loss function."""

    def __init__(self, delta=1.0):
        self.delta = delta

    def loss(self, y, predicted):
        diff = np.abs(y - predicted)
        huber_loss = np.where(diff <= self.delta, 0.5 * np.square(diff), self.delta * (diff - 0.5 * self.delta))
        return np.mean(huber_loss)

    def derivative(self, y, predicted):
        diff = y - predicted
        return np.where(np.abs(diff) <= self.delta, -diff, -self.delta * np.sign(diff))

class HingeLoss(LossFunction):
    """Class for Hinge loss function."""

    def loss(self, y, predicted):
        hinge_loss = np.maximum(0, 1 - y * predicted)
        return np.mean(hinge_loss)

    def derivative(self, y, predicted):
        return np.where(y * predicted < 1, -y, 0)

class SquaredHingeLoss(LossFunction):
    """Class for Squared Hinge loss function."""

    def loss(self, y, predicted):
        squared_hinge_loss = np.square(np.maximum(0, 1 - y * predicted))
        return np.mean(squared_hinge_loss)

    def derivative(self, y, predicted):
        return np.where(y * predicted < 1, -2 * y * (1 - y * predicted), 0)

class CrossEntropyLossMultiClass(LossFunction):
    """Class for Cross Entropy loss function for multi-class classification."""

    def loss(self, y, predicted):
        epsilon = 1e-9
        cross_entropy_loss = -np.sum(y * np.log(predicted + epsilon)) / len(y)
        return cross_entropy_loss

    def derivative(self, y, predicted):
        epsilon = 1e-9
        return (predicted - y) / ((predicted + epsilon) * (1 - predicted + epsilon))
