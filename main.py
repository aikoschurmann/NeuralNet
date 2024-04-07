import ender
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
X_train = X_train.reshape(-1, 28*28) / 255.0  # Flatten and normalize training images
X_test = X_test.reshape(-1, 28*28) / 255.0    # Flatten and normalize test images
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)  # One-hot encode training labels
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)    # One-hot encode test labels

# Split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

alpha = 0.0001

BCE = ender.losses.BinaryCrossEntropy()
adam = optimizer=ender.optimizers.Adam(alpha)
regularizationType = ender.networks.regularizationType.L2

model = ender.networks.FFNN(loss_function = BCE,
                            optimizer = adam,
                            learning_rate = alpha,
                            regularization = regularizationType,
                            lmbda=1)


model.add_layer(ender.layer.DenseLayer(28*28, 128, initializer=ender.initializers.XavierInitializer()))
model.add_layer(ender.layer.DenseLayer(128, 256, initializer=ender.initializers.XavierInitializer()))
model.add_layer(ender.layer.DenseLayer(256, 128, initializer=ender.initializers.XavierInitializer()))
model.add_layer(ender.layer.OutputLayer(128, 10, initializer=ender.initializers.XavierInitializer()))

# Define hyperparameters
epochs = 10
learning_rate = 0.001
batch_size = 50

# Training the neural network
model.summary()
scheduler = ender.schedulers.ReduceLROnPlateauScheduler(initial_lr=learning_rate, factor=0.1, patience=6)

train_losses, val_losses, val_accuracies = model.train(X_train, 
                                                       y_train, 
                                                       epochs=epochs, 
                                                       batch_size=batch_size, 
                                                       validation_data=(X_val, y_val),
                                                       lr_scheduler=scheduler)


# Set style
plt.style.use('dark_background')

# Plotting the loss over epochs
plt.plot(range(1, epochs+1), train_losses, label='Train Loss', color='blue', linestyle='-')
plt.plot(range(1, epochs+1), val_losses, label='Validation Loss', color='red', linestyle='--')

# Add labels and title
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Loss over Epochs', fontsize=14)

# Add legend
plt.legend(fontsize=10)

# Add grid
plt.grid(True, alpha=0.5)

# Show plot
plt.tight_layout()  # Adjust layout to prevent clipping of labels

plt.show()

# Test the model
test_loss, test_accuracy = model.test(X_test, y_test, batch_size=batch_size)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

def plot_mnist_with_confidence(image, confidence_scores):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # Reshape the MNIST image to 28x28
    image = image.reshape(28, 28)

    # Plot the MNIST image
    ax1.imshow(image, cmap='gray')
    ax1.axis('off')

    # Plot the confidence bars
    classes = np.arange(10)
    ax2.bar(classes, confidence_scores, color='blue')  # Use bar instead of barh
    ax2.set_xticks(classes)
    ax2.set_xticklabels(classes)
    ax2.set_ylabel('Confidence')
    ax2.set_xlabel('Digit')
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

for i, x in enumerate(X_val):
    confidence_scores = model.forward(x.reshape(1, -1)).flatten()
    plot_mnist_with_confidence(x, confidence_scores)