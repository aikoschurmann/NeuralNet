import os
import tensorflow as tf
from ender.src.backend import np
import keras

from sklearn.model_selection import train_test_split

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data
X_train = X_train.reshape(-1, 28*28) / 255.0  # Flatten and normalize training images
X_test = X_test.reshape(-1, 28*28) / 255.0    # Flatten and normalize test images
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)  # One-hot encode training labels
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)    # One-hot encode test labels

# Split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)


# Define hyperparameters
epochs = 20
learning_rate = 0.001
batch_size = 500

# Create a Sequential model
#use conv2d for convolutional neural network

model = keras.models.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])



# Compile the model

model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])   

# Train the model
print("Training the model...")
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

# Evaluate the model

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)


