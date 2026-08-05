import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Import your custom library
# Assuming 'ender' is the package containing the classes from the previous prompt
import ender 
from ender.src.activations import ReLU, Softmax
from ender.src.losses import CrossEntropyLossMultiClass
from ender.src.layer import DenseLayer
from ender.src.optimizers import Adam
from ender.src.initializers import HeInitializer, XavierInitializer
from ender.src.schedulers import ReduceLROnPlateauScheduler
from ender.src.FFNN import RegularizationType, FFNN

# 1. Load and Preprocess Data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Flatten and Normalize
X_train = X_train.reshape(-1, 28*28) / 255.0
X_test = X_test.reshape(-1, 28*28) / 255.0

# One-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Split Validation Set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# 2. Define Hyperparameters
BATCH_SIZE = 500
EPOCHS = 30  # 15 is usually sufficient for >95% on MNIST
LEARNING_RATE = 0.001
L2_LAMBDA = 0.0001 # Reduced from 1.0 to prevent underfitting

# 3. Initialize Model
# Note: We pass the Class 'Adam', not an instance, so FFNN can manage the learning rate internally
model = FFNN(
    optimizer=Adam,
    learning_rate=LEARNING_RATE,
    loss_function=CrossEntropyLossMultiClass(), # Changed from Binary to Categorical
    regularization=RegularizationType.L2,
    lmbda=L2_LAMBDA
)

# 4. Build Architecture
# Input -> Hidden (ReLU) -> Hidden (ReLU) -> ... -> Output (Softmax)


# Hidden Layer 1: 784 -> 128
model.add_layer(DenseLayer(
    inputs=28*28,
    outputs=128,
    activation=ReLU(),
    initializer=HeInitializer() # He is better for ReLU than Xavier
))


# Hidden Layer 2: 128 -> 64
model.add_layer(DenseLayer(
    inputs=128,
    outputs=64,
    activation=ReLU(),
    initializer=HeInitializer()
))


# Hidden Layer 3: 64 -> 32
model.add_layer(DenseLayer(
    inputs=64,
    outputs=32,
    activation=ReLU(),
    initializer=HeInitializer()
))


# Output Layer: 32 -> 10 (Softmax for probabilities)
model.add_layer(DenseLayer(
    inputs=32,
    outputs=10,
    activation=Softmax(),
    initializer=XavierInitializer()
))

model.summary()

# 5. Training
scheduler = ReduceLROnPlateauScheduler(initial_lr=LEARNING_RATE, factor=0.5, patience=3)

train_losses, val_losses, val_accuracies = model.train(
    X_train, 
    y_train, 
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE, 
    validation_data=(X_val, y_val),
    lr_scheduler=scheduler
)

# 6. Plot Training Results 
plt.style.use('dark_background')
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color='cyan')
ax1.plot(train_losses, label='Train Loss', color='cyan', linestyle='-')
ax1.plot(val_losses, label='Val Loss', color='magenta', linestyle='--')
ax1.tick_params(axis='y', labelcolor='cyan')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.2)

ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Accuracy', color='yellow') 
ax2.plot(val_accuracies, label='Val Accuracy', color='yellow', linestyle='-.')
ax2.tick_params(axis='y', labelcolor='yellow')
ax2.legend(loc='upper right')

plt.title('Training Metrics: Loss & Accuracy')
plt.tight_layout()
plt.show()

# 7. Testing
test_loss, test_accuracy = model.test(X_test, y_test, batch_size=BATCH_SIZE)
print(f"\nFinal Test Results - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2%}")

# 8. Visualization Function
def plot_prediction(image, probabilities):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Image
    ax1.imshow(image.reshape(28, 28), cmap='gray')
    ax1.axis('off')
    ax1.set_title("Input Digit")
    
    # Bar Chart
    classes = np.arange(10)
    colors = ['gray'] * 10
    colors[np.argmax(probabilities)] = 'limegreen' # Highlight prediction
    
    ax2.bar(classes, probabilities, color=colors)
    ax2.set_xticks(classes)
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel('Confidence')
    ax2.set_title(f"Prediction: {np.argmax(probabilities)}")
    
    plt.tight_layout()
    plt.show()

# Visualize first 5 failures or successes
print("\nVisualizing Predictions...")
for i in range(5):
    # Get single sample
    x_sample = X_val[i:i+1] # Keep shape (1, 784)
    y_true = np.argmax(y_val[i])
    
    # Predict
    probs = model.forward(x_sample).flatten()
    
    print(f"True Label: {y_true}")
    plot_prediction(x_sample, probs)