from ender.src.backend import np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from dfs import augment_data

# Import your custom library
import ender 
from ender.src.activations import ReLU, Softmax
from ender.src.losses import CategoricalCrossEntropyWithLogits
from ender.src.layer import DenseLayer, ActivationLayer, BatchNormalizationLayer, DropoutLayer
from ender.src.conv import Conv2D, MaxPooling2D, FlattenLayer
from ender.src.optimizers import Adam
from ender.src.initializers import HeInitializer, XavierInitializer
from ender.src.callbacks import EarlyStopping
from ender.src.FFNN import RegularizationType, FFNN

# 1. Load and Preprocess Data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshape for CNN (N, Channels, Height, Width)
X_train = X_train.reshape(-1, 1, 28, 28) / 255.0
X_test = X_test.reshape(-1, 1, 28, 28) / 255.0

# One-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Split Validation Set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Disable data augmentation for CNN to save training time in NumPy
# X_train, y_train = augment_data(X_train, y_train)

# 2. Define Hyperparameters
BATCH_SIZE = 250
EPOCHS = 5
LEARNING_RATE = 0.001
L2_LAMBDA = 0.0001 # Reduced from 1.0 to prevent underfitting

# 3. Initialize Model
# Note: We pass the Class 'Adam', not an instance, so FFNN can manage the learning rate internally
model = FFNN(
    optimizer=Adam,
    learning_rate=LEARNING_RATE,
    loss_function=CategoricalCrossEntropyWithLogits(), # Loss with Logits
    regularization=RegularizationType.L2,
    lmbda=L2_LAMBDA
)

# CNN Architecture
model.add_layer(Conv2D(in_channels=1, out_channels=8, kernel_size=3))
model.add_layer(ActivationLayer(ReLU()))
model.add_layer(MaxPooling2D(pool_size=2))

model.add_layer(Conv2D(in_channels=8, out_channels=16, kernel_size=3))
model.add_layer(ActivationLayer(ReLU()))
model.add_layer(MaxPooling2D(pool_size=2))

model.add_layer(FlattenLayer())

# Dense layers
model.add_layer(DenseLayer(inputs=16 * 5 * 5, outputs=64, initializer=HeInitializer()))
model.add_layer(ActivationLayer(ReLU()))
model.add_layer(DropoutLayer(rate=0.3))

model.add_layer(DenseLayer(inputs=64, outputs=10, initializer=XavierInitializer()))

model.summary()

train_losses, val_losses, val_accuracies = model.train(
    X_train, 
    y_train, 
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE, 
    validation_data=(X_val, y_val),
    callbacks=[EarlyStopping(patience=3)]
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
plt.savefig('training_metrics.png')

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
    plt.savefig(f'prediction_{np.argmax(probabilities)}.png')
    plt.close(fig)

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