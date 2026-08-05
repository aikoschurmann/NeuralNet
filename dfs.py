from scipy.ndimage import rotate, shift
import numpy as np

def augment_data(X, y):
    X_aug, y_aug = [], []
    for image, label in zip(X, y):
        image = image.reshape(28, 28)
        # Add original
        X_aug.append(image.flatten())
        y_aug.append(label)
        # Add rotated (-15 to 15 degrees)
        angle = np.random.uniform(-15, 15)
        rot_img = rotate(image, angle, reshape=False)
        X_aug.append(rot_img.flatten())
        y_aug.append(label)
        # Add shifted (move pixels slightly)
        shifted_img = shift(image, [np.random.randint(-2, 2), np.random.randint(-2, 2)])
        X_aug.append(shifted_img.flatten())
        y_aug.append(label)
    return np.array(X_aug), np.array(y_aug)