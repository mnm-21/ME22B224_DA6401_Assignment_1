import numpy as np
from sklearn.model_selection import train_test_split


def load_data(dataset="fashion_mnist", val_split=0.1):
    # load raw data from keras
    if dataset == "mnist":
        from keras.datasets import mnist

        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    else:
        from keras.datasets import fashion_mnist

        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    # flatten 28x28 images to 784-dim vectors and normalize to [0, 1]
    X_train = X_train.reshape(-1, 784).astype(np.float64) / 255.0
    X_test = X_test.reshape(-1, 784).astype(np.float64) / 255.0

    # one-hot encode labels (10 classes)
    num_classes = 10
    y_train_oh = np.eye(num_classes)[y_train]
    y_test_oh = np.eye(num_classes)[y_test]

    # split training data into train and validation
    X_train, X_val, y_train_oh, y_val_oh = train_test_split(
        X_train, y_train_oh, test_size=val_split, random_state=224
    )

    return X_train, y_train_oh, X_val, y_val_oh, X_test, y_test_oh
