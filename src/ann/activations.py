import numpy as np


# sigmoid: squashes input to (0, 1)
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))  # clip to avoid overflow


def sigmoid_deriv(a):
    return a * (1 - a)  # derivative in terms of output


# tanh: squashes input to (-1, 1)
def tanh(z):
    return np.tanh(z)


def tanh_deriv(a):
    return 1 - a**2


# relu: max(0, x)
def relu(z):
    return np.maximum(0, z)


def relu_deriv(a):
    return (a > 0).astype(float)  # 1 if active, 0 if dead


# softmax: converts logits to probabilities
def softmax(z):
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


# maps activation name to (function, derivative)
ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_deriv),
    "tanh": (tanh, tanh_deriv),
    "relu": (relu, relu_deriv),
}
