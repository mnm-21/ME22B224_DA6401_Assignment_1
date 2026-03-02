import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_deriv(a):
    return a * (1 - a)

def tanh(z):
    return np.tanh(z)

def tanh_deriv(a):
    return 1 - a ** 2

def relu(z):
    return np.maximum(0, z)

def relu_deriv(a):
    return (a > 0).astype(float)

def softmax(z):
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

ACTIVATIONS = {
    'sigmoid': (sigmoid, sigmoid_deriv),
    'tanh': (tanh, tanh_deriv),
    'relu': (relu, relu_deriv),
}