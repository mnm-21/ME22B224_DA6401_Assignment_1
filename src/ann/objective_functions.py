import numpy as np
from .activations import softmax

def cross_entropy(y_true, logits):
    probs = softmax(logits)
    eps = 1e-12
    loss = -np.sum(y_true * np.log(probs + eps)) / y_true.shape[0]
    return loss

def cross_entropy_grad(y_true, logits):
    probs = softmax(logits)
    return (probs - y_true) / y_true.shape[0]

def mse(y_true, logits):
    probs = softmax(logits)
    return np.mean(np.sum((y_true - probs) ** 2, axis=1))

def mse_grad(y_true, logits):
    probs = softmax(logits)
    n = y_true.shape[0]
    d_probs = 2 * (probs - y_true) / n
    # softmax jacobian: d_logits = probs * (d_probs - sum(d_probs * probs))
    s = np.sum(d_probs * probs, axis=1, keepdims=True)
    return probs * (d_probs - s)

LOSSES = {
    'cross_entropy': (cross_entropy, cross_entropy_grad),
    'mean_squared_error': (mse, mse_grad),
}