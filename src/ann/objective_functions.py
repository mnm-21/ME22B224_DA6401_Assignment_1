import numpy as np
from .activations import softmax

# cross-entropy loss: -sum(y * log(p))
def cross_entropy(y_true, logits):
    probs = softmax(logits)
    eps = 1e-12  # small constant to avoid log(0)
    loss = -np.sum(y_true * np.log(probs + eps))
    return loss

# gradient of cross-entropy w.r.t. logits: (p - y)
def cross_entropy_grad(y_true, logits):
    probs = softmax(logits)
    return (probs - y_true)

# mean squared error: sum of (y - p)^2
def mse(y_true, logits):
    probs = softmax(logits)
    return np.sum((y_true - probs) ** 2)

# mse gradient needs to go through the softmax jacobian
def mse_grad(y_true, logits):
    probs = softmax(logits)
    d_probs = 2 * (probs - y_true)
    # applying softmax jacobian: d_logits = probs * (d_probs - sum(d_probs * probs))
    s = np.sum(d_probs * probs, axis=1, keepdims=True)
    return probs * (d_probs - s)

# maps loss name to (loss_fn, grad_fn)
LOSSES = {
    'cross_entropy': (cross_entropy, cross_entropy_grad),
    'mean_squared_error': (mse, mse_grad),
}