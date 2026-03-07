import numpy as np
from .activations import ACTIVATIONS

class Layer:
    def __init__(self, in_dim, out_dim, activation='relu', weight_init='xavier'):
        # weight initialization
        if weight_init == 'xavier':
            # xavier: variance = 2/(fan_in + fan_out)
            scale = np.sqrt(2.0 / (in_dim + out_dim))
            self.W = np.random.randn(in_dim, out_dim) * scale
        elif weight_init == 'zeros':
            self.W = np.zeros((in_dim, out_dim))
        else:
            # small random weights
            self.W = np.random.randn(in_dim, out_dim) * 0.01
        self.b = np.zeros((1, out_dim))

        # activation function and its derivative
        self.act_fn, self.act_deriv = ACTIVATIONS[activation]

        # gradients (computed in backward pass)
        self.grad_W = None
        self.grad_b = None

    def forward(self, x):
        self.x = x                      # save input for backward
        self.z = x @ self.W + self.b    # pre-activation
        self.a = self.act_fn(self.z)    # post-activation
        return self.a

    def backward(self, d_a):
        # d_a = gradient of loss w.r.t. activations of this layer
        d_z = d_a * self.act_deriv(self.a)       # chain rule through activation
        self.grad_W = self.x.T @ d_z             # gradient w.r.t. weights
        self.grad_b = np.sum(d_z, axis=0, keepdims=True)  # gradient w.r.t. bias
        d_prev = d_z @ self.W.T                  # gradient to pass to previous layer
        return d_prev