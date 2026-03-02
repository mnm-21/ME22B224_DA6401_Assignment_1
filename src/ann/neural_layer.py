import numpy as np
from .activations import ACTIVATIONS

class Layer:
    def __init__(self, in_dim, out_dim, activation='relu', weight_init='xavier'):
        if weight_init == 'xavier':
            scale = np.sqrt(2.0 / (in_dim + out_dim))
            self.W = np.random.randn(in_dim, out_dim) * scale
        else:
            self.W = np.random.randn(in_dim, out_dim) * 0.01
        self.b = np.zeros((1, out_dim))

        self.act_fn, self.act_deriv = ACTIVATIONS[activation]

        self.grad_W = None
        self.grad_b = None

    def forward(self, x):
        self.x = x
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            self.z = np.nan_to_num(x @ self.W + self.b, nan=0.0, posinf=1e6, neginf=-1e6)
        self.a = self.act_fn(self.z)
        return self.a

    def backward(self, d_a):
        d_z = d_a * self.act_deriv(self.a)
        d_z = np.clip(d_z, -1e6, 1e6)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            self.grad_W = np.nan_to_num(self.x.T @ d_z, nan=0.0, posinf=1e6, neginf=-1e6)
            self.grad_b = np.sum(d_z, axis=0, keepdims=True)
            d_prev = np.nan_to_num(d_z @ self.W.T, nan=0.0, posinf=1e6, neginf=-1e6)
        return d_prev