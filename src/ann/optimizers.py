import numpy as np

# vanilla mini-batch gradient descent
class SGD:
    def __init__(self, layers, lr=0.01, weight_decay=0):
        self.layers = layers
        self.lr = lr
        self.wd = weight_decay

    def step(self):
        for layer in self.layers:
            # W update includes L2 regularization term
            layer.W -= self.lr * (layer.grad_W + self.wd * layer.W)
            layer.b -= self.lr * layer.grad_b


# SGD with momentum: accumulates velocity to smooth out updates
class Momentum:
    def __init__(self, layers, lr=0.01, beta=0.9, weight_decay=0):
        self.layers = layers
        self.lr = lr
        self.beta = beta
        self.wd = weight_decay
        self.v_w = [np.zeros_like(l.W) for l in layers]
        self.v_b = [np.zeros_like(l.b) for l in layers]

    def step(self):
        for i, layer in enumerate(self.layers):
            # v = beta * v_prev + current_gradient
            self.v_w[i] = self.beta * self.v_w[i] + layer.grad_W + self.wd * layer.W
            self.v_b[i] = self.beta * self.v_b[i] + layer.grad_b
            layer.W -= self.lr * self.v_w[i]
            layer.b -= self.lr * self.v_b[i]


# nesterov accelerated gradient: looks ahead before computing gradient
class NAG:
    def __init__(self, layers, lr=0.01, beta=0.9, weight_decay=0):
        self.layers = layers
        self.lr = lr
        self.beta = beta
        self.wd = weight_decay
        self.v_w = [np.zeros_like(l.W) for l in layers]
        self.v_b = [np.zeros_like(l.b) for l in layers]

    def pre_step(self):
        # lookahead: temporarily move weights in the direction of momentum
        for i, layer in enumerate(self.layers):
            layer.W -= self.lr * self.beta * self.v_w[i]
            layer.b -= self.lr * self.beta * self.v_b[i]

    def step(self):
        for i, layer in enumerate(self.layers):
            # undo the lookahead
            layer.W += self.lr * self.beta * self.v_w[i]
            layer.b += self.lr * self.beta * self.v_b[i]
            # update velocity with gradient computed at the lookahead position
            self.v_w[i] = self.beta * self.v_w[i] + layer.grad_W + self.wd * layer.W
            self.v_b[i] = self.beta * self.v_b[i] + layer.grad_b
            # apply the update
            layer.W -= self.lr * self.v_w[i]
            layer.b -= self.lr * self.v_b[i]


# rmsprop: adaptive learning rate per parameter
class RMSProp:
    def __init__(self, layers, lr=0.001, beta=0.9, eps=1e-8, weight_decay=0):
        self.layers = layers
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.wd = weight_decay
        self.v_w = [np.zeros_like(l.W) for l in layers]  # running avg of squared gradients
        self.v_b = [np.zeros_like(l.b) for l in layers]

    def step(self):
        for i, layer in enumerate(self.layers):
            gw = layer.grad_W + self.wd * layer.W
            gb = layer.grad_b
            # update running average of squared gradients
            self.v_w[i] = self.beta * self.v_w[i] + (1 - self.beta) * gw ** 2
            self.v_b[i] = self.beta * self.v_b[i] + (1 - self.beta) * gb ** 2
            # divide by sqrt to get adaptive lr
            layer.W -= self.lr * gw / (np.sqrt(self.v_w[i]) + self.eps)
            layer.b -= self.lr * gb / (np.sqrt(self.v_b[i]) + self.eps)


OPTIMIZERS = {
    'sgd': SGD,
    'momentum': Momentum,
    'nag': NAG,
    'rmsprop': RMSProp,
}