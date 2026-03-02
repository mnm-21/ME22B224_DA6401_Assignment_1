import numpy as np
from .neural_layer import Layer
from .activations import softmax
from .objective_functions import LOSSES
from .optimizers import OPTIMIZERS

class NeuralNetwork:
    def __init__(self, args):
        self.lr = args.learning_rate
        self.wd = args.weight_decay

        # build layers
        sizes = [784] + list(args.hidden_size) + [10]
        self.layers = []
        for i in range(len(sizes) - 1):
            act = args.activation if i < len(sizes) - 2 else args.activation
            layer = Layer(sizes[i], sizes[i + 1], act, args.weight_init)
            self.layers.append(layer)
        # output layer has no activation (we apply softmax separately)
        # override the last layer to be linear
        self.layers[-1].act_fn = lambda z: z
        self.layers[-1].act_deriv = lambda a: np.ones_like(a)

        self.loss_fn, self.loss_grad = LOSSES[args.loss]

        opt_cls = OPTIMIZERS[args.optimizer]
        if args.optimizer == 'rmsprop':
            self.optimizer = opt_cls(self.layers, lr=self.lr, weight_decay=self.wd)
        else:
            self.optimizer = opt_cls(self.layers, lr=self.lr, weight_decay=self.wd)

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        self.logits = out
        return out

    def backward(self, y_true, y_pred):
        grad_W_list = []
        grad_b_list = []

        d = self.loss_grad(y_true, y_pred)
        for layer in reversed(self.layers):
            d = layer.backward(d)
            grad_W_list.append(layer.grad_W)
            grad_b_list.append(layer.grad_b)

        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b

    def update_weights(self):
        self.optimizer.step()

    def train(self, X_train, y_train, X_val, y_val, epochs=1, batch_size=32, wandb_run=None):
        n = X_train.shape[0]
        best_val_acc = 0
        best_weights = None

        for ep in range(epochs):
            idx = np.random.permutation(n)
            X_shuffled = X_train[idx]
            y_shuffled = y_train[idx]

            epoch_loss = 0
            for start in range(0, n, batch_size):
                xb = X_shuffled[start:start + batch_size]
                yb = y_shuffled[start:start + batch_size]

                if hasattr(self.optimizer, 'pre_step'):
                    self.optimizer.pre_step()

                logits = self.forward(xb)
                loss = self.loss_fn(yb, logits)
                epoch_loss += loss * xb.shape[0]
                self.backward(yb, logits)
                self.update_weights()

            epoch_loss /= n
            train_acc = self.evaluate(X_train, y_train)
            val_acc = self.evaluate(X_val, y_val)
            val_loss = self._compute_loss(X_val, y_val)

            print(f"Epoch {ep+1}/{epochs} | loss: {epoch_loss:.4f} | train_acc: {train_acc:.4f} | val_acc: {val_acc:.4f}")

            if wandb_run:
                wandb_run.log({
                    'epoch': ep + 1,
                    'train_loss': epoch_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = self.get_weights()

        return best_weights

    def _compute_loss(self, X, y):
        logits = self.forward(X)
        return self.loss_fn(y, logits)

    def evaluate(self, X, y):
        logits = self.forward(X)
        probs = softmax(logits)
        preds = np.argmax(probs, axis=1)
        labels = np.argmax(y, axis=1)
        return np.mean(preds == labels)

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()
