import numpy as np
from .neural_layer import Layer
from .activations import softmax
from .objective_functions import LOSSES
from .optimizers import OPTIMIZERS
from sklearn.metrics import f1_score


class NeuralNetwork:
    def __init__(self, args):
        self.lr = args.learning_rate
        self.wd = args.weight_decay

        # build layers: input(784) -> hidden layers -> output(10)
        sizes = [784] + list(args.hidden_size) + [10]
        self.layers = []
        for i in range(len(sizes) - 1):
            layer = Layer(sizes[i], sizes[i + 1], args.activation, args.weight_init)
            self.layers.append(layer)

        # last layer outputs raw logits (no activation)
        self.layers[-1].act_fn = lambda z: z
        self.layers[-1].act_deriv = lambda a: np.ones_like(a)

        self.loss_fn, self.loss_grad = LOSSES[args.loss]
        self.optimizer = OPTIMIZERS[args.optimizer](
            self.layers, lr=self.lr, weight_decay=self.wd
        )

    def forward(self, X):
        # pass input through each layer sequentially
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        self.logits = out
        return out

    def backward(self, y_true, y_pred):
        # backprop from last layer to first
        self.grad_W = np.empty(len(self.layers), dtype=object)
        self.grad_b = np.empty(len(self.layers), dtype=object)
        d = self.loss_grad(y_true, y_pred)
        for i, layer in enumerate(reversed(self.layers)):
            d = layer.backward(d)
            self.grad_W[i] = layer.grad_W
            self.grad_b[i] = layer.grad_b
        return self.grad_W, self.grad_b

    def update_weights(self):
        self.optimizer.step()

    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=1,
        batch_size=32,
        wandb_run=None,
        log_details=False,
    ):
        n = X_train.shape[0]
        best_val_f1 = 0
        best_weights = None
        step = 0

        for ep in range(epochs):
            idx = np.random.permutation(n)
            X_shuffled = X_train[idx]
            y_shuffled = y_train[idx]

            epoch_loss = 0
            for start in range(0, n, batch_size):
                xb = X_shuffled[start : start + batch_size]
                yb = y_shuffled[start : start + batch_size]

                # NAG needs a lookahead step before forward pass
                if hasattr(self.optimizer, "pre_step"):
                    self.optimizer.pre_step()

                # forward, loss, backward, update
                logits = self.forward(xb)
                loss = self.loss_fn(yb, logits)
                epoch_loss += loss * xb.shape[0]
                self.backward(yb, logits)
                self.update_weights()
                step += 1

                # detailed per-step logging (gradient norms, dead neurons, activations)
                if log_details and wandb_run:
                    log = {"step": step}
                    for j, layer in enumerate(self.layers[:-1]):
                        log[f"grad_norm_layer{j}"] = float(np.linalg.norm(layer.grad_W))

                    import wandb as _wb

                    for j, layer in enumerate(self.layers[:-1]):
                        acts = layer.a
                        dead = np.all(acts == 0, axis=0)
                        log[f"dead_count_layer{j}"] = int(np.sum(dead))
                        log[f"dead_frac_layer{j}"] = float(np.mean(dead))
                        log[f"act_mean_layer{j}"] = float(np.mean(acts))
                        log[f"act_dist_layer{j}"] = _wb.Histogram(acts.flatten())

                    # log per-neuron gradient norms for first layer (used in 2.9 symmetry analysis)
                    if self.layers[0].grad_W is not None:
                        gw = self.layers[0].grad_W
                        for k in range(min(5, gw.shape[1])):
                            log[f"neuron{k}_grad_norm"] = float(
                                np.linalg.norm(gw[:, k])
                            )
                    wandb_run.log(log)

            # end of epoch: compute metrics
            epoch_loss /= n
            train_acc = self.evaluate(X_train, y_train)
            val_acc, val_f1 = self.evaluate_metrics(X_val, y_val)
            val_loss = self._compute_loss(X_val, y_val)

            print(
                f"Epoch {ep+1}/{epochs} | loss: {epoch_loss:.4f} | train_acc: {train_acc:.4f} | val_acc: {val_acc:.4f} | val_f1: {val_f1:.4f}"
            )

            if wandb_run and not log_details:
                wandb_run.log(
                    {
                        "epoch": ep + 1,
                        "train_loss": epoch_loss,
                        "train_accuracy": train_acc,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                        "val_f1": val_f1,
                    }
                )

            # detailed logging needed for some sections
            if wandb_run and log_details:
                wandb_run.log(
                    {
                        "epoch": ep + 1,
                        "train_loss": epoch_loss,
                        "train_accuracy": train_acc,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                        "val_f1": val_f1,
                        "step": step,
                    }
                )

            # save weights if this is the best f1 so far
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_weights = self.get_weights()

        return best_weights

    def _compute_loss(self, X, y):
        logits = self.forward(X)
        return self.loss_fn(y, logits)

    def evaluate(self, X, y):
        acc, _ = self.evaluate_metrics(X, y)
        return acc

    def evaluate_metrics(self, X, y):
        # returns both accuracy and macro f1
        logits = self.forward(X)
        probs = softmax(logits)
        preds = np.argmax(probs, axis=1)
        labels = np.argmax(y, axis=1)
        acc = np.mean(preds == labels)
        f1 = f1_score(labels, preds, average="macro")
        return acc, f1

    def get_weights(self):
        # save all layer weights as a dictionary
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        # load weights from a dictionary (e.g. from np.load)
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()
