# DA6401 Assignment 1 — MLP from Scratch

NumPy-only implementation of a configurable Multi-Layer Perceptron for MNIST and Fashion-MNIST classification.

## Links

- **W&B Report**: [PLACEHOLDER — add your report link here]
- **GitHub Repo**: [PLACEHOLDER — add your repo link here]

## Project Structure

```
src/
├── ann/
│   ├── activations.py       # sigmoid, tanh, relu, softmax + derivatives
│   ├── neural_layer.py      # single layer: forward, backward, weight init
│   ├── neural_network.py    # full network: train loop, evaluation, weight save/load
│   ├── objective_functions.py  # cross-entropy, MSE + gradients
│   └── optimizers.py        # SGD, Momentum, NAG, RMSProp
├── utils/
│   └── data_loader.py       # loads MNIST/Fashion-MNIST, normalize, one-hot, split
├── train.py                 # training script with CLI args and W&B logging
├── inference.py             # loads saved model and reports metrics
└── wandb_experiments.py     # runs all experiments for the W&B report
```

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python src/train.py -d mnist -e 10 -b 64 -o rmsprop -lr 0.001 -a relu -nhl 3 -sz 128 -w_i xavier
```

All CLI arguments:

| Flag | Description | Default |
|------|-------------|---------|
| `-d` | Dataset (mnist / fashion_mnist) | fashion_mnist |
| `-e` | Epochs | 10 |
| `-b` | Batch size | 64 |
| `-l` | Loss (cross_entropy / mean_squared_error) | cross_entropy |
| `-o` | Optimizer (sgd / momentum / nag / rmsprop) | sgd |
| `-lr` | Learning rate | 0.01 |
| `-wd` | Weight decay (L2) | 0.0 |
| `-nhl` | Number of hidden layers | 3 |
| `-sz` | Hidden layer sizes (list) | 128 128 128 |
| `-a` | Activation (sigmoid / tanh / relu) | relu |
| `-w_i` | Weight init (random / xavier / zeros) | xavier |
| `-w_p` | W&B project name | da6401_assignment_1 |

The best model is saved as `src/best_model.npy` and its config as `src/best_config.json`, selected by validation Macro F1-Score.

## Inference

```bash
python src/inference.py --model_path src/best_model.npy -d mnist -nhl 3 -sz 128 -a relu -w_i xavier
```

Outputs Accuracy, Precision, Recall, and F1-Score on the test set.

## W&B Experiments

To run all report experiments:

```bash
python src/wandb_experiments.py --section all
```

Or run a specific section:

```bash
python src/wandb_experiments.py --section 2.3
```

Available sections: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.8, 2.9, 2.10

## Implementation Notes

- The network outputs raw logits (no softmax on the output layer). Softmax is applied inside the loss function and evaluation.
- Gradients are computed analytically via backpropagation, flowing from the last layer to the first. Each layer exposes `grad_W` and `grad_b` after `backward()`.
- Best model selection uses validation Macro F1-Score.
- Weight saving/loading uses NumPy's `.npy` format with `get_weights()` and `set_weights()`.
