# DA6401 Assignment 1 — MLP from Scratch

NumPy-only implementation of a configurable Multi-Layer Perceptron for MNIST and Fashion-MNIST classification.

## Links

- **W&B Report**: [Assignment 1 - DA6401 - ME22B224](https://wandb.ai/mayank-chandak21-/da6401_assignment_1_ME22B224/reports/Assignment-1-DA6401-ME22B224--VmlldzoxNjEzMzc1Nw?accessToken=twnos1wcgccoerce7rie8qqdrkssdj57l2z25bkfd7dfgbztgcmevgts8pqi4sks)
- **GitHub Repo**: [ME22B224_DA6401_Assignment_1](https://github.com/mnm-21/ME22B224_DA6401_Assignment_1.git)

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
python src/train.py -d mnist -e 10 -b 32 -o momentum -lr 0.076 -a sigmoid -nhl 2 -sz 128 64 -w_i xavier
```

All CLI arguments:

| Flag | Description | Default |
|------|-------------|---------|
| `-d` | Dataset (mnist / fashion_mnist) | fashion_mnist |
| `-e` | Epochs | 10 |
| `-b` | Batch size | 32 |
| `-l` | Loss (cross_entropy / mean_squared_error) | cross_entropy |
| `-o` | Optimizer (sgd / momentum / nag / rmsprop) | momentum |
| `-lr` | Learning rate | 0.076 |
| `-wd` | Weight decay (L2) | 0.0 |
| `-nhl` | Number of hidden layers | 2 |
| `-sz` | Hidden layer sizes (list) | 128 64 |
| `-a` | Activation (sigmoid / tanh / relu) | sigmoid |
| `-w_i` | Weight init (random / xavier / zeros) | xavier |
| `-w_p` | W&B project name | da6401_assignment_1 |

The best model is saved as `src/best_model.npy` and its config as `src/best_config.json`, selected by validation Macro F1-Score.

## Inference

```bash
python src/inference.py --model_path src/best_model.npy -d mnist -nhl 2 -sz 128 64 -a sigmoid -w_i xavier
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
