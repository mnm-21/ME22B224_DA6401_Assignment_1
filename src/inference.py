import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ann.neural_network import NeuralNetwork
from ann.activations import softmax
from utils.data_loader import load_data

def parse_arguments():
    p = argparse.ArgumentParser(description='Run inference on test set')
    p.add_argument('-d', '--dataset', default='fashion_mnist', choices=['mnist', 'fashion_mnist'])
    p.add_argument('-e', '--epochs', type=int, default=10)
    p.add_argument('-b', '--batch_size', type=int, default=64)
    p.add_argument('-l', '--loss', default='cross_entropy', choices=['cross_entropy', 'mean_squared_error'])
    p.add_argument('-o', '--optimizer', default='sgd', choices=['sgd', 'momentum', 'nag', 'rmsprop'])
    p.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    p.add_argument('-wd', '--weight_decay', type=float, default=0.0)
    p.add_argument('-nhl', '--num_layers', type=int, default=3)
    p.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128, 128, 128])
    p.add_argument('-a', '--activation', default='relu', choices=['sigmoid', 'tanh', 'relu'])
    p.add_argument('-w_i', '--weight_init', default='xavier', choices=['random', 'xavier'])
    p.add_argument('-w_p', '--wandb_project', default='da6401_assignment_1')
    p.add_argument('--model_path', default='best_model.npy')
    return p.parse_args()


def load_model(model_path):
    """load saved weights from .npy file"""
    data = np.load(model_path, allow_pickle=True).item()
    return data


def evaluate_model(model, X_test, y_test):
    """run model on test set and compute all metrics"""
    logits = model.forward(X_test)
    probs = softmax(logits)
    preds = np.argmax(probs, axis=1)
    labels = np.argmax(y_test, axis=1)

    # compute metrics using sklearn
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average='macro')
    rec = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')

    return {
        'logits': logits,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
    }


def main():
    args = parse_arguments()

    if len(args.hidden_size) == 1:
        args.hidden_size = args.hidden_size * args.num_layers
    elif len(args.hidden_size) < args.num_layers:
        args.hidden_size = args.hidden_size + [args.hidden_size[-1]] * (args.num_layers - len(args.hidden_size))
    args.hidden_size = args.hidden_size[:args.num_layers]

    # load test data (we don't need train/val for inference)
    _, _, _, _, X_test, y_test = load_data(args.dataset)

    # build model with same architecture and load saved weights
    model = NeuralNetwork(args)
    weights = load_model(args.model_path)
    model.set_weights(weights)

    results = evaluate_model(model, X_test, y_test)

    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-score:  {results['f1']:.4f}")
    print("Evaluation complete!")

    return results


if __name__ == '__main__':
    main()
