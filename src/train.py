import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import numpy as np
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_arguments():
    p = argparse.ArgumentParser(description='Train a neural network')
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
    p.add_argument('-w_i', '--weight_init', default='xavier', choices=['random', 'xavier', 'zeros'])
    p.add_argument('-w_p', '--wandb_project', default='da6401_assignment_1')
    p.add_argument('--model_save_path', default='best_model.npy')
    p.add_argument('--detailed_log', action='store_true', help='Log gradient norms, activations per step')
    return p.parse_args()


def main():
    args = parse_arguments()

    # if user gives fewer hidden sizes than layers, pad with the last value
    if len(args.hidden_size) == 1:
        args.hidden_size = args.hidden_size * args.num_layers
    elif len(args.hidden_size) < args.num_layers:
        args.hidden_size = args.hidden_size + [args.hidden_size[-1]] * (args.num_layers - len(args.hidden_size))
    args.hidden_size = args.hidden_size[:args.num_layers]

    print(f"Config: {vars(args)}")

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.dataset)
    print(f"Data loaded: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    import wandb
    run = wandb.init(project=args.wandb_project, config=vars(args))

    model = NeuralNetwork(args)
    best_weights = model.train(X_train, y_train, X_val, y_val,
                               epochs=args.epochs, batch_size=args.batch_size,
                               wandb_run=run, log_details=args.detailed_log)

    # save best model
    if best_weights:
        save_path = os.path.join(BASE_DIR, args.model_save_path)
        np.save(save_path, best_weights)
        print(f"Best model saved to {save_path}")

    # final test eval
    if best_weights:
        model.set_weights(best_weights)
    test_acc, test_f1 = model.evaluate_metrics(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f} | Test F1: {test_f1:.4f}")

    # save config alongside the model
    import json
    config = vars(args)
    config['test_accuracy'] = float(test_acc)
    config['test_f1'] = float(test_f1)
    config_path = os.path.join(BASE_DIR, 'best_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    run.finish()
    print("Training complete!")


if __name__ == '__main__':
    main()
