# experiments for wandb report

import sys, os, json
import numpy as np
np.seterr(all='ignore')
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import numpy as np
import wandb
from sklearn.metrics import confusion_matrix, f1_score
from ann.neural_network import NeuralNetwork
from ann.activations import softmax
from utils.data_loader import load_data

PROJECT_NAME = 'da6401_assignment_1_ME22B224'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Args:
    """quick way to create args without argparse"""

    def __init__(self, **kwargs):
        defaults = dict(
            dataset='mnist', epochs=10, batch_size=64,
            loss='cross_entropy', optimizer='sgd', learning_rate=0.01,
            weight_decay=0.0, num_layers=3, hidden_size=[128, 128, 128],
            activation='relu', weight_init='xavier',
        )
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v) # assign updated values to args object


def run_experiment(args, run_name, log_details=False, tags=None):
    """train a model, evaluate on test set, return everything I need"""
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.dataset)

    config = {k: v for k, v in vars(args).items()}
    # make sure previous run is done before starting new one
    if wandb.run is not None:
        wandb.finish()
    run = wandb.init(project=PROJECT_NAME, name=run_name, config=config, tags=tags or [])
    
    model = NeuralNetwork(args)
    best_weights = model.train(X_train, y_train, X_val, y_val,
                               epochs=args.epochs, batch_size=args.batch_size,
                               wandb_run=run, log_details=log_details)
    if best_weights:
        model.set_weights(best_weights)
    test_accuracy, test_f1 = model.evaluate_metrics(X_test, y_test)
    run.summary['test_accuracy'] = test_accuracy
    run.summary['test_f1'] = test_f1
    print(f"{run_name}: test_acc={test_accuracy:.4f} | test_f1={test_f1:.4f}")

    return model, best_weights, run, X_test, y_test


# 2.1 Data Exploration
def section_2_1():
    print("==============================================================================")
    print("2.1 Data Exploration")
    from keras.datasets import mnist
    (X_train, y_train), _ = mnist.load_data()
    class_names = [str(i) for i in range(10)]

    if wandb.run is not None:
        wandb.finish()
    run = wandb.init(project=PROJECT_NAME, name='data_exploration', tags=['2.1'])
    columns = ['class_id', 'class_name', 'image']
    table = wandb.Table(columns=columns)
    for c in range(10):
        idxs = np.where(y_train == c)[0][:5]
        for idx in idxs:
            img = wandb.Image(X_train[idx])
            table.add_data(c, class_names[c], img)
    run.log({'sample_images': table})
    run.finish()
    print("Logged sample images table.")
    print("==============================================================================")


# 2.2 Hyperparameter Sweep
def section_2_2():
    print("==============================================================================")
    print("2.2 Hyperparameter Sweep (300 runs)")
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
        'parameters': {
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 0.0001, 'max': 0.1},
            'optimizer': {'values': ['sgd', 'momentum', 'nag', 'rmsprop']},
            'activation': {'values': ['sigmoid', 'tanh', 'relu']},
            'num_layers': {'values': [2, 3, 4, 5, 6]},
            'hidden_size': {'values': [32, 64, 128]},
            'weight_init': {'values': ['random', 'xavier']},
            'batch_size': {'values': [32, 64]},
            'weight_decay': {'values': [0.0, 0.0005, 0.001]},
            'architecture_pattern': {'values': ['constant', 'funnel']},
        }
    }

    all_results = []

    def sweep_train():
        run = wandb.init()
        cfg = wandb.config

        # if funnel pattern, halve neurons per layer (min 32)
        if cfg.architecture_pattern == 'funnel':
            h_sizes = [max(32, int(cfg.hidden_size / (2**i))) for i in range(cfg.num_layers)]
        else:
            h_sizes = [cfg.hidden_size] * cfg.num_layers

        args = Args(
            dataset='mnist', epochs=10,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            optimizer=cfg.optimizer,
            activation=cfg.activation,
            num_layers=cfg.num_layers,
            hidden_size=h_sizes,
            weight_init=cfg.weight_init,
            weight_decay=cfg.weight_decay,
        )
        X_train, y_train, X_val, y_val, X_test, y_test = load_data('mnist')
        model = NeuralNetwork(args)
        best_weights = model.train(X_train, y_train, X_val, y_val,
                                   epochs=args.epochs, batch_size=args.batch_size,
                                   wandb_run=run)
        if best_weights:
            model.set_weights(best_weights)
        test_acc, test_f1 = model.evaluate_metrics(X_test, y_test)
        run.summary['test_accuracy'] = test_acc
        run.summary['test_f1'] = test_f1
        all_results.append({
            'test_accuracy': float(test_acc),
            'test_f1': float(test_f1),
            'weights': best_weights,
            'optimizer': cfg.optimizer,
            'activation': cfg.activation,
            'num_layers': cfg.num_layers,
            'hidden_size': h_sizes,  
            'architecture_pattern': cfg.architecture_pattern,
            'learning_rate': cfg.learning_rate,
            'batch_size': cfg.batch_size,
            'weight_init': cfg.weight_init,
            'weight_decay': cfg.weight_decay,
        })

    sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME)
    wandb.agent(sweep_id, function=sweep_train, count=300)

    if wandb.run is not None:
        wandb.finish()

    # sort by f1 and pick top 3
    all_results.sort(key=lambda x: x['test_f1'], reverse=True)
    top3 = all_results[:3]

    # save best model weights
    if top3[0]['weights']:
        np.save(os.path.join(BASE_DIR, 'best_model.npy'), top3[0]['weights'])
        print(f"  Saved best_model.npy (F1={top3[0]['test_f1']:.4f}, Acc={top3[0]['test_accuracy']:.4f})")

    # save configs without the weight arrays
    top3_configs = [{k: v for k, v in c.items() if k != 'weights'} for c in top3]
    with open(os.path.join(BASE_DIR, 'best_configs.txt'), 'w') as f:
        for i, cfg in enumerate(top3_configs):
            f.write(f"Config {i+1} (test_f1={cfg['test_f1']:.4f}, test_acc={cfg['test_accuracy']:.4f})\n")
            for k, v in cfg.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")
    with open(os.path.join(BASE_DIR, 'best_configs.json'), 'w') as f:
        json.dump(top3_configs, f, indent=2)

    # single best config separately
    with open(os.path.join(BASE_DIR, 'best_config.json'), 'w') as f:
        json.dump(top3_configs[0], f, indent=2)
    print(f"Sweep complete. Top 3 configs saved to best_configs.txt")
    for i, cfg in enumerate(top3_configs):
        print(f"#{i+1}: F1={cfg['test_f1']:.4f} Acc={cfg['test_accuracy']:.4f} | {cfg['optimizer']} {cfg['activation']} {cfg['num_layers']}x{cfg['hidden_size']}")
    print("==============================================================================")

# 2.3 Optimizer Showdown
def section_2_3():
    print("==============================================================================")
    print("2.3 Optimizer Comparison")
    for opt in ['sgd', 'momentum', 'nag', 'rmsprop']:
        args = Args(optimizer=opt, epochs=10, num_layers=3,
                    hidden_size=[128, 128, 128], activation='relu')
        model, _, run, _, _ = run_experiment(args, f'optim_{opt}', tags=['2.3'])
        run.finish()
    print("==============================================================================")

# 2.4 Vanishing Gradient
def section_2_4():
    print("==============================================================================")
    print("2.4 Vanishing Gradient Analysis")
    configs = [
        ('relu', 3, [128, 128, 128]),
        ('sigmoid', 3, [128, 128, 128]),
        ('relu', 5, [128]*5),
        ('sigmoid', 5, [128]*5),
    ]
    for act, nl, hs in configs:
        args = Args(optimizer='rmsprop', learning_rate=0.001, epochs=10,
                    activation=act, num_layers=nl, hidden_size=hs)
        name = f'vanishing_{act}_{nl}layers'
        model, _, run, _, _ = run_experiment(args, name, log_details=True, tags=['2.4'])
        run.finish()
    print("==============================================================================")

# 2.5 Dead Neuron Investigation
def section_2_5():
    print("==============================================================================")
    print("2.5 Dead Neuron Investigation")
    
    for act in ['relu', 'tanh']:
        for i in range(3):
            args = Args(activation=act, learning_rate=0.1, optimizer='sgd',
                        epochs=10, num_layers=3, hidden_size=[128, 128, 128])
            model, _, run, _, _ = run_experiment(args, f'dead_neuron_{act}_trial_{i+1}',
                                                 log_details=True, tags=['2.5'])
            run.finish()
    print("==============================================================================")

# 2.6 Loss Function Comparison
def section_2_6():
    print("==============================================================================")
    print("2.6 Loss Function Comparison")
    for loss in ['cross_entropy', 'mean_squared_error']:
        args = Args(loss=loss, optimizer='rmsprop', learning_rate=0.001,
                    epochs=10, num_layers=3, hidden_size=[128, 128, 128])
        name = f'loss_{loss}'
        model, _, run, _, _ = run_experiment(args, name, tags=['2.6'])
        run.finish()
    print("==============================================================================")

# 2.8 Confusion Matrix
def section_2_8():
    print("==============================================================================")
    print("2.8 Confusion Matrix (best model from sweep)")
    best_config_path = os.path.join(BASE_DIR, 'best_config.json')
    best_model_path = os.path.join(BASE_DIR, 'best_model.npy')
    
    if not os.path.exists(best_config_path) or not os.path.exists(best_model_path):
        print(f"ERROR: Run section 2.2 first to generate best_model.npy and best_config.json in {BASE_DIR}")
        return

    with open(best_config_path) as f:
        cfg = json.load(f)
    args = Args(
        optimizer=cfg['optimizer'], activation=cfg['activation'],
        num_layers=cfg['num_layers'],
        hidden_size=cfg['hidden_size'],  
        learning_rate=cfg['learning_rate'],
        batch_size=cfg['batch_size'],
        weight_init=cfg['weight_init'],
        weight_decay=cfg['weight_decay'],
    )
    _, _, _, _, X_test, y_test = load_data('mnist')

    model = NeuralNetwork(args)
    weights = np.load(best_model_path, allow_pickle=True).item()
    model.set_weights(weights)

    if wandb.run is not None:
        wandb.finish()
    run = wandb.init(project=PROJECT_NAME, name='best_model_confusion', config=cfg,
                     tags=['2.8'])
    logits = model.forward(X_test)
    probs = softmax(logits)
    preds = np.argmax(probs, axis=1)
    labels = np.argmax(y_test, axis=1)

    class_names = [str(i) for i in range(10)]
    run.log({'confusion_matrix': wandb.plot.confusion_matrix(
        y_true=labels, preds=preds, class_names=class_names)})

    acc = float(np.mean(preds == labels))
    f1 = f1_score(labels, preds, average='macro')
    run.summary['test_accuracy'] = acc
    run.summary['test_f1'] = f1
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")
    run.finish()
    print("==============================================================================")

# 2.9 Weight Init & Symmetry
def section_2_9():
    print("==============================================================================")
    print("2.9 Weight Initialization & Symmetry")
    for init in ['zeros', 'xavier']:
        args = Args(weight_init=init, optimizer='sgd', learning_rate=0.01,
                    epochs=5, batch_size=64, num_layers=3,
                    hidden_size=[128, 128, 128], activation='relu')
        name = f'symmetry_{init}'
        model, _, run, _, _ = run_experiment(args, name, log_details=True, tags=['2.9'])
        run.finish()
    print("==============================================================================")

# 2.10 Fashion-MNIST Transfer
def section_2_10():
    print("==============================================================================")
    print("2.10 Fashion-MNIST Transfer")
    best_configs_path = os.path.join(BASE_DIR, 'best_configs.json')
    if not os.path.exists(best_configs_path):
        print(f"ERROR: Run section 2.2 first to generate best_configs.json in {BASE_DIR}")
        return
    with open(best_configs_path) as f:
        top3 = json.load(f)
    print(f"Using top 3 configs from MNIST sweep:")
    for i, cfg in enumerate(top3):
        print(f"#{i+1}: {cfg['optimizer']} {cfg['activation']} {cfg['num_layers']}x{cfg['hidden_size']}")
    for i, cfg in enumerate(top3):
        args = Args(
            dataset='fashion_mnist', epochs=15,
            optimizer=cfg['optimizer'], activation=cfg['activation'],
            num_layers=cfg['num_layers'],
            hidden_size=cfg['hidden_size'],
            learning_rate=cfg['learning_rate'],
            batch_size=cfg['batch_size'],
            weight_init=cfg['weight_init'],
            weight_decay=cfg['weight_decay'],
        )
        name = f'fashion_top{i+1}_{cfg["optimizer"]}_{cfg["activation"]}'
        model, best_weights, run, X_test, y_test = run_experiment(args, name, tags=['2.10'])
        # save the best fashion model
        if i == 0 and best_weights:
            np.save(os.path.join(BASE_DIR, 'best_fashion_model.npy'), best_weights)
            with open(os.path.join(BASE_DIR, 'best_fashion_config.json'), 'w') as f:
                json.dump(vars(args), f, indent=2)
            print(f"  Saved best_fashion_model.npy from top config")
        run.finish()
        print("==============================================================================")

SECTIONS = {
    '2.1': section_2_1, '2.2': section_2_2, '2.3': section_2_3, '2.4': section_2_4, '2.5': section_2_5, '2.6': section_2_6, '2.8': section_2_8, '2.9': section_2_9, '2.10': section_2_10,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run W&B report experiments')
    parser.add_argument('--section', default='all')
    a = parser.parse_args()

    if a.section == 'all':
        for name, fn in SECTIONS.items():
            fn()
    elif a.section in SECTIONS:
        SECTIONS[a.section]()
    else:
        print(f"Unknown section: {a.section}")
        print(f"Available: {', '.join(SECTIONS.keys())}, all")
