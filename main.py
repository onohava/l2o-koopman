import argparse
import os

from torch.utils.tensorboard import SummaryWriter

from src.models.koopman import koopmanAE
from src.models.dmd import DMDEmbedding

from src.core.trainer import *
from src.core.evaluator import *
from src.models.optimizee import (
    MNISTNet, MNISTLoss,
    QuadraticLoss, QuadOptimizee,
    CIFARLoss, CIFARNet
)

from src.utils.serialization import save_checkpoint

parser = argparse.ArgumentParser(description='L2O with Koopman/DMD')
parser.add_argument('--dataset', type=str, default='quadratics', choices=['mnist', 'quadratics', 'cifar'])
parser.add_argument('--method', type=str, default='l2o', choices=['l2o', 'kae', 'dmd'])
parser.add_argument('--log_dir', type=str, default='runs')
parser.add_argument('--epochs', type=int, default=5)
args = parser.parse_args()


def get_param_count(optimizee_cls):
    dummy = optimizee_cls()
    return sum(p.numel() for p in dummy.parameters())


def run_experiment(dataset_name, method_name, log_dir):
    # Configuration based on Dataset
    window_size = None
    Optimizee = None
    n_epochs = None
    optim_it = None
    lr = None
    TargetLoss = None

    if dataset_name == 'quadratics':
        TargetLoss = QuadraticLoss
        Optimizee = QuadOptimizee
        lr = 0.003
        window_size = 8
        n_epochs = args.epochs
        optim_it = 100
    elif dataset_name == 'mnist':
        TargetLoss = MNISTLoss
        Optimizee = MNISTNet
        lr = 0.01
        window_size = 8
        n_epochs = args.epochs
        optim_it = 100
    elif dataset_name == 'cifar':
        TargetLoss = CIFARLoss
        Optimizee = CIFARNet
        lr = 0.01
        window_size = 8
        n_epochs = args.epochs
        optim_it = 100

    writer = SummaryWriter(log_dir=f"{log_dir}/{dataset_name}_{method_name}")

    # Setup Model (L2O, KAE, or DMD)
    model = None
    if method_name == 'kae':
        input_dim = 2 * window_size
        print(f"Initializing KAE | Input Dim: {input_dim}")
        model = koopmanAE(input_dim, 1, 8, window_size, window_size)

    elif method_name == 'dmd':
        print(f"Initializing DMD | Rank: 4")
        model = DMDEmbedding(window_size=window_size, rank=4)

    # Train
    print(f"Starting Training: {dataset_name} | {method_name}")
    loss, best_state = fit_optimizer(
        TargetLoss, Optimizee,
        lr=lr, n_epochs=n_epochs,
        optim_it=optim_it,
        koopman_model=model,
        writer=writer
    )
    writer.close()

    save_checkpoint(log_dir, dataset_name, method_name, best_state, model)

    # Final Evaluation
    fit_data = run_evaluation(TargetLoss, Optimizee, best_state, model, optim_it)
    return fit_data


def run_evaluation(TargetLoss, Optimizee, best_state, model, optim_it):
    print("Evaluating against baselines...")
    # Baselines
    QUAD_LRS = [0.1, 0.01]
    fit_data = np.zeros((10, optim_it, len(OPT_NAMES) + 1))  # 10 tests

    # Standard Optimizers
    for i, ((opt, extra_kwargs), lr) in enumerate(zip(NORMAL_OPTS, QUAD_LRS)):
        np.random.seed(0)
        fit_data[:, :, i] = np.array(
            fit_normal(TargetLoss, Optimizee, opt, lr=lr, n_epochs=1, n_tests=10, **extra_kwargs))

    # Learned L2O
    if model is not None:
        latent_dim = model.get_latent_dim(False)
    else:
        latent_dim = 0

    opt = w(Optimizer(latent_dim))
    opt.load_state_dict(best_state)

    # Run Test
    l2o_results = [
        do_fit(opt, None, TargetLoss, Optimizee, 1, optim_it, out_mul=1.0, should_train=False, koopman_model=model)
        for _ in range(10)]
    fit_data[:, :, len(OPT_NAMES)] = np.array(l2o_results)

    return fit_data


if __name__ == "__main__":
    fit_data = run_experiment(args.dataset, args.method, args.log_dir)