#!/bin/bash

# ==========================================
# L2O-Koopman Experiment Runner
# ==========================================

# 1. Quadratic Functions
echo "Running Quadratics - Baseline (L2O)..."
python main.py --dataset quadratics --method l2o --epochs 10

echo "Running Quadratics - Koopman (KAE)..."
python main.py --dataset quadratics --method kae --epochs 10

echo "Running Quadratics - DMD..."
python main.py --dataset quadratics --method dmd --epochs 10


# 2. MNIST
echo "Running MNIST - Baseline..."
python main.py --dataset mnist --method l2o --epochs 5

echo "Running MNIST - Koopman..."
python main.py --dataset mnist --method kae --epochs 5

echo "Running MNIST - DMD..."
python main.py --dataset mnist --method dmd --epochs 5


# 3. CIFAR-10
echo "Running CIFAR - Baseline..."
python main.py --dataset cifar --method l2o --epochs 5

echo "Running CIFAR - DMD..."
python main.py --dataset cifar --method dmd --epochs 5

echo "Running CIFAR - Koopman..."
python main.py --dataset cifar --method kae --epochs 5

echo "All experiments completed. Run 'tensorboard --logdir runs' to view results."