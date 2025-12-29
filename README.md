# L2O-Koopman: Meta-Learning with Koopman Operator Theory

This repository implements a "Learning to Optimize" (L2O) framework enhanced by Koopman Operator Theory. It investigates whether providing a meta-optimizer (LSTM) with a linearized, global view of the optimization landscape—via a Koopman Autoencoder (KAE) or Dynamic Mode Decomposition (DMD)—improves convergence and generalization compared to standard optimizers.

## Project Overview

The core idea is to augment the state input of the meta-optimizer. Standard L2O typically uses only the gradient. This project adds a "context vector" derived from the trajectory history:
1.  **Vanilla L2O**: Input = `[Gradient]`
2.  **L2O + KAE**: Input = `[Gradient, Latent_State]`. The latent state is learned by a deep autoencoder that enforces linear evolution dynamics ($\psi_{t+1} = K\psi_t$).
3.  **L2O + DMD**: Input = `[Gradient, Eigenvalues]`. The context is the spectral decomposition (eigenvalues) of the optimization path computed via Singular Value Decomposition (SVD).

## Repository Structure

* **`main.py`**: The primary entry point. Experiments, dataset selection, and TensorBoard logging.
* **`train.py`**: Contains the meta-optimization training loop (`fit_optimizer`) and the differentiable inner optimization loop (`do_fit`).
* **`window.py`**: Manages the sliding window of optimization history.
    * `KAEWindow`: Formats history for neural network encoders.
    * `DMDWindow`: Formats history for matrix decomposition (SVD).
* **`dmd.py`**: Implements "Windowed DMD" to extract eigenvalues on the fly during optimization.
* **`models/`**:
    * `koopman.py`: The Koopman Autoencoder architecture.
    * `optimizer.py`: The LSTM Meta-Optimizer.
    * `optimizee.py`: Task definitions (Quadratic functions, MNIST, CIFAR-10).
* **`test.py`**: Utilities for evaluating baseline optimizers (Adam, SGD, etc.).
* **`plots.py`**: Helper functions for visualizing results (optional).

## Supported Datasets

| Dataset | Description | Difficulty | Context |
| :--- | :--- | :--- | :--- |
| **Quadratics** | Synthetic convex regression ($y=x^Tx$). | Easy | 10 Parameters. Ideal for verifying dynamics. |
| **MNIST** | Non-convex MLP for digit classification. | Medium | ~16k Parameters. Tests high-dim handling. |
| **CIFAR-10** | Non-convex CNN (Tiny ConvNet). | Hard | ~2k Parameters. Tests architecture generalization. |

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/l2o-koopman.git](https://github.com/your-username/l2o-koopman.git)
    cd l2o-koopman
    ```

2.  **Install dependencies:**
    ```bash
    pip install torch torchvision numpy matplotlib tqdm tensorboard
    ```

## Usage

Run experiments via the command line. The script automatically handles dimension sizing and logging. We provide simple `experiments.sh` file to run all the experiments. You can copy the parts you want to try.

