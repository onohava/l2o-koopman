import unittest
import torch
import sys
import os

sys.path.append(os.getcwd())

from src.core.window import KAEWindow, DMDWindow
from src.models.koopman import koopmanAE
from src.models.dmd import DMDEmbedding
from src.models.optimizer import Optimizer
from src.models.optimizee import QuadOptimizee


class TestL2OKoopman(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cpu')
        self.window_size = 8
        self.n_params = 10  # Example for Quadratics
        # Input dim = (Params + 1 Loss) * Window Size
        self.input_dim = (self.n_params + 1) * self.window_size

    def test_kae_shapes(self):
        """
        CRITICAL: Verifies the Koopman Autoencoder accepts the correct input size
        and produces the expected latent vector.
        """
        print("\nTesting Koopman Autoencoder Dimensions...")
        latent_dim = 5

        # 1. Initialize Model
        model = koopmanAE(self.input_dim, 1, latent_dim, self.window_size, self.window_size)
        model.to(self.device)

        # 2. Check Latent Dimension Reporting
        reported_dim = model.get_latent_dim(preproc=False)
        self.assertEqual(reported_dim, latent_dim - 1, "KAE reported incorrect latent dim size for LSTM")

        # 3. Test Forward Pass (Encoder)
        dummy_input = torch.randn(1, self.input_dim)  # Batch size 1
        z = model.encoder(dummy_input)
        self.assertEqual(z.shape[1], latent_dim,
                         f"Encoder output shape mismatch. Got {z.shape}, expected (1, {latent_dim})")

    def test_kae_window_logic(self):
        """
        Verifies the Window buffer correctly flattens history for the Neural Network.
        """
        print("\nTesting KAE Window Buffer...")
        latent_dim = 5
        model = koopmanAE(self.input_dim, 1, latent_dim, self.window_size, self.window_size)
        window = KAEWindow(model, self.window_size, self.device)

        # 1. Push data until full
        for i in range(self.window_size + 1):
            theta = torch.randn(self.n_params)
            loss = float(i)
            out = window.push_and_encode(theta, loss)

            if i < self.window_size:
                # Should return zeros if not full
                self.assertTrue(torch.all(out == 0), "Window outputted data before being full!")
            else:
                # Should return valid vector
                self.assertEqual(out.shape[0], latent_dim - 1, "Window output content shape mismatch")

    def test_dmd_logic(self):
        """
        Verifies the DMD logic computes eigenvalues without crashing on random data.
        """
        print("\nTesting DMD Logic...")
        rank = 4
        # DMD output size = 2 * rank (Real + Imag parts)
        expected_dim = rank * 2

        dmd_model = DMDEmbedding(self.window_size, rank=rank)
        window = DMDWindow(dmd_model, self.window_size, self.device)

        # 1. Push Data
        for i in range(self.window_size + 2):
            # Use noise to avoid Singular Matrix errors in SVD
            theta = torch.randn(self.n_params)
            loss = float(i)
            out = window.push_and_encode(theta, loss)

        # 2. Check Output
        self.assertEqual(out.shape[0], expected_dim,
                         f"DMD output shape mismatch. Got {out.shape[0]}, expected {expected_dim}")

    def test_optimizer_integration(self):
        """
        Verifies the LSTM Optimizer can accept [Gradient + Context] without crashing.
        """
        print("\nTesting LSTM Meta-Optimizer Integration...")
        context_dim = 8
        hidden_sz = 20

        # 1. Init Optimizer
        opt_net = Optimizer(latent_dim=context_dim, preproc=False, hidden_sz=hidden_sz)

        # 2. Mock Inputs
        # 1 param, size 10 -> so we need input for 10 parameters
        grad = torch.randn(10, 1)  # Gradient column
        context = torch.randn(10, context_dim)  # Context broadcasted to params

        inp = torch.cat([grad, context], dim=1)  # The concatenation happening in train.py

        # 3. Mock Hidden States
        h = [torch.zeros(10, hidden_sz) for _ in range(2)]
        c = [torch.zeros(10, hidden_sz) for _ in range(2)]

        # 4. Forward Pass
        update, new_h, new_c = opt_net(inp, h, c)

        self.assertEqual(update.shape, (10, 1), "Optimizer produced wrong update shape")


if __name__ == '__main__':
    unittest.main()