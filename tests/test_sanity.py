import unittest
import torch
import sys
import os
from types import SimpleNamespace

sys.path.append(os.getcwd())

from src.core.window import KAEWindow, DMDWindow, BaseWindow

from src.models.koopman import koopmanAE


class TestL2OKoopman(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cpu')
        self.window_size = 8
        self.n_params = 10
        # Input dim per snapshot = Params + 1 Loss
        self.snapshot_dim = self.n_params + 1
        # Full window flat input dim
        self.flat_input_dim = self.snapshot_dim * self.window_size

    def test_base_fixed_window_logic(self):
        """
        NEW: Verifies the BaseWindow acts as a Fixed Window
        (concatenating raw history).
        """
        print("\nTesting Base (Fixed) Window Logic...")
        window = BaseWindow(self.window_size, self.device)

        # 1. Test Warm-up (Window not full)
        theta = torch.randn(self.n_params)
        loss = 1.0
        out = window.push_and_encode(theta, loss)

        expected_zeros_size = self.flat_input_dim
        self.assertEqual(out.numel(), expected_zeros_size,
                         "Fixed Window should return full-size zero vector during warm-up")
        self.assertTrue(torch.all(out == 0), "Fixed Window should be zeros during warm-up")

        for i in range(self.window_size - 1):
            window.push_and_encode(torch.randn(self.n_params), float(i))

        # 3. Test Full Window Output
        out_full = window.push_and_encode(torch.randn(self.n_params), 10.0)
        self.assertEqual(out_full.shape[0], self.flat_input_dim,
                         f"Fixed Window output size mismatch. Got {out_full.shape[0]}, expected {self.flat_input_dim}")
        self.assertFalse(torch.all(out_full == 0), "Fixed Window should contain data when full")

    def test_kae_window_logic(self):
        """
        Verifies the KAE Window buffer handles padding and encoding.
        """
        print("\nTesting KAE Window Buffer...")

        model = koopmanAE(self.flat_input_dim, 1, 8, self.window_size, self.window_size)
        window = KAEWindow(model, self.window_size, self.device)

        # Push data until full
        for i in range(self.window_size + 1):
            theta = torch.randn(self.n_params)
            loss = float(i)
            out = window.push_and_encode(theta, loss)

            if i < self.window_size - 1:  # -1 because KAEWindow returns zeroes if NOT ready
                # Logic check: window.py returns zeros if not ready
                self.assertTrue(torch.all(out == 0), f"Window outputted data at step {i} before being full!")
            elif i >= self.window_size:
                # Should return valid vector
                self.assertEqual(out.shape[0], 7, "Window output content shape mismatch")

    def test_dmd_logic(self):
        """
        Verifies the DMD logic computes eigenvalues.
        """
        print("\nTesting DMD Logic...")
        rank = 4
        latent_dim = rank * 2

        # FIX: Create a config object as expected by DMDWindow
        dmd_config = SimpleNamespace(
            rank=rank,
            latent_dim=latent_dim
        )

        window = DMDWindow(dmd_config, self.window_size, self.device)
        out = None

        # Push enough data to fill window and trigger DMD
        for i in range(self.window_size + 5):
            theta = torch.randn(self.n_params)
            loss = float(i)
            out = window.push_and_encode(theta, loss)

        self.assertEqual(out.shape[0], latent_dim,
                         f"DMD output shape mismatch. Got {out.shape[0]}, expected {latent_dim}")

        # Verify it's not all zeros (DMD worked)
        self.assertFalse(torch.all(out == 0), "DMD produced all zeros - SVD might have failed or input was empty")


if __name__ == '__main__':
    unittest.main()