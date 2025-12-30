import torch
import torch.nn as nn
from collections import deque


class BaseWindow:
    """
    Base class that acts as a Fixed Window (Sliding Window).
    It stores optimization history and returns the flattened raw trajectory
    concatenated into a single vector.
    """

    def __init__(self, window_m, device):
        self.win = deque(maxlen=window_m)  # Strict window size
        self.device = device
        self.window_m = window_m

    def push_snapshot(self, theta_flat: torch.Tensor, loss_scalar: float):
        """
        Stores the current parameter state + loss.
        Returns: True if window is full and ready for processing, False otherwise.
        """
        theta_device = theta_flat.detach().to(self.device)

        s_t = torch.cat([theta_device,
                         torch.tensor([float(loss_scalar)], device=self.device)])
        self.win.append(s_t)
        return len(self.win) >= self.win.maxlen

    def reset(self):
        self.win.clear()

    def push_and_encode(self, theta_flat: torch.Tensor, loss_scalar: float) -> torch.Tensor:
        """
        Default Fixed Window behavior:
        Returns the raw flattened history of the window.
        """
        is_ready = self.push_snapshot(theta_flat, loss_scalar)

        single_snapshot_size = theta_flat.numel() + 1

        full_window_size = self.window_m * single_snapshot_size

        if not is_ready:
            return torch.zeros(full_window_size, device=self.device)

        return torch.cat(list(self.win), dim=0)


class KAEWindow(BaseWindow):
    """
    Window for Koopman Autoencoder (Global with Padding).
    Automatically adapts input size to match the KAE's trained expectation.
    """

    def __init__(self, kae_model, window_m, device):
        super().__init__(window_m, device)
        self.model = kae_model

        if hasattr(self.model, 'input_dim'):
            self.expected_N = self.model.input_dim
        elif isinstance(self.model.encoder, nn.Sequential):
            self.expected_N = self.model.encoder[0].in_features
        elif hasattr(self.model.encoder, 'fc1'):
            self.expected_N = self.model.encoder.fc1.in_features
        else:
            raise AttributeError("KAEWindow: Could not determine model input size.")

        self.psi_dim = self.model.get_latent_dim(False)

    def push_and_encode(self, theta_flat: torch.Tensor, loss_scalar: float) -> torch.Tensor:
        is_ready = self.push_snapshot(theta_flat, loss_scalar)

        if not is_ready:
            return torch.zeros(self.psi_dim, device=self.device)

        flat = torch.cat(list(self.win), dim=0)
        current_N = flat.numel()

        if current_N == self.expected_N:
            x = flat
        elif current_N < self.expected_N:
            padding = torch.zeros(self.expected_N - current_N, device=self.device)
            x = torch.cat([flat, padding])
        else:
            x = flat[:self.expected_N]

        model_device = next(self.model.parameters()).device
        if x.device != model_device:
            x = x.to(model_device)

        z = self.model.encoder(x)

        z = z.view(-1)
        psi = z[:self.psi_dim]

        psi = torch.nan_to_num(psi, nan=0.0, posinf=0.0, neginf=0.0)

        return psi.to(self.device)


class DMDWindow(BaseWindow):
    """
    Window for Dynamic Mode Decomposition.
    Naturally handles any input size via SVD, no padding needed.
    """

    def __init__(self, dmd_config, window_m, device):
        super().__init__(window_m, device)
        self.rank = dmd_config.rank
        self.latent_dim = dmd_config.latent_dim

    def push_and_encode(self, theta_flat: torch.Tensor, loss_scalar: float) -> torch.Tensor:
        is_ready = self.push_snapshot(theta_flat, loss_scalar)

        if not is_ready:
            return torch.zeros(self.latent_dim, device=self.device)

        data = torch.stack(list(self.win), dim=1)

        X = data[:, :-1]
        Y = data[:, 1:]

        try:
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)

            r = min(self.rank, S.shape[0])
            if r == 0: return torch.zeros(self.latent_dim, device=self.device)

            U_r = U[:, :r]
            S_r_inv = torch.diag(1.0 / (S[:r] + 1e-9))
            V_r = Vh[:r, :].T

            A_tilde = U_r.T @ Y @ V_r @ S_r_inv

            eigs = torch.linalg.eigvals(A_tilde)

            magnitudes = torch.abs(eigs)
            idx = torch.argsort(magnitudes, descending=True)
            eigs = eigs[idx]

            out = torch.zeros(self.latent_dim, device=self.device)
            n_eigs = min(len(eigs), self.rank)

            out[:n_eigs] = eigs[:n_eigs].real

            imag_start = self.rank
            imag_end = self.rank + n_eigs
            if imag_end <= self.latent_dim:
                out[imag_start: imag_end] = eigs[:n_eigs].imag

            return out

        except RuntimeError:
            return torch.zeros(self.latent_dim, device=self.device)