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
    def __init__(self, kae_model, window_m, device, compression_dim=256):
        """
        compression_dim: The fixed size we want to shrink our parameters down to.
                         The KAE will ALWAYS see vectors of this size.
        """
        super().__init__(window_m, device)
        self.model = kae_model
        self.compression_dim = compression_dim
        self.projection_matrix = None  # Will be created on first push
        self.psi_dim = self.model.get_latent_dim(False)

        expected_input = (compression_dim + 1) * window_m

        if hasattr(self.model, 'input_dim') and self.model.input_dim != expected_input:
            print(f"WARNING: KAE expects input {self.model.input_dim}, but Window produces {expected_input}.")

    def _get_projection_matrix(self, input_dim):
        if hasattr(self.model, 'projection_cache') and self.model.projection_cache is not None:
            self.projection_matrix = self.model.projection_cache
            return self.projection_matrix

        # Johnson–Lindenstrauss lemma
        if self.projection_matrix is None:
            print(f"Creating Random Projection: {input_dim} -> {self.compression_dim}")
            P = torch.randn(input_dim, self.compression_dim, device=self.device)
            P = P / (self.compression_dim ** 0.5)

            self.projection_matrix = P.detach()
            self.projection_matrix.requires_grad = False

            if hasattr(self.model, 'projection_cache'):
                self.model.projection_cache = self.projection_matrix

        return self.projection_matrix

    def push_snapshot(self, theta_flat: torch.Tensor, loss_scalar: float):
        theta_device = theta_flat.detach().to(self.device)

        D = theta_device.numel()
        K = self.compression_dim

        if D > K:
            P = self._get_projection_matrix(D)
            compressed_theta = torch.matmul(theta_device.unsqueeze(0), P).squeeze(0)
        else:
            padding = torch.zeros(K - D, device=self.device)
            compressed_theta = torch.cat([theta_device, padding])

        s_t = torch.cat([compressed_theta,
                         torch.tensor([float(loss_scalar)], device=self.device)])
        self.win.append(s_t)
        return len(self.win) >= self.win.maxlen

    def push_and_encode(self, theta_flat: torch.Tensor, loss_scalar: float) -> torch.Tensor:
        is_ready = self.push_snapshot(theta_flat, loss_scalar)

        if not is_ready:
            return torch.zeros(self.psi_dim, device=self.device)

        x = torch.cat(list(self.win), dim=0)

        model_device = next(self.model.parameters()).device
        if x.device != model_device:
            x = x.to(model_device)

        z = self.model.encoder(x)
        z = z.view(-1)
        psi = z[:self.psi_dim]
        return torch.nan_to_num(psi, nan=0.0)


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