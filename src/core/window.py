import torch
from collections import deque


class BaseWindow:
    """
    Base class that handles the storage of optimization history.
    """
    def __init__(self, window_m, device):
        self.win = deque(maxlen=window_m + 1)
        self.device = device
        self.window_m = window_m

    def push_snapshot(self, theta_flat: torch.Tensor, loss_scalar: float):
        """
        Stores the current parameter state + loss.
        Returns: True if window is full and ready for processing, False otherwise.
        """
        s_t = torch.cat([theta_flat.detach(),
                         torch.tensor([float(loss_scalar)], device=self.device)])
        self.win.append(s_t)
        return len(self.win) >= self.win.maxlen

    def reset(self):
        self.win.clear()


class KAEWindow(BaseWindow):
    """
    Window for Koopman Autoencoder.
    Flattens the history into a single vector for the Encoder.
    """

    def __init__(self, kae_model, window_m, device):
        super().__init__(window_m, device)
        self.model = kae_model
        # N = The input size the encoder expects (dynamically calculated in main.py)
        self.N = self.model.encoder.N
        self.psi_dim = self.model.get_latent_dim(False)

    def push_and_encode(self, theta_flat: torch.Tensor, loss_scalar: float) -> torch.Tensor:
        is_ready = self.push_snapshot(theta_flat, loss_scalar)

        if not is_ready:
            return torch.zeros(self.psi_dim, device=self.device)

        # Flatten the entire deque into one long vector [param_t0, loss_t0, param_t1, loss_t1...]
        flat = torch.cat(list(self.win), dim=0)

        # Pad or truncate to match Encoder input size N
        if flat.numel() < self.N:
            x = torch.zeros(self.N, device=self.device, dtype=flat.dtype)
            x[:flat.numel()] = flat
        else:
            x = flat[:self.N]

        z = self.model.encoder(x)
        z = z.view(-1)
        psi = z[:self.psi_dim]

        psi = torch.nan_to_num(psi, nan=0.0, posinf=0.0, neginf=0.0)
        return psi


class DMDWindow(BaseWindow):
    """
    Window for Dynamic Mode Decomposition.
    Maintains Matrix structure for SVD.
    """
    def __init__(self, dmd_config, window_m, device):
        super().__init__(window_m, device)
        self.rank = dmd_config.rank
        self.latent_dim = dmd_config.latent_dim

    def push_and_encode(self, theta_flat: torch.Tensor, loss_scalar: float) -> torch.Tensor:
        is_ready = self.push_snapshot(theta_flat, loss_scalar)

        if not is_ready:
            return torch.zeros(self.latent_dim, device=self.device)

        # Stack into Matrix X (Features x Time)
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

            # Compute Koopman Matrix Approximation
            # A_tilde = U^T * Y * V * S^-1
            A_tilde = U_r.T @ Y @ V_r @ S_r_inv

            eigs = torch.linalg.eigvals(A_tilde)

            magnitudes = torch.abs(eigs)
            idx = torch.argsort(magnitudes, descending=True)
            eigs = eigs[idx]

            out = torch.zeros(self.latent_dim, device=self.device)
            n_eigs = min(len(eigs), self.rank)

            out[:n_eigs] = eigs[:n_eigs].real
            out[self.rank: self.rank + n_eigs] = eigs[:n_eigs].imag

            return out

        except RuntimeError:
            return torch.zeros(self.latent_dim, device=self.device)