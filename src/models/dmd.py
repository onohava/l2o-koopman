from src.core.window import DMDWindow

class DMDEmbedding:
    """
    A lightweight wrapper that mimics the KoopmanAE interface
    but uses purely mathematical DMD.
    """
    def __init__(self, window_size, rank=None):
        self.window_size = window_size
        self.rank = rank if rank else window_size - 1
        # Latent dim = Rank (Real parts) + Rank (Imaginary parts)
        self.latent_dim = self.rank * 2

    def get_latent_dim(self, preproc=False):
        return self.latent_dim

    def to(self, device):
        return self

    def eval(self):
        pass

    def train(self):
        pass

    def parameters(self):
        # DMD has no learnable parameters
        return []

    def create_window(self, device):
        return DMDWindow(self, self.window_size, device)