from torch import nn
import torch
from src.core.window import KAEWindow


def gaussian_init_(n_units, std=1):
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std / n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]
    return Omega


class encoderNet(nn.Module):
    def __init__(self, n_inputs: int, b, ALPHA=1):
        super(encoderNet, self).__init__()
        self.N = n_inputs
        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(self.N, 16 * ALPHA)
        self.fc2 = nn.Linear(16 * ALPHA, 16 * ALPHA)
        self.fc3 = nn.Linear(16 * ALPHA, b)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # x is (batch, N) or just (N)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class decoderNet(nn.Module):
    def __init__(self, n_inputs: int, b, ALPHA=1):
        super(decoderNet, self).__init__()
        self.N = n_inputs
        self.b = b
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(b, 16 * ALPHA)
        self.fc2 = nn.Linear(16 * ALPHA, 16 * ALPHA)
        self.fc3 = nn.Linear(16 * ALPHA, self.N)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class dynamics(nn.Module):
    def __init__(self, b, init_scale):
        super(dynamics, self).__init__()
        self.dynamics = nn.Linear(b, b, bias=False)
        self.dynamics.weight.data = gaussian_init_(b, std=1)
        U, _, V = torch.svd(self.dynamics.weight.data)
        self.dynamics.weight.data = torch.mm(U, V.t()) * init_scale

    def forward(self, x):
        x = self.dynamics(x)
        return x


class koopmanAE(nn.Module):
    def __init__(self, input_dim, n, b, steps, steps_back, alpha=1, init_scale=1):
        super(koopmanAE, self).__init__()
        self.steps = steps
        self.steps_back = steps_back
        self.latent_dim = b

        # input_dim should be (Params + 1) * WindowSize
        self.encoder = encoderNet(input_dim, b, ALPHA=alpha)
        self.dynamics = dynamics(b, init_scale)
        self.decoder = decoderNet(input_dim, b, ALPHA=alpha)

        self.projection_cache = None

    def forward(self, x, mode='forward'):
        out = []
        z = self.encoder(x)
        q = z
        if mode == 'forward':
            for _ in range(self.steps):
                q = self.dynamics(q)
                out.append(self.decoder(q))
            out.append(self.decoder(z))
            return out, []
        return [], []

    def get_latent_dim(self, preproc: bool):
        return self.latent_dim - (2 if preproc else 1)

    def create_window(self, device):
        return KAEWindow(self, self.steps, device)