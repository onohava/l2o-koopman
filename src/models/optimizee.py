import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torchvision import datasets
import numpy as np
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()

def w(v):
    if USE_CUDA:
        return v.cuda()
    return v


# --- QUADRATICS ---
class QuadraticLoss:
    def __init__(self, **kwargs):
        self.W = w(Variable(torch.randn(10, 10)))
        self.y = w(Variable(torch.randn(10)))

    def get_loss(self, theta):
        return torch.sum((self.W.matmul(theta) - self.y) ** 2)


class QuadOptimizee(nn.Module):
    def __init__(self, theta=None):
        super().__init__()
        if theta is None:
            self.theta = nn.Parameter(torch.zeros(10))
        else:
            self.theta = theta

    def forward(self, target):
        return target.get_loss(self.theta)

    def all_named_parameters(self):
        return [('theta', self.theta)]


# --- MNIST ---
class MNISTLoss:
    def __init__(self, training=True):
        dataset = datasets.MNIST(
            './data', train=True, download=True,
            transform=torchvision.transforms.ToTensor()
        )
        indices = list(range(len(dataset)))
        np.random.RandomState(10).shuffle(indices)
        if training:
            indices = indices[:len(indices) // 2]
        else:
            indices = indices[len(indices) // 2:]

        self.loader = torch.utils.data.DataLoader(
            dataset, batch_size=128,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))

        self.batches = []
        self.cur_batch = 0

    def sample(self):
        if self.cur_batch >= len(self.batches):
            self.batches = []
            self.cur_batch = 0
            for b in self.loader:
                self.batches.append(b)
        batch = self.batches[self.cur_batch]
        self.cur_batch += 1
        return batch


class MNISTNet(nn.Module):
    def __init__(self, layer_size=20, n_layers=1, **kwargs):
        super().__init__()
        if kwargs != {}:
            self.params = kwargs
        else:
            inp_size = 28 * 28
            self.params = {}
            for i in range(n_layers):
                self.params[f'mat_{i}'] = nn.Parameter(
                    torch.randn(inp_size, layer_size) * 0.001)
                self.params[f'bias_{i}'] = nn.Parameter(torch.zeros(layer_size))
                inp_size = layer_size

            self.params['final_mat'] = nn.Parameter(torch.randn(inp_size, 10) * 0.001)
            self.params['final_bias'] = nn.Parameter(torch.zeros(10))

            self.mods = nn.ParameterList()
            for v in self.params.values():
                self.mods.append(v)

        self.activation = nn.Sigmoid()
        self.loss = nn.NLLLoss()

    def all_named_parameters(self):
        return [(k, v) for k, v in self.params.items()]

    def forward(self, loss):
        inp, out = loss.sample()
        inp = w(Variable(inp.view(inp.size()[0], 28 * 28)))
        out = w(Variable(out))

        cur_layer = 0
        while f'mat_{cur_layer}' in self.params:
            inp = self.activation(torch.matmul(inp, self.params[f'mat_{cur_layer}']) + self.params[f'bias_{cur_layer}'])
            cur_layer += 1

        inp = F.log_softmax(torch.matmul(inp, self.params['final_mat']) + self.params['final_bias'], dim=1)
        l = self.loss(inp, out)
        return l


# --- CIFAR  ---
class CIFARLoss:
    def __init__(self, training=True):
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        dataset = datasets.CIFAR10(
            './data', train=True, download=True,
            transform=transform
        )
        indices = list(range(len(dataset)))
        np.random.RandomState(10).shuffle(indices)
        if training:
            indices = indices[:len(indices) // 2]
        else:
            indices = indices[len(indices) // 2:]

        self.loader = torch.utils.data.DataLoader(
            dataset, batch_size=64,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))

        self.batches = []
        self.cur_batch = 0

    def sample(self):
        if self.cur_batch >= len(self.batches):
            self.batches = []
            self.cur_batch = 0
            for b in self.loader:
                self.batches.append(b)
        batch = self.batches[self.cur_batch]
        self.cur_batch += 1
        return batch


class CIFARNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        if kwargs != {}:
            self.params = kwargs
        else:
            self.params = {}
            self.params['conv1_w'] = nn.Parameter(torch.randn(6, 3, 5, 5) * 0.01)
            self.params['conv1_b'] = nn.Parameter(torch.zeros(6))

            self.params['fc1_w'] = nn.Parameter(torch.randn(6 * 14 * 14, 10) * 0.01)
            self.params['fc1_b'] = nn.Parameter(torch.zeros(10))

            self.mods = nn.ParameterList()
            for v in self.params.values():
                self.mods.append(v)

        self.loss = nn.CrossEntropyLoss()

    def all_named_parameters(self):
        return [(k, v) for k, v in self.params.items()]

    def forward(self, loss_obj):
        inp, out = loss_obj.sample()
        inp = w(Variable(inp))
        out = w(Variable(out))

        x = F.conv2d(inp, self.params['conv1_w'], self.params['conv1_b'])
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 6 * 14 * 14)

        x = torch.matmul(x, self.params['fc1_w']) + self.params['fc1_b']

        return self.loss(x, out)