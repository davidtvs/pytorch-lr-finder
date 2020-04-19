import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LinearMLP(nn.Module):
    def __init__(self, layer_dim):
        super(LinearMLP, self).__init__()
        io_pairs = zip(layer_dim[:-1], layer_dim[1:])
        layers = [nn.Linear(idim, odim) for idim, odim in io_pairs]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
