import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import pytest

from model import LinearMLP
from dataset import XORDataset, ExtraXORDataset


def use_cuda():
    if pytest.custom_cmdopt.cpu_only:
        return False
    else:
        return torch.cuda.is_available()


class TaskTemplate(type):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        if hasattr(obj, "__post_init__"):
            obj.__post_init__()
        return obj


class BaseTask(metaclass=TaskTemplate):
    def __init__(self):
        self.batch_size = -1
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.device = None
        self.train_loader = None
        self.val_loader = None

    def __post_init__(self):
        # Check whether cuda is available or not, and we will cast `self.device`
        # to `torch.device` here to make sure operations related to moving tensor
        # would work fine later.
        if not use_cuda():
            self.device = None
        if self.device is None:
            return

        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        elif not isinstance(self.device, torch.device):
            raise TypeError("Invalid type of device.")


class XORTask(BaseTask):
    def __init__(self, validate=False):
        super(XORTask, self).__init__()
        bs, steps = 8, 64
        dataset = XORDataset(bs * steps)
        if validate:
            self.train_loader = DataLoader(Subset(dataset, range(steps - bs)))
            self.val_loader = DataLoader(Subset(dataset, range(steps - bs, steps)))
        else:
            self.train_loader = DataLoader(dataset)
            self.val_loader = None

        self.batch_size = bs
        self.model = LinearMLP([8, 4, 1])
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-5)
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda")


class ExtraXORTask(BaseTask):
    def __init__(self, validate=False):
        super(ExtraXORTask, self).__init__()
        bs, steps = 8, 64
        dataset = ExtraXORDataset(bs * steps, extra_dims=2)
        if validate:
            self.train_loader = DataLoader(Subset(dataset, range(steps - bs)))
            self.val_loader = DataLoader(Subset(dataset, range(steps - bs, steps)))
        else:
            self.train_loader = DataLoader(dataset)
            self.val_loader = None

        self.model = LinearMLP([8, 4, 1])
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-5)
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda")


class DiscriminativeLearningRateTask(BaseTask):
    def __init__(self, validate=False):
        super(DiscriminativeLearningRateTask, self).__init__()
        bs, steps = 8, 64
        dataset = XORDataset(bs * steps)
        if validate:
            self.train_loader = DataLoader(Subset(dataset, range(steps - bs)))
            self.val_loader = DataLoader(Subset(dataset, range(steps - bs, steps)))
        else:
            self.train_loader = DataLoader(dataset)
            self.val_loader = None

        dataset = XORDataset(128)
        self.model = LinearMLP([8, 4, 1])
        self.optimizer = optim.SGD(
            [
                {"params": self.model.net[0].parameters(), "lr": 1e-3},
                {"params": self.model.net[1].parameters(), "lr": 1e-5},
            ],
            lr=1e-5,
            momentum=0.5,
        )
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda")
