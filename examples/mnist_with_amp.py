"""
Train a simple neural net for MNIST dataset with mixed precision training.

Examples
--------
- Run with `torch.amp`:
    ```bash
    $ python mnist_with_amp.py --batch_size=32 --seed=42 --tqdm --amp_backend=torch
    ```
- Run without mixed precision training:
    ```bash
    $ python mnist_with_amp.py --batch_size=32 --seed=42 --tqdm --amp_backend=""
    ```
"""

from argparse import ArgumentParser
import random
import sys
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms

from torch_lr_finder import LRFinder
from apex import amp


SEED = 0

def reset_seed(seed):
    """
    ref: https://forums.fast.ai/t/accumulating-gradients/33219/28
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def simple_timer(func):
    def wrapper(*args, **kwargs):
        st = time.time()
        func(*args, **kwargs)
        print('--- Time taken from {}: {} seconds'.format(
            func.__qualname__, time.time() - st
        ))
    return wrapper


# redirect output from tqdm
def conceal_stdout(enabled):
    if enabled:
        f = open(os.devnull, 'w')
        sys.stdout = f
        sys.stderr = f
    else:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.conv2_drop = nn.Dropout2d()
        self.net = nn.Sequential(
            self.conv1,  # (24, 24, 16)
            nn.MaxPool2d(2),  # (12, 12, 16)
            nn.ReLU(True),
            self.conv2,  # (10, 10, 32)
            self.conv2_drop,
            nn.MaxPool2d(2),  # (5, 5, 32)
            nn.ReLU(True),
        )
        self.fc1 = nn.Linear(5*5*32, 64)
        self.fc2 = nn.Linear(64, 16)

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 5*5*32)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


@simple_timer
def warm_up(trainset):
    trainloader = DataLoader(trainset, batch_size=256, shuffle=True)

    device = torch.device('cuda')
    model = ConvNet()
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
    criterion = nn.NLLLoss()

    conceal_stdout(True)
    lr_finder = LRFinder(model, optimizer, criterion, device='cuda')
    lr_finder.range_test(trainloader, end_lr=10, num_iter=10, step_mode='exp')
    conceal_stdout(False)


@simple_timer
def run_normal(trainset, batch_size, no_tqdm=True):
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda')
    model = ConvNet()
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
    criterion = nn.NLLLoss()

    conceal_stdout(no_tqdm)
    lr_finder = LRFinder(model, optimizer, criterion, device='cuda')
    lr_finder.range_test(trainloader, end_lr=10, num_iter=100, step_mode='exp')
    lr_finder.plot()
    conceal_stdout(no_tqdm and False)


@simple_timer
def run_amp_apex(trainset, batch_size, no_tqdm=True, opt_level='O1'):
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda')
    model = ConvNet()
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
    criterion = nn.NLLLoss()

    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

    conceal_stdout(no_tqdm)
    lr_finder = LRFinder(model, optimizer, criterion, device='cuda', amp_backend='apex')
    lr_finder.range_test(trainloader, end_lr=10, num_iter=100, step_mode='exp')
    lr_finder.plot()
    conceal_stdout(no_tqdm and False)

@simple_timer
def run_amp_torch(trainset, batch_size, no_tqdm=True):
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda')
    model = ConvNet()
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
    criterion = nn.NLLLoss()

    amp_config = {
        'device_type': 'cuda',
        'dtype': torch.float16,
    }
    grad_scaler = torch.cuda.amp.GradScaler()

    conceal_stdout(no_tqdm)
    lr_finder = LRFinder(
        model, optimizer, criterion,
        amp_backend='torch', amp_config=amp_config, grad_scaler=grad_scaler
    )
    lr_finder.range_test(trainloader, end_lr=10, num_iter=100, step_mode='exp')
    lr_finder.plot()
    conceal_stdout(no_tqdm and False)

def parse_args():
    parser = ArgumentParser(add_help=True)
    parser.add_argument('--amp_backend', type=str, default='',
                        help='Backend for auto-mixed precision training, available: '
                        '[torch, apex]. If not specified, amp is disabled.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--data_folder', type=str, default='./data',
                        help='Location of MNIST dataset.')
    parser.add_argument('--cudnn_benchmark', action='store_true',
                        help='Add this flag to make cudnn auto-tuner able to find '
                        'the best algorithm on your machine. This may improve the '
                        'performance when you are running script of mixed precision '
                        'training.')
    parser.add_argument('--tqdm', action='store_true',
                        help='Add this flag to show the output from tqdm.')
    parser.add_argument('--warm_up', action='store_true',
                        help='Add this flag to run a warm-up snippet.')
    parser.add_argument('--opt_level', type=str, default='O1',
                        help='Optimization level for amp. (works only for `apex`)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # turn this mode on may improve the performance on some GPUs
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = datasets.MNIST(args.data_folder, train=True, download=True, transform=transform)

    reset_seed(args.seed)
    if args.warm_up:
        warm_up(trainset)

    if args.amp_backend == '':
        run_normal(trainset, args.batch_size, no_tqdm=not args.tqdm)
    elif args.amp_backend == 'apex':
        run_amp_apex(trainset, args.batch_size, no_tqdm=not args.tqdm, opt_level=args.opt_level)
    elif args.amp_backend == 'torch':
        run_amp_torch(trainset, args.batch_size, no_tqdm=not args.tqdm)
    else:
        print('Unknown amp backend: {}'.format(args.amp_backend))

