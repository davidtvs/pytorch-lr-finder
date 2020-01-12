# PyTorch learning rate finder

A PyTorch implementation of the learning rate range test detailed in [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186) by Leslie N. Smith and the tweaked version used by [fastai](https://github.com/fastai/fastai).

The learning rate range test is a test that provides valuable information about the optimal learning rate. During a pre-training run, the learning rate is increased linearly or exponentially between two boundaries. The low initial learning rate allows the network to start converging and as the learning rate is increased it will eventually be too large and the network will diverge.

Typically, a good static learning rate can be found half-way on the descending loss curve. In the plot below that would be `lr = 0.002`.

For cyclical learning rates (also detailed in Leslie Smith's paper) where the learning rate is cycled between two boundaries `(start_lr, end_lr)`, the author advises the point at which the loss starts descending and the point at which the loss stops descending or becomes ragged for `start_lr` and `end_lr` respectively.  In the plot below, `start_lr = 0.0002` and `end_lr=0.2`.

![Learning rate range test](images/lr_finder_cifar10.png)

## Installation

Python 2.7 and above:

```bash
pip install torch-lr-finder
```

Install with the support of mixed precision training (requires Python 3, see also [this section](#Mixed-precision-training)):

```bash
pip install torch-lr-finder -v --global-option="amp"
```

## Implementation details and usage

### Tweaked version from fastai

Increases the learning rate in an exponential manner and computes the training loss for each learning rate. `lr_finder.plot()` plots the training loss versus logarithmic learning rate.

```python
from torch_lr_finder import LRFinder

model = ...
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-7, weight_decay=1e-2)
lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
lr_finder.range_test(trainloader, end_lr=100, num_iter=100)
lr_finder.plot() # to inspect the loss-learning rate graph
lr_finder.reset() # to reset the model and optimizer to their initial state
```

### Leslie Smith's approach

Increases the learning rate linearly and computes the evaluation loss for each learning rate. `lr_finder.plot()` plots the evaluation loss versus learning rate.
This approach typically produces more precise curves because the evaluation loss is more susceptible to divergence but it takes significantly longer to perform the test, especially if the evaluation dataset is large.

```python
from torch_lr_finder import LRFinder

model = ...
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-2)
lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
lr_finder.range_test(trainloader, val_loader=val_loader, end_lr=1, num_iter=100, step_mode="linear")
lr_finder.plot(log_lr=False)
lr_finder.reset()
```

### Notes

- Examples for CIFAR10 and MNIST can be found in the examples folder.
- The optimizer passed to `LRFinder` should not have an `LRScheduler` attached to it.
- `LRFinder.range_test()` will change the model weights and the optimizer parameters. Both can be restored to their initial state with `LRFinder.reset()`.
- The learning rate and loss history can be accessed through `lr_finder.history`. This will return a dictionary with `lr` and `loss` keys.
- When using `step_mode="linear"` the learning rate range should be within the same order of magnitude.

## Additional support for training

### Gradient accumulation

You can set the `accumulation_steps` parameter in `LRFinder.range_test()` with a proper value to perform gradient accumulation:

```python
from torch.utils.data import DataLoader
from torch_lr_finder import LRFinder

desired_batch_size, real_batch_size = 32, 4
accumulation_steps = desired_batch_size // real_batch_size

dataset = ...

# Beware of the `batch_size` used by `DataLoader`
trainloader = DataLoader(dataset, batch_size=real_bs, shuffle=True)

model = ...
criterion = ...
optimizer = ...

lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
lr_finder.range_test(trainloader, end_lr=10, num_iter=100, step_mode="exp", accumulation_steps=accumulation_steps)
lr_finder.plot()
lr_finder.reset()
```

### Mixed precision training

Currently, we use [`apex`](https://github.com/NVIDIA/apex) as the dependency for mixed precision training.
To enable mixed precision training, you just need to call `amp.initialize()` before running `LRFinder`. e.g.

```python
from torch_lr_finder import LRFinder
from apex import amp

# Add this line before running `LRFinder`
model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

lr_finder = LRFinder(model, optimizer, criterion, device='cuda')
lr_finder.range_test(trainloader, end_lr=10, num_iter=100, step_mode='exp')
lr_finder.plot()
lr_finder.reset()
```

Note that the benefit of mixed precision training requires a nvidia GPU with tensor cores (see also: [NVIDIA/apex #297](https://github.com/NVIDIA/apex/issues/297))

Besides, you can try to set `torch.backends.cudnn.benchmark = True` to improve the training speed. (but it won't work for some cases, you should use it at your own risk)
