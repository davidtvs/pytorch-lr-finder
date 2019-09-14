# PyTorch learning rate finder

A PyTorch implementation of the learning rate range test detailed in [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186) by Leslie N. Smith and the tweaked version used by [fastai](https://github.com/fastai/fastai).

The learning rate range test is a test that provides valuable information about the optimal learning rate. During a pre-training run, the learning rate is increased linearly or exponentially between two boundaries. The low initial learning rate allows the network to start converging and as the learning rate is increased it will eventually be too large and the network will diverge.

Typically, a good static learning rate can be found half-way on the descending loss curve. In the plot below that would be `lr = 0.002`.

For cyclical learning rates (also detailed in Leslie Smith's paper) where the learning rate is cycled between two boundaries `(base_lr, max_lr)`, the author advises the point at which the loss starts descending and the point at which the loss stops descending or becomes ragged for `base_lr` and `max_lr` respectively.  In the plot below, `base_lr = 0.0002` and `max_lr=0.2`.

![Learning rate range test](images/lr_finder_cifar10.png)


## Installation

Python 2.7 and above:
``pip install torch-lr-finder``

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
lr_finder.plot()
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
lr_finder.range_test(trainloader, end_lr=1, num_iter=100, step_mode="linear")
lr_finder.plot(log_lr=False)
```

### Notes

- Examples for CIFAR10 and MNIST can be found in the examples folder.
- `LRFinder.range_test()` will change the model weights and the optimizer parameters. Both can be restored to their initial state with `LRFinder.reset()`.
- The learning rate and loss history can be accessed through `lr_finder.history`. This will return a dictionary with `lr` and `loss` keys.
- When using `step_mode="linear"` the learning rate range should be within the same order of magnitude.
