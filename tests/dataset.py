import numpy as np
import torch
from torch.utils.data import Dataset


class XORDataset(Dataset):
    def __init__(self, length, shape=None):
        """
        Arguments:
            length (int): length of dataset, which equals `len(self)`.
            shape (list, tuple, optional): shape of dataset. If it isn't
                specified, it will be initialized to `(length, 8)`.
                Default: None.
        """
        _shape = (length,) + tuple(shape) if shape else (length, 8)
        raw = np.random.randint(0, 2, _shape)
        self.data = torch.FloatTensor(raw)

        label = np.bitwise_xor.reduce(raw, axis=1)
        self.label = torch.tensor(label, dtype=torch.float32).unsqueeze(dim=1)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


class ExtraXORDataset(XORDataset):
    """ A XOR dataset which is able to return extra values. """

    def __init__(self, length, shape=None, extra_dims=1):
        """
        Arguments:
            length (int): length of dataset, which equals `len(self)`.
            shape (list, tuple, optional): shape of dataset. If it isn't
                specified, it will be initialized to `(length, 8)`.
                Default: None.
            extra_dims (int, optional): dimension of extra values.
                Default: 1.
        """
        super(ExtraXORDataset, self).__init__(length, shape=shape)
        if extra_dims:
            _extra_shape = (length, extra_dims)
            self.extras = torch.randint(0, 2, _extra_shape)
        else:
            self.extras = None

    def __getitem__(self, index):
        if self.extras is not None:
            retval = [self.data[index], self.label[index]]
            retval.extend([v for v in self.extras[index]])
            return retval
        else:
            return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)
