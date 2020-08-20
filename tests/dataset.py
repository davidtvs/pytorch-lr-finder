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


class SimplePOSTaggerDataset(Dataset):
    """
    Example borrowed from:
    https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
    """

    def __init__(self):
        sentences = [
            "The dog ate the apple",
            "Everybody read that book",
            "The quick brown fox jumps over the lazy dog",
            "He always wore his sunglasses at night",
        ]
        pos_tags = [
            ["DET", "NOUN", "VERB", "DET", "NOUN"],
            ["NOUN", "VERB", "DET", "NOUN"],
            ["DET", "ADJ", "ADJ", "NOUN", "VERB", "ADP", "DET", "ADJ", "NOUN"],
            ["NOUN", "ADV", "VERB", "ADJ", "NOUN", "ADP", "NOUN"],
        ]
        self.data = [v.lower().split() for v in sentences]

        tag_set = set.union(*[set(v) for v in pos_tags])
        self.tag_to_ix = {tag: i for i, tag in enumerate(tag_set)}
        self.tag_to_ix.update({"[UNK]": -1})  # unknown tag, for other special tokens

        self.label = [self.get_indices_of_tags(tags) for tags in pos_tags]

    def get_indices_of_tags(self, tags):
        return torch.LongTensor([self.tag_to_ix[tag] for tag in tags])

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


class RandomDataset(Dataset):
    def __init__(self, length):
        self.data = torch.rand((length, 4))
        self.label = torch.rand((length, 1))

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)
