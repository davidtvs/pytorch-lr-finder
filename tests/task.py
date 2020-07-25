import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import pytest

from model import LinearMLP, LSTMTagger
from dataset import XORDataset, ExtraXORDataset, SimplePOSTaggerDataset


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

        self.model.to(self.device)


class XORTask(BaseTask):
    def __init__(self, batch_size=8, steps=100, validate=False):
        super(XORTask, self).__init__()
        n_total = batch_size * steps
        dataset = XORDataset(n_total)
        if validate:
            n_train = int(n_total * 0.9)
            self.train_loader = DataLoader(
                Subset(dataset, range(n_train)),
                batch_size=batch_size
            )
            self.val_loader = DataLoader(
                Subset(dataset, range(n_train, n_total)),
                batch_size=batch_size
            )
        else:
            self.train_loader = DataLoader(dataset, batch_size=batch_size)
            self.val_loader = None

        self.batch_size = batch_size
        self.model = LinearMLP([8, 4, 1])
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-5)
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda")


class ExtraXORTask(BaseTask):
    def __init__(self, batch_size=8, steps=100, validate=False):
        super(ExtraXORTask, self).__init__()
        n_total = batch_size * steps
        dataset = ExtraXORDataset(n_total, extra_dims=2)
        if validate:
            n_train = int(n_total * 0.9)
            self.train_loader = DataLoader(
                Subset(dataset, range(n_train)),
                batch_size=batch_size
            )
            self.val_loader = DataLoader(
                Subset(dataset, range(n_train, n_total)),
                batch_size=batch_size
            )
        else:
            self.train_loader = DataLoader(dataset, batch_size=batch_size)
            self.val_loader = None

        self.batch_size = batch_size
        self.model = LinearMLP([8, 4, 1])
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-5)
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda")


class DiscriminativeLearningRateTask(BaseTask):
    def __init__(self, batch_size=8, steps=100, validate=False):
        super(DiscriminativeLearningRateTask, self).__init__()
        n_total = batch_size * steps
        dataset = XORDataset(n_total)
        if validate:
            n_train = int(n_total * 0.9)
            self.train_loader = DataLoader(
                Subset(dataset, range(n_train)),
                batch_size=batch_size
            )
            self.val_loader = DataLoader(
                Subset(dataset, range(n_train, n_total)),
                batch_size=batch_size
            )
        else:
            self.train_loader = DataLoader(dataset, batch_size=batch_size)
            self.val_loader = None

        self.batch_size = batch_size
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


class SimplePOSTaggerTask(BaseTask):
    def __init__(self, batch_size=2, steps=100, validate=False):
        super(BaseTask, self).__init__()
        dataset = SimplePOSTaggerDataset()
        n_total = len(dataset)
        assert batch_size <= n_total, "`batch_size` is greater than size of dataset"
        if validate:
            n_train = int(n_total * 0.9)
            self.train_loader = DataLoader(
                Subset(dataset, range(n_train)),
                batch_size=batch_size,
                collate_fn=collate_wrapper,
            )
            self.val_loader = DataLoader(
                Subset(dataset, range(n_train, n_total)),
                batch_size=batch_size,
                collate_fn=collate_wrapper,
            )
        else:
            self.train_loader = DataLoader(
                dataset, batch_size=batch_size, collate_fn=collate_wrapper
            )
            self.val_loader = None

        vocab_set = set.union(*[set(v) for v in dataset.data])
        vocab_to_ix = {vocab: i for i, vocab in enumerate(vocab_set)}
        tagset_size = len(dataset.tag_to_ix)

        self.batch_size = batch_size
        self.model = LSTMTagger(6, 6, tagset_size, vocab_to_ix)
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-5)
        self.criterion = nn.NLLLoss()
        self.device = torch.device("cuda")


class SentenceBatch(object):
    def __init__(self, batch):
        padded_sentences, padded_tags = self.pad(batch)
        self.inputs = [v[0] for v in padded_sentences]
        self.labels = torch.stack([v[1] for v in padded_tags])

    def pad(self, batch):
        sentences = [v[0] for v in batch]
        tags = [v[1] for v in batch]
        max_len = max([len(v) for v in sentences])
        padded_sentences = [self.pad_sentence(v, max_len) for v in sentences]
        padded_tags = [self.pad_tag(v, max_len) for v in tags]
        return padded_sentences, padded_tags

    def pad_sentence(self, sentence, max_len):
        # "[PAD]" indicates a token just for padding sentence
        return sentence + ["[PAD]"] * (max_len - len(sentence))

    def pad_tag(self, tags, max_len):
        # -1 (tag: "[UNK]") for "[PAD]" token
        return torch.cat([tags, torch.LongTensor([-1] * (max_len - len(tags)))])


def collate_wrapper(batch):
    sentence_batch = SentenceBatch(batch)
    return sentence_batch.inputs, sentence_batch.labels
