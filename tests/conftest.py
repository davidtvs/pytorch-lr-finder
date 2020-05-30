import pytest
import random
import os
import numpy as np
import torch


class CustomCommandLineOption(object):
    """An object for storing command line options parsed by pytest.

    Since `pytest.config` global object is deprecated and removed in version
    5.0, this class is made to work as a store of command line options for
    those components which are not able to access them via `request.config`.
    """

    def __init__(self):
        self._content = {}

    def __str__(self):
        return str(self._content)

    def add(self, key, value):
        self._content.update({key: value})

    def delete(self, key):
        del self._content[key]

    def __getattr__(self, key):
        if key in self._content:
            return self._content[key]
        else:
            return super(CustomCommandLineOption, self).__getattr__(key)


def pytest_addoption(parser):
    parser.addoption(
        "--cpu_only", action="store_true", help="Forcibly run all tests on CPU."
    )


def pytest_configure(config):
    # Bind a config object to `pytest` module instance
    pytest.custom_cmdopt = CustomCommandLineOption()
    pytest.custom_cmdopt.add("cpu_only", config.getoption("--cpu_only"))

    # Set the random seed so that the tests are reproducible between test runs and
    # hopefully torch and numpy versions. This seed should also allow all range tests
    # with a starting lr of 1e-5 and an ending lr of 1e-1 to run the full test without
    # diverging
    seed = 1
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
