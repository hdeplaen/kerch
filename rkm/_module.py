"""
Abstract class defining a general module in the toolbox.
"""

import torch
from abc import ABCMeta, abstractmethod

from ._logger import _logger


class _module(_logger,
              torch.nn.Module,
              object,
              metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, **kwargs):
        # for some obscure reason, calling the super init does not lead to the call of both classes.
        # by consequence, we make the calls manually to each parents
        _logger.__init__(self)
        torch.nn.Module.__init__(self)