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
        torch.nn.Module.__init__(self)
        _logger.__init__(self, **kwargs)

    def set_log_level(self, level: int=None) -> int:
        level = super(_module, self).set_log_level(level)
        for child in self.children():
            if isinstance(child, _logger):
                child.set_log_level(level)
        return level
