"""
Abstract class defining a general module in the toolbox.
"""

import torch
from abc import ABCMeta, abstractmethod

from ._logger import _Logger
from .utils import capitalize_only_first


class _Module(_Logger,
              torch.nn.Module,
              object,
              metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, **kwargs):
        # for some obscure reason, calling the super init does not lead to the call of both classes.
        # by consequence, we make the calls manually to each parents
        torch.nn.Module.__init__(self)
        _Logger.__init__(self, **kwargs)

    def __repr__(self):
        return capitalize_only_first(self.__str__())

    def set_log_level(self, level: int = None) -> int:
        level = super(_Module, self).set_log_level(level)
        for child in self.children():
            if isinstance(child, _Logger):
                child.set_log_level(level)
        return level
