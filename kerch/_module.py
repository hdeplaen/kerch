"""
Abstract class defining a general module in the toolbox.
"""

import torch
from typing import Iterator
from abc import ABCMeta, abstractmethod

from ._logger import _Logger
from .utils import capitalize_only_first


class _Module(_Logger,
              torch.nn.Module,
              object,
              metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        # for some obscure reason, calling the super init does not lead to the call of both classes.
        # by consequence, we make the calls manually to each parents
        torch.nn.Module.__init__(self)
        _Logger.__init__(self, *args, **kwargs)

    def __repr__(self):
        return capitalize_only_first(self.__str__())

    def set_log_level(self, level: int = None) -> int:
        level = super(_Module, self).set_log_level(level)
        for child in self.children():
            if isinstance(child, _Logger):
                child.set_log_level(level)
        return level

    def _euclidean_parameters(self, recurse=True):
        if recurse:
            for module in self.children():
                if isinstance(module, _Module):
                    yield from module._euclidean_parameters(recurse)

    def _stiefel_parameters(self, recurse=True):
        if recurse:
            for module in self.children():
                if isinstance(module, _Module):
                    yield from module._stiefel_parameters(recurse)

    def _slow_parameters(self, recurse=True):
        if recurse:
            for module in self.children():
                if isinstance(module, _Module):
                    yield from module._slow_parameters(recurse)

    def manifold_parameters(self, recurse=True, type='euclidean') -> Iterator[torch.nn.Parameter]:
        switcher = {'euclidean': self._euclidean_parameters,
                    'stiefel': self._stiefel_parameters,
                    'slow': self._slow_parameters}
        gen = switcher.get(type, 'Invalid manifold type.')

        memo = set()
        for p in gen(recurse=recurse):
            if p not in memo:
                memo.add(p)
                yield p
