"""
Abstract class allowing for torch.nn.Modules to also host a cache of torch objects
which can also be ported like torch.nn.Parameters are, on GPU for example.
"""

import torch
from abc import ABCMeta, abstractmethod

from ._module import _Module


class _Cache(_Module,
             metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        super(_Cache, self).__init__(*args, **kwargs)

        # we initiate the cache
        self._reset()

    def _apply(self, fn):
        # this if the native function by torch.nn.Module when ported. Here, the sole porting
        # of the cache is also added.
        with torch.no_grad():
            for _, cache_entry in self._cache.items():
                cache_entry.data = fn(cache_entry)
        return super(_Cache, self)._apply(fn)

    def _reset(self):
        # this just resets the cache and makes it empty. If refilled, the new elements
        # will be on the same support as the rest of the module as created by its parameters.
        self._log.debug("The cache is resetted.")
        self._cache = {}

    def reset(self, children=False):
        self._reset()
        for cache in self.children():
            if isinstance(cache, _Cache):
                cache.reset(children=children)
