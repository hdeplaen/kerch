"""
Abstract class allowing for torch.nn.Modules to also host a cache of torch objects
which can also be ported like torch.nn.Parameters are, on GPU for example.
"""

import torch
from typing import Union
from abc import ABCMeta, abstractmethod

from ._module import _Module
from .utils import kwargs_decorator


class _Cache(_Module,
             metaclass=ABCMeta):
    cache_level_switcher = switcher = {"oblivious": 0,
                                       "lightweight": 1,
                                       "normal": 2,
                                       "heavy": 3,
                                       "total": 4}

    @abstractmethod
    @kwargs_decorator({
        "cache_level": "normal"
    })
    def __init__(self, *args, **kwargs):
        super(_Cache, self).__init__(*args, **kwargs)

        # we initiate the cache
        self._cache = {}
        self.cache_level = kwargs["cache_level"]

    @property
    def cache_level(self) -> str:
        r"""
        Cache level possible values (defaults to "normal"):
        * "oblivious": the cache is inexistent and everything is computed on the go.
        * "lightweight": the cache is very light. For example, only the kernel matrix and statistics of the sample
            points are saved.
        * "normal": same as lightweight, but the statistics of the out-of-sample points are also saved.
        * "heavy": in addition to the statistics, the final kernel matrices of the out-of-sample points are saved.
        * "total": every step of any computation is saved.
        """
        switcher = _Cache.cache_level_switcher
        inv_switcher = {switcher[k]: k for k in switcher}
        return inv_switcher[self._cache_level]

    @cache_level.setter
    def cache_level(self, val: Union[str, int]):
        if isinstance(val, str):
            val = _Cache.cache_level_switcher.get(val, "Unrecognized cache level.")
        self._cache_level = val

    def _apply(self, fn):
        # this if the native function by torch.nn.Module when ported. Here, the sole porting
        # of the cache is also added.
        with torch.no_grad():
            for _, cache_entry in self._cache.items():
                cache_entry.sample = fn(cache_entry)
        return super(_Cache, self)._apply(fn)

    def _reset_cache(self):
        # this just resets the cache and makes it empty. If refilled, the new elements
        # will be on the same support as the rest of the module as created by its parameters.
        self._log.debug("The cache is resetted.")
        for val in self._cache.values(): del val
        self._cache = {}

    def _remove_from_cache(self, key: str):
        if key in self._cache:
            del self._cache[key]

    def reset(self, children=False):
        self._reset_cache()
        for cache in self.children():
            if isinstance(cache, _Cache):
                cache.reset(children=children)
