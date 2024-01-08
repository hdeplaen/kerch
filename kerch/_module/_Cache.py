"""
Abstract class allowing for torch.nn.Modules to also host a cache of torch objects
which can also be ported like torch.nn.Parameters are, on GPU for example.
"""

import torch
from typing import Union, List, Iterable
from abc import ABCMeta, abstractmethod

from ._Module import _Module
from ..utils import kwargs_decorator, extend_docstring


@extend_docstring(_Module)
class _Cache(_Module,
             metaclass=ABCMeta):
    r"""
    :param cache_level: level for a cache, defaults to 'normal'.
    :type cache_level: str
    """

    _cache_level_switcher = switcher = {"none": 0,
                                        "light": 1,
                                        "normal": 2,
                                        "heavy": 3,
                                        "total": 4}

    _cache_table = {"forward_sample_default_representation": "light",
                    "forward_sample_other_representation": "light",
                    "forward_oos_default_representation": "total",
                    "forward_oos_other_representation": "total",
                    "sample_phi": "light",
                    "sample_C": "light",
                    "sample_K": "light",
                    "I_primal": "normal",
                    "I_dual": "normal",
                    "PPCA_B_primal": "normal",
                    "PPCA_B_dual": "normal",
                    "PPCA_Inv_primal": "normal",
                    "PPCA_Inv_dual": "normal",
                    "sample_projections": "none",
                    "kernel_explicit_projections": "none",
                    "kernel_implicit_projections": "none",
                    "Wasserstein_kernel_dist": "normal",
                    "projections_sample_data_default": "light",
                    "projections_sample_data_nondefault": "total",
                    "projections_sample_statistics_default": "light",
                    "projections_sample_statistics_nondefault": "light",
                    "projections_oos_data_level_default": "heavy",
                    "projections_oos_data_level_nondefault": "total",
                    "projections_oos_statistics_default": "normal",
                    "projections_oos_statistics_nondefault": "normal"}

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
        Cache level possible values (defaults to `"normal"`):

        * `"none"`: the cache is inexistent and everything is computed on the go.
        * `"light"`: the cache is very light. For example, only the kernel matrix and statistics of the sample points are saved.
        * `"normal"`: same as light, but the statistics of the out-of-sample points are also saved.
        * `"heavy"`: in addition to the statistics, the final kernel matrices of the out-of-sample points are saved.
        * `"total"`: every step of any computation is saved.
        """
        switcher = _Cache._cache_level_switcher
        inv_switcher = {switcher[k]: k for k in switcher}
        return inv_switcher[self._cache_level]

    @cache_level.setter
    def cache_level(self, val: Union[str, int]):
        val = self._get_level(val)
        self._cache_level = val

    def _get_level(self, level: Union[int, str, None]) -> int:
        r"""
        Transforms the cache level to an int if not already.
        """
        if level is None:
            return self._cache_level
        if isinstance(level, str):
            level = _Cache._cache_level_switcher.get(level, "Unrecognized cache level.")
        return level

    def _apply(self, fn):
        r"""
        This if the native function by torch.nn.Module when ported. Here, the sole porting
        of the cache is also added. This is used for example to port the data to the gpu or cpu.
        """
        with torch.no_grad():
            for value in self._cache.values():
                cache_entry = value[1]
                if isinstance(cache_entry, torch.Tensor):
                    cache_entry.data = fn(cache_entry)
                elif isinstance(cache_entry, _Cache):
                    cache_entry._apply(fn)
        return super(_Cache, self)._apply(fn)

    def _get(self, key, fun, level_key=None, default_level: str = 'total', force: bool = False,
             overwrite: bool = False):
        r"""
        Retrieves an element from the cache. If the element is not present, it saved to the cache provided its level
        is lower or equal to the default level. This can be overwritten by the overwrite argument.

        :param name: key of the cache element
        :param fun: function to compute the element
        :param level_key: key referencing the level in _Cache._cache_table. If not specified, the default_key is used.
        :param default_level: level where to save the cache element. Defaults to 'total'.
        :param force: if the value is True, the element will nevertheless be saved whatever level
        :param overwrite: forces the recomputation of the memory element and overwriting if a previous value has been
            saved.

        :type key: str
        :type level_key: str, optional
        :type default_level: str, optional
        :type fun: function handle
        :type force: bool
        :type overwrite: bool
        """

        if not overwrite:
            if key in self._cache:
                return self._cache[key][1]
            elif type(key) is tuple:
                reverted_key = (key[1], key[0])
                if reverted_key in self._cache:
                    return self._cache[reverted_key][1].T

        try:
            level = _Cache._cache_table[level_key]
        except KeyError:
            level = default_level

        level = self._get_level(level)

        val = fun()
        if level <= self._cache_level or force:
            self._cache[key] = (level, val)
        return val

    def _reset_cache(self) -> None:
        r"""
        this just resets the cache and makes it empty. If refilled, the new elements
        will be on the same support as the rest of the level as created by its parameters.
        """

        self._log.debug(f"The cache is resetted.")
        # for val in self._cache.values(): del val
        self._cache = {}

    def _clean_cache(self, max_level: Union[str, int, None] = None):
        r"""
        Cleans all cache elements that have been forced.

        :param max_level: all levels above this level will be cleaned, max_level excluded. Defaults to the default
            cache level.
        """
        max_level = self._get_level(max_level)
        self._log.debug(f"The cache is cleaned for levels {max_level} and above).")
        # for val in self._cache.values(): del val
        if max_level == 0:
            self._cache = {}
        else:
            for key, value in self._cache.items():
                if value[0] > max_level:
                    del self._cache[key]

    def _remove_from_cache(self, key: Union[str, List[str]]) -> None:
        r"""
        Removes a specific element from the cache.

        :param key: Key of the cache element to be removed
        """

        def _del_entry(key_item) -> None:
            if key_item in self._cache:
                del self._cache[key_item]

        if not hasattr(key, '__iter__'):
            key = [key]

        for key_item in key:
            _del_entry(key_item)

    def reset(self, children=False) -> None:
        r"""
        Resets the cache to empty.

        :param children: TODO
        """
        self._reset_cache()
        for cache in self.children():
            if isinstance(cache, _Cache):
                cache.reset(children=children)

    def cache_keys(self, private: bool = False) -> Iterable[str]:
        r"""
        Returns an iterable containing the different cache keys.

        :param private: Some cache elements are private and are not returned unless set to True. Defaults to False.
        :type private: bool, optional
        """
        for key in self._cache.keys():
            if key[0] != "_" or private:
                yield key

    def print_keys(self, private: bool = False) -> None:
        r"""
        Prints different cache keys.

        :param private: Some cache elements are private and are not printed unless set to True. Defaults to False.
        :type private: bool, optional
        """
        for key in self.cache_keys(private):
            print(key)
