# coding=utf-8
"""
Abstract class allowing for torch.nn.Modules to also host a cache of torch objects
which can also be ported like torch.nn.Parameters are, on GPU for example.
"""

import torch
from typing import Union, List, Iterable
from abc import ABCMeta, abstractmethod

from .module import Module
from ..utils import kwargs_decorator, extend_docstring, DEFAULT_CACHE_LEVEL


@extend_docstring(Module)
class Cache(Module,
            metaclass=ABCMeta):
    r"""
    :param cache_level: level for a cache, defaults to 'normal'.
    :type cache_level: str
    """

    _cache_level_switcher = {"none": 0,
                             "light": 1,
                             "normal": 2,
                             "heavy": 3,
                             "total": 4}

    @abstractmethod
    @kwargs_decorator({
        "cache_level": "normal"
    })
    def __init__(self, *args, **kwargs):
        super(Cache, self).__init__(*args, **kwargs)

        # we initiate the cache
        self._cache = {}
        self.cache_level = kwargs["cache_level"]

    @property
    def cache_level(self) -> str:
        r"""
        Cache level possible values (defaults to `"normal"`):

        * `"none"`: the cache is non-existent and everything is computed on the go.
        * `"light"`: the cache is very light. For example, only the kernel matrix and statistics of the sample points are saved.
        * `"normal"`: same as light, but the statistics of the out-of-sample points are also saved.
        * `"heavy"`: in addition to the statistics, the final kernel matrices of the out-of-sample points are saved.
        * `"total"`: every step of any computation is saved.
        """
        switcher = Cache._cache_level_switcher
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
            level = Cache._cache_level_switcher.get(level, "Unrecognized cache level.")
        return level

    def _apply(self, fn):

        r"""
        This if the native function by torch.nn.Module when ported. Here, the sole porting
        of the cache is also added. This is used for example to port the data to the gpu or cpu.
        """
        with torch.no_grad():
            for value in self._cache.values():
                cache_entry = value[2]
                if isinstance(cache_entry, torch.Tensor):
                    cache_entry.data = fn(cache_entry)
                elif isinstance(cache_entry, Cache):
                    cache_entry._apply(fn)
        return super(Cache, self)._apply(fn)

    def _save(self, key, fun=None, level_key=None, default_level: str = 'total', force: bool = False, persisting=False):
        r"""
        We refer to _get.
        """

        # determine which level to store in
        try:
            level = DEFAULT_CACHE_LEVEL[level_key]
        except KeyError:
            level = default_level
        level = self._get_level(level)

        # we now compute, store if the level corresponds and return the value
        assert callable(fun) is not None, \
            f"Cannot store {key} in the cache as no callable argument fun has been provided"
        val = fun()
        if level <= self._cache_level or force:
            self._cache[key] = (level, persisting, val)
        return val

    def _get(self, key, fun=None, level_key=None, default_level: str = 'total', force: bool = False,
             overwrite: bool = False, persisting=False):
        r"""
        Retrieves an element from the cache. If the element is not present, it saved to the cache provided its level
        is lower or equal to the default level. This can be overwritten by the overwrite argument.

        :param key: key of the cache element
        :param fun: function to compute the element
        :param level_key: key referencing the level in _Cache._cache_table. If not specified, the default_key is used.
        :param default_level: level where to save the cache element. Defaults to 'total'.
        :param force: if the value is True, the element will nevertheless be saved whatever level
        :param overwrite: forces the recomputation of the memory element and overwriting if a previous value has been
            saved.
        :param persisting: These values are meant to persisting after a cache reset when calling
            _cache_reset(reset_persisting=False). Defaults to False.

        :type key: str
        :type level_key: str, optional
        :type default_level: str, optional
        :type fun: function handle
        :type force: bool, optional
        :type overwrite: bool, optional
        :type persisting: bool, optional
        """

        # a cache element is represented by the tuple(level, persisting, value)

        # we first check if the value is already in the cache
        if not overwrite:
            if key in self._cache:
                return self._cache[key][2]
            # elif type(key) is tuple:
            #     reverted_key = (key[1], key[0])
            #     if reverted_key in self._cache:
            #         return self._cache[reverted_key][1].T

        # if it is not, we save it
        return self._save(key=key, fun=fun, level_key=level_key, default_level=default_level,
                          force=force, persisting=persisting)

    def _reset_cache(self, reset_persisting=True) -> None:
        r"""
        this just resets the cache and makes it empty. If refilled, the new elements
        will be on the same support as the rest of the level as created by its parameters.
        """

        if reset_persisting:
            self._logger.debug("The cache is fully resetted.")
            # for val in self._cache.values(): del val
            self._cache = {}
        else:
            self._logger.debug("The cache is resetted at the exception of the persisting elements.")
            for key, val in list(self._cache.items()):
                if not val[1]:
                    del self._cache[key]

    def _clean_cache(self, max_level: Union[str, int, None] = None):
        r"""
        Cleans all cache elements that have been forced.

        :param max_level: all levels above this level will be cleaned, max_level excluded. Defaults to the default
            cache level.
        """
        max_level = self._get_level(max_level)
        self._logger.debug(f"The cache is cleaned for levels {max_level} and above).")
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

    def reset(self, children=False, reset_persisting=True) -> None:
        r"""
        Resets the cache to empty.

        :param children: TODO
        :param reset_persisting: Indicates whether the elements for which persisting=True should also be resetted.
            Defaults to True.
        :type reset_persisting: bool, optional
        """
        self._reset_cache(reset_persisting=reset_persisting)
        for cache in self.children():
            if isinstance(cache, Cache):
                cache.reset(children=children, reset_persisting=reset_persisting)

    def cache_keys(self, private: bool = False) -> Iterable[str]:
        r"""
        Returns an iterable containing the different cache keys.

        :param private: Some cache elements are private and are not returned unless set to True. Defaults to False.
        :type private: bool, optional
        """
        for key in self._cache.keys():
            if key[0] != "_" or private:
                yield key

    def print_cache_keys(self, private: bool = False) -> None:
        r"""
        Prints different cache keys.

        :param private: Some cache elements are private and are not printed unless set to True. Defaults to False.
        :type private: bool, optional
        """
        for key in self.cache_keys(private):
            print(key)
