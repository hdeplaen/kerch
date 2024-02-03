# coding=utf-8
"""
Abstract class allowing for torch.nn.Modules to also host a cache of torch objects
which can also be ported like torch.nn.Parameters are, on GPU for example.
"""
from __future__ import annotations

import torch
from typing import Union, List, Iterable, Any, Type
from abc import ABCMeta

from .module import Module
from ..utils import reverse_dict, extend_docstring, DEFAULT_CACHE_LEVEL


@extend_docstring(Module)
class Cache(Module,
            metaclass=ABCMeta):
    r"""
    :param cache_level: Cache level for saving temporary execution results during the execution. The higher the cache,
        the more is saved. Defaults to ``'normal'``. We refer to the :doc:`/features/cache` documentation for further
        information.
    :type cache_level: str, optional
    """

    _cache_elements = []

    _cache_level_switcher = {"none": 0,
                             "light": 1,
                             "normal": 2,
                             "heavy": 3,
                             "total": 4}

    def __init__(self, *args, **kwargs):
        super(Cache, self).__init__(*args, **kwargs)

        # we initiate the cache
        self._cache = {}
        self.cache_level = kwargs.pop('cache_level', 'normal')

    @property
    def cache_level(self) -> str:
        r"""
        Cache level for saving temporary execution results during the execution. The higher the cache,
        the more is saved. Defaults to ``'normal'`` unless set otherwise during instantiation. The different possible
        values are:

        * ``"none"``: the cache is non-existent and everything is computed on the go.
        * ``"light"``: the cache is very light. For example, only the kernel matrix and statistics of the sample points are saved.
        * ``"normal"``: same as light, but the statistics of the out-of-sample points are also saved.
        * ``"heavy"``: in addition to the statistics, the final kernel matrices of the out-of-sample points are saved.
        * ``"total"``: every step of any computation is saved.

        We refer to the :doc:`/features/cache` documentation for further information.
        """
        switcher = Cache._cache_level_switcher
        inv_switcher = {switcher[k]: k for k in switcher}
        return inv_switcher[self._cache_level]

    @cache_level.setter
    def cache_level(self, val: Union[str, int]):
        val = self._get_level(val)
        self._cache_level = val
        for value in self._cache.values():
            if isinstance(value, Cache):
                value._cache_level = val

    def _get_level(self, level: Union[int, str, None]) -> int:
        r"""
        Transforms the cache level to an int if not already.
        """
        if level is None:
            return self._cache_level
        if isinstance(level, str):
            level = Cache._cache_level_switcher.get(level, "Unrecognized cache level.")
        return level

    def _apply(self, fn, recurse=True):
        r"""
        This if the native function by :external+torch:py:class:`torch.nn.modules.module.Module`, used when porting the
        module. This ensures that the cache is also ported. This is used for example to port the data to the GPU or CPU.

        .. note::
            This method is documented for completeness, but it should never be required to call it directly.

        """
        with torch.no_grad():
            for value in self._cache.values():
                cache_entry = value[2]
                if isinstance(cache_entry, torch.Tensor):
                    cache_entry.data = fn(cache_entry)
                elif isinstance(cache_entry, Cache):
                    cache_entry._apply(fn)
        if recurse:
            for child in self.children():
                child._apply(fn, recurse=recurse)
        return super(Cache, self)._apply(fn)

    def _save(self, key, fun, level_key=None, default_level: str = 'total', force: bool = False,
              persisting=False) -> Any:
        r"""
        Saves an element in the cache.

        :param key: key of the cache element.
        :param fun: function to compute the element if not in the cache already.
        :param level_key: key referencing the default level to use in ``kerch.DEFAULT_CACHE_LEVELS``. If not specified, the ``default_level`` argument is used.
        :param default_level: level where to save the cache element. Defaults to 'total'.
        :param force: if the value is ``True``, the element will nevertheless be saved whatever level is specified. Defaults to ``False``.
        :param persisting: These values are meant to persist after a cache reset when calling
            :py:meth:`~kerch.feature.Cache.reset` with ``reset_persisting=False``. Defaults to ``False``.

        :type key: str
        :type fun: function handle
        :type level_key: str, optional
        :type default_level: str, optional
        :type force: bool, optional
        :type persisting: bool, optional
        :return: The result of ``fun()``
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

    def _get(self, key, fun=None, level_key=None, default_level: str = 'normal', force: bool = False,
             overwrite: bool = False, persisting=False, destroy=False) -> Any:
        r"""
        Retrieves an element from the cache. If the element is not present, it saved to the cache provided its level
        is lower or equal to the default level. This can be overwritten by the overwrite argument.

        :param key: key of the cache element.
        :param fun: function to compute the element if not in the cache already.
        :param level_key: key referencing the default level to use in ``kerch.DEFAULT_CACHE_LEVELS``. If not specified, the ``default_level`` argument is used.
        :param default_level: level where to save the cache element. Defaults to 'normal'.
        :param force: if the value is ``True``, the element will nevertheless be saved whatever level is specified. Defaults to ``False``.
        :param persisting: These values are meant to persist after a cache reset when calling
            :py:meth:`~kerch.feature.Cache.reset` with ``reset_persisting=False``. Defaults to ``False``.
        :param destroy: This destroys the value from the cache after being read/computed. This is meant for short-term
            memory. Defaults to ``False``.

        :type key: str
        :type level_key: str, optional
        :type default_level: str, optional
        :type fun: function handle
        :type force: bool, optional
        :type overwrite: bool, optional
        :type persisting: bool, optional
        :type destroy: bool, optional

        :return: The result of ``fun()``
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
        val = self._save(key=key, fun=fun, level_key=level_key, default_level=default_level,
                         force=force, persisting=persisting)
        if destroy:
            self._remove_from_cache(key)
        return val

    def _reset_cache(self, reset_persisting: bool = True, avoid_classes: list | None = None) -> None:
        r"""
        This just resets the cache and makes it empty.

        :param reset_persisting: Persisting elements are meant to resist to a cache reset
            (see :py:meth:`~kerch.feature.Cache._save`). The option allows to also reset them if ``True``. Defaults to ``True``.
        :type reset_persisting: bool, optional
        :param avoid_classes: Class of which the elements must be avoided to be resetted. Default to ``[]``.
        :type avoid_classes: list(type(:class:`~kerch.feature.Cache`)), optional

        .. note::
            This method is documented for completeness, but it should never be required to call it directly.

        """
        elements_avoided = list()
        if avoid_classes is not None:
            for cl in avoid_classes:
                elements_avoided.extend(cl._cache_elements)

        if reset_persisting and len(elements_avoided) == 0:
            self._logger.debug("The cache is fully resetted.")
            self._cache = {}
        else:
            self._logger.debug("The cache is resetted at the exception of the persisting elements and avoided classes.")
            for key in list(self._cache.keys()):
                if key in self._cache and key not in elements_avoided:
                    val = self._cache[key]
                    if not val[1] or reset_persisting:
                        self._cache.pop(key)

    def _clean_cache(self, max_level: Union[str, int, None] = None):
        r"""
        Cleans all cache elements above a certain level. This is relevant for cleaning the cache elements that have
        been forced (see :py:meth:`~kerch.feature.Cache._save`).

        :param max_level: all levels above this level will be cleaned, ``max_level`` excluded. Defaults to the default
            cache level.
        :type max_level: str | int, optional

        .. note::
            This method is documented for completeness, but it should never be required to call it directly.

        """
        max_level = self._get_level(max_level)
        self._logger.debug(f"The cache is cleaned for levels {max_level} and above).")
        # for val in self._cache.values(): del val
        if max_level == 0:
            self._reset_cache()
        else:
            for key in list(self._cache.keys()):
                if key in self._cache.keys():
                    value = self._cache[key]
                    if value[0] > max_level:
                        self._cache.pop(key)

    def _remove_from_cache(self, key: Union[str, List[str]]) -> None:
        r"""
        Removes one or more specific element(s) from the cache.

        :param key: Key(s) of the cache element to be removed.
        :type key: str | list[str]

        .. note::
            This method is documented for completeness, but it should never be required to call it directly.

        """

        def _remove_entry(key_item) -> None:
            if key_item in self._cache:
                self._cache.pop(key_item)

        if not hasattr(key, '__iter__'):
            key = [key]

        for key_item in key:
            _remove_entry(key_item)

    def reset(self, recurse=False, reset_persisting=True) -> None:
        r"""
        Resets the cache to be empty. We refer to the :doc:`/features/cache` documentation for more information.

        :param recurse: If ``True``, resets the cache of this module and also of its potential children. otherwise,
            it only resets the cache for this module. Defaults to ``True``.
        :type recurse: bool, optional
        :param reset_persisting: Persisting elements are meant to resist to a cache reset (see
            :py:meth:`~kerch.feature.Cache._save`). The option allows to also reset them if ``True``. Defaults to
            ``True``.
        :type reset_persisting: bool, optional
        """
        self._reset_cache(reset_persisting=reset_persisting)
        if recurse:
            for cache in self.children():
                if isinstance(cache, Cache):
                    cache.reset(recurse=recurse, reset_persisting=reset_persisting)

    def cache_keys(self, private: bool = False) -> Iterable[str]:
        r"""
        Returns an iterable containing the different cache keys.
        We refer to the :doc:`/features/cache` documentation for more information.

        :param private: Some cache elements are private and are not returned unless set to ``True``. Defaults to ``False``.
        :type private: bool, optional
        """
        for key in self._cache.keys():
            if key[0] != "_" or private:
                yield key

    def print_cache(self, private: bool = False) -> None:
        r"""
        Prints the cache content. We refer to the :doc:`/features/cache` documentation for further information.

        :param private: Some cache elements are private and are not returned unless set to ``True``. Defaults to ``False``.
        :type private: bool, optional
        """
        from ..transform import TransformTree
        transforms = list()

        for key, val in self._cache.items():
            if key[0] != "_" or private:
                if isinstance(val[2], TransformTree):
                    transforms.append((key, val[2]))
                print(key, end=' [')
                level = reverse_dict(Cache._cache_level_switcher).get(val[0])
                print(level, end='] ')
                if val[1]:
                    print('persisting', end='')
                print(end=': ')
                print(val[2])

        for key, transform in transforms:
            print('\n' + key.upper() + ":")
            transform.print_cache(private)
