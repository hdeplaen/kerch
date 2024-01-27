# coding=utf-8
from __future__ import annotations
from typing import Union, Tuple
from abc import ABCMeta, abstractmethod

import torch
from ..feature import Cache
from ..utils.errors import BijectionError, ImplicitError
from ..utils import extend_docstring


class Transform(Cache, metaclass=ABCMeta):
    def __init__(self, explicit: bool, name: str, default_path: bool = False, **kwargs):
        logger_name = 'Explicit' if explicit else 'Implicit'
        super(Transform, self).__init__(**{'logger_name': logger_name, **kwargs})
        self._parent = None
        self._offspring: dict = {}
        self._name: str = name
        self._explicit: bool = explicit
        self._default: bool = False
        self._default_path: bool = default_path
        self._set_cache_levels()

    def _set_cache_levels(self):
        # CACHE LEVELS
        if self._default:
            self._sample_data_level = "transform_sample_data_default"
            self._sample_statistics_level = "transform_sample_statistics_default"
            self._oos_data_level = "transform_oos_data_default"
            self._oos_statistics_level = "transform_oos_statistics_default"
        else:
            self._sample_data_level = "transform_sample_data_nondefault"
            self._sample_statistics_level = "transform_sample_statistics_nondefault"
            self._oos_data_level = "transform_oos_data_nondefault"
            self._oos_statistics_level = "transform_oos_statistics_nondefault"

    def __str__(self):
        if self.default:
            return self._name + " (default)"
        return self._name

    @staticmethod
    def _get_names(x=None, y=None) -> Tuple[str, str]:
        if x is not None:
            x_name = id(x)
        else:
            x_name = 0
        if y is not None:
            y_name = id(y)
        else:
            y_name = 0
        return str(x_name), str(y_name)

    @property
    def parent(self) -> Union[Transform, None]:
        return self._parent

    @property
    def default(self) -> bool:
        return self._default

    @default.setter
    def default(self, val: bool):
        self._default = val
        self._set_cache_levels()


    @property
    def explicit(self) -> bool:
        return self._explicit

    @property
    def offspring(self) -> dict:
        return self._offspring

    def add_offspring(self, val: Transform) -> None:
        # add offspring to parent
        if type(val) in self.offspring:
            self._logger.debug(f"Overwriting offspring of type {type(val)} in parent {type(self._parent)}.")
        self._offspring[type(val)] = val

        # add parent to offspring
        if val.parent is not None:
            self._logger.debug(f"Overwriting parent of offspring {type(self)}.")
        val._parent = self

    def add_parent(self, val: Transform) -> None:
        val.add_offspring(self)

    @property
    def is_leaf(self) -> bool:
        return not bool(self.offspring)

    @property
    def is_root(self) -> bool:
        return self.parent is None

    # DATA
    @abstractmethod
    def _explicit_statistics(self, sample):
        pass

    def _implicit_statistics(self, sample, x=None):
        raise BijectionError

    @abstractmethod
    def _explicit_sample(self):
        pass

    def _implicit_sample(self):
        raise ImplicitError

    @property
    def sample(self) -> torch.Tensor:
        return self._get(self._name + " data_sample", level_key=self._sample_data_level,
                         fun=self._explicit_sample if self.explicit else self._implicit_sample)

    def statistics_sample(self, sample=None) -> torch.Tensor:
        def fun(sample):
            if sample is None:
                sample = self.parent.sample
            if self.explicit:
                return self._explicit_statistics(sample=sample)
            else:
                return self._implicit_statistics(sample=sample)

        return self._get(self._name + " statistics_sample", level_key=self._sample_data_level, fun=lambda: fun(sample))

    # OOS
    @abstractmethod
    def _explicit_statistics_oos(self, oos, x=None):
        pass

    def _implicit_statistics_oos(self, oos, x=None):
        raise BijectionError

    @abstractmethod
    def _explicit_oos(self, x=None):
        pass

    def _implicit_oos(self, x=None, y=None):
        raise BijectionError

    def statistics_oos(self, x=None, y=None, oos=None) -> Union[torch.Tensor, (torch.Tensor, torch.Tensor)]:
        x_name, y_name = Transform._get_names(x, y)

        if self.explicit:
            if x is None:
                return self.statistics_sample()
            else:
                def fun(x, oos):
                    if oos is None:
                        oos = self.parent.oos(x=x)
                    return self._explicit_statistics_oos(x=x, oos=oos)

                return self._get(self._name + " statistics_oos_" + x_name, level_key=self._oos_statistics_level,
                                 fun=lambda: fun(x, oos), force=True)
        else:
            def fun(x, oos):
                if oos is None:
                    oos = self.parent.oos(x=x)
                return self._implicit_statistics_oos(x=x, oos=oos)

            if x is None:
                x_stat = self.statistics_sample()
            else:
                x_stat = self._get(self._name + " statistics_oos_" + x_name, level_key=self._oos_statistics_level,
                                   fun=lambda: fun(x, oos), force=True)
            if y is None:
                y_stat = self.statistics_sample()
            else:
                y_stat = self._get(self._name + " statistics_oos_" + y_name, level_key=self._oos_statistics_level,
                                   fun=lambda: fun(y, oos), force=True)
            return (x_stat, y_stat)

    def oos(self, x=None, y=None) -> torch.Tensor:
        x_name, y_name = Transform._get_names(x, y)

        if self.explicit:
            if x is None:
                return self._explicit_sample()
            else:
                return self._get(self._name + " data_oos_" + x_name, level_key=self._oos_data_level,
                                 fun=lambda: self._explicit_oos(x=x))
        else:
            if x is None and y is None:
                return self._implicit_sample()
            elif x is None:  # y is relevant
                return self._get(self._name + ' data_oos_' + y_name, level_key=self._oos_data_level,
                                 fun=lambda: self._implicit_oos(x=y)).T
            elif y is None:  # x is relevant
                return self._get(self._name + ' data_oos_' + x_name, level_key=self._oos_data_level,
                                 fun=lambda: self._implicit_oos(x=x))
            else:
                return self._get(self._name + ' data_oos_' + x_name + '_' + y_name, level_key=self._oos_data_level,
                                 fun=lambda: self._implicit_oos(x=x, y=y))

    def _revert(self, oos):
        if self.explicit:
            return self._revert_explicit(oos)
        else:
            return self._revert_implicit(oos)

    def _revert_explicit(self, oos):
        raise BijectionError

    def _revert_implicit(self, oos):
        raise ImplicitError

    def _implicit_diag(self, x=None):
        return torch.diag(self.oos(x, x))[:, None]

    @property
    @extend_docstring(Cache.cache_level)
    def cache_level(self) -> str:
        switcher = Cache._cache_level_switcher
        inv_switcher = {switcher[k]: k for k in switcher}
        return inv_switcher[self._cache_level]

    @cache_level.setter
    def cache_level(self, val: Union[str, int]):
        val = self._get_level(val)
        self._cache_level = val
        try:
            for o in self._offspring.values():
                o.cache_level = val
        except AttributeError:
            pass

    @extend_docstring(Cache.print_cache)
    def print_cache(self, private: bool = False) -> None:
        super(Transform, self).print_cache(private)
        try:
            for o in self._offspring.values():
                o.print_cache(private)
        except AttributeError:
            pass
    @extend_docstring(Cache._clean_cache)
    def _clean_cache(self, max_level: Union[str, int, None] = None):
        super(Transform, self)._clean_cache(max_level)
        try:
            for o in self._offspring.values():
                o._clean_cache(max_level)
        except AttributeError:
            pass