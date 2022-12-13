"""
Author: HENRI DE PLAEN
Date: December 2022
Copyright: KU Leuven
License: MIT

This function applies different operations (currently centering and normalization) to a list of vectors or to a matrix.
It is able to also apply it out-of-sample based on statistics. It is built to avoid computing multiple times the same
statistics. It is constructed by a series of transforms forming each a node of a tree. The computation is recursive
from a leaf (the final expected result) to the root (to develop on the stack the full set of operations to be
performed) and then back from the root to the leaf (effective computation).
"""

from __future__ import annotations
from typing import Union, List
from abc import ABCMeta, abstractmethod

import torch
from torch import Tensor
from torch.nn import Parameter

from .._logger import _Logger
from ..utils.type import EPS

_normalize = 'normalize'
_center = 'center'


class _Transform(_Logger, metaclass=ABCMeta):
    def __init__(self, explicit: bool, type: str, lighweight: bool = True, default_path: bool = False, **kwargs):
        super(_Transform, self).__init__(**kwargs)
        self._parent = None
        self._children: dict = {}
        self._type: str = type
        self._explicit: bool = explicit
        self._default: bool = False
        self._default_path: bool = default_path
        self._lightweight = lighweight

        # DATA
        self._data = None
        self._statistics = None

    def __str__(self):
        if self._default:
            return self._type + " (default)"
        return self._type

    @property
    def parent(self) -> Union[_Transform, None]:
        return self._parent

    @parent.setter
    def parent(self, val: _Transform):
        if self.parent is not None:
            self._log.debug(f"Overwriting parent of child {self.type}.")
        self.parent = val
        if self.type in self.parent.children:
            self._log.debug(f"Overwriting child of type {self.type} in parent {self.parent.type}.")
        self.parent.children[self.type] = self

    @property
    def default(self) -> bool:
        return self._default

    @default.setter
    def default(self, val: bool):
        self._default = val

    @property
    def explicit(self) -> bool:
        return self._explicit

    @property
    def children(self) -> dict:
        return self._children

    def add_child(self, val: _Transform) -> None:
        val.parent = self

    @property
    def type(self) -> str:
        return self._type

    @property
    def is_leaf(self) -> bool:
        return not bool(self.children)

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @abstractmethod
    def _explicit_statistics(self):
        pass

    @abstractmethod
    def _implicit_statistics(self):
        pass

    @abstractmethod
    def _explicit_data(self):
        pass

    @abstractmethod
    def _implicit_data(self):
        pass

    @property
    def data(self):
        if self._data is None:
            if self.explicit:
                data = self._explicit_data()
            else:
                data = self._implicit_data()

            if not self._lightweight or self.default:
                self._data = data
        else:
            return self._data

    @property
    def statistics(self):
        if self._statistics is None:
            if self.explicit:
                statistics = self._explicit_statistics()
            else:
                statistics = self._implicit_statistics()

            if not self._lightweight or self._default_path:
                self._statistics = statistics
        else:
            return self._statistics

    @abstractmethod
    def _explicit_statistics_oos(self, x=None):
        pass

    @abstractmethod
    def _implicit_statistics_oos(self, x=None):
        pass

    @abstractmethod
    def _explicit_data_oos(self, x=None):
        pass

    @abstractmethod
    def _implicit_data_oos(self, x=None, y=None):
        pass

    def statistics_oos(self, x=None, y=None):
        if self.explicit:
            if x is None:
                return self.statistics
            else:
                return self._explicit_statistics_oos(x=x)
        else:  # implicit
            if x is None:
                stat_x = self.statistics
            else:
                stat_x = self._implicit_statistics_oos(x=x)
            if y == x:
                stat_y = stat_x
            elif y is None:
                stat_y = self.statistics
            else:
                stat_y = self._implicit_statistics_oos(x=y)
            return stat_x, stat_y

    def data_oos(self, x=None, y=None):
        if self.explicit:
            return self._explicit_data_oos(x=x)
        else:
            return self._implicit_data_oos(x=x, y=y)


class _CenterTransform(_Transform):
    def __init__(self, explicit: bool, default_path: bool = False):
        super(_CenterTransform, self).__init__(explicit=explicit, type=_center, default_path=default_path)

    def _explicit_statistics(self):
        vec = self.parent.data
        return torch.mean(vec, dim=0)

    def _implicit_statistics(self):
        mat = self.parent.data
        mean = torch.mean(mat, dim=1, keepdim=True)
        mean_tot = torch.mean(mean)
        return mean, mean_tot

    def _explicit_data(self):
        return self.parent.data - self.statistics

    def _implicit_data(self):
        mat = self.parent.data
        mean, mean_tot = self.statistics
        return mat - mean - mean.T + mean_tot

    def _explicit_statistics_oos(self, x=None):
        return self.statistics

    def _implicit_statistics_oos(self, x=None):
        mat_x = self.parent.data_oos(x=x)
        return torch.mean(mat_x, dim=1, keepdim=True)

    def _explicit_data_oos(self, x=None):
        return self.parent.data_oos(x=x) - self.statistics_oos(x=x)

    def _implicit_data_oos(self, x=None, y=None):
        mean_x, mean_y = self.statistics_oos(x=x, y=y)
        mean_tot = self.statistics[1]
        return self.parent.data_oos(x=x, y=y) - mean_x \
               - mean_y.T \
               + mean_tot


class _NormalizeTransform(_Transform):
    def __init__(self, explicit: bool, default_path: bool = False):
        super(_NormalizeTransform, self).__init__(explicit=explicit, type=_normalize, default_path=default_path)

    def _explicit_statistics(self):
        vec = self.parent.data
        return torch.norm(vec, dim=1, keepdim=True)

    def _implicit_statistics(self):
        mat = self.parent.data
        return torch.sqrt(torch.diag(mat))[:, None]

    def _explicit_data(self):
        norm = self.statistics
        return self.parent.data / torch.clamp(norm, min=EPS)

    def _implicit_data(self):
        norm = self.statistics
        return self.parent.data / torch.clamp(norm * norm.T, min=EPS)

    def _explicit_statistics_oos(self, x=None):
        vec = self.parent.data_oos(x=x)
        return torch.norm(vec, dim=1, keepdim=True), vec

    def _implicit_statistics_oos(self, x=None):
        mat_x = self.parent.data_oos(x=x, y=x)
        return torch.sqrt(torch.diag(mat_x))[:, None]

    def _explicit_data_oos(self, x=None):
        norm, vec = self.statistics_oos(x=x)
        return vec / torch.clamp(norm, min=EPS)

    def _implicit_data_oos(self, x=None, y=None):
        mat = self.parent.data_oos(x=x, y=y)
        norm_x, norm_y = self.statistics_oos(x=x, y=y)
        return mat / torch.clamp(norm_x * norm_y.Y, min=EPS)


class TransformTree(_Transform):
    def __init__(self, explicit: bool, data, default_transforms=None, **kwargs):
        super(TransformTree, self).__init__(explicit=explicit, type='base', **kwargs)

        if default_transforms is None:
            default_transforms = []

        self._default_transforms = default_transforms

        # create default tree
        node = self
        for operation in self._default_transforms:
            if operation == _center:
                child = _CenterTransform(explicit=self.explicit, default_path=True)
            elif operation == _normalize:
                child = _NormalizeTransform(explicit=self.explicit, default_path=True)
            else:
                raise NameError("Transform type not recognized.")
            node.add_child(child)
            node = child
        self._default_node = node
        node.default = True

        self._base = data
        self._data = None
        self._data_oos = None

    # TODO: rewrite __str__ to show full tree

    @property
    def default_transforms(self) -> List:
        return self._default_transforms

    def _get_data(self) -> Union[Tensor, Parameter]:
        if callable(self._base):
            return self._base()
        return self._base

    def _explicit_statistics(self):
        return None

    def _implicit_statistics(self):
        return None

    def _explicit_data(self):
        if callable(self._base):
            return self._base()
        return self._base

    def _implicit_data(self):
        if callable(self._base):
            return self._base()
        return self._base

    @property
    def default_data(self) -> Tensor:
        return self._default_node.data

    @property
    def default_statistics(self) -> Tensor:
        return self._default_node.statistics

    def _explicit_statistics_oos(self, x=None):
        pass

    def _implicit_statistics_oos(self, x=None):
        pass

    def _explicit_data_oos(self, x=None):
        if callable(self._data_oos):
            return self._data_oos(x)
        return self._data_oos

    def _implicit_data_oos(self, x=None, y=None):
        assert callable(self._data_oos), "data_fun should be callable in the implicit case as it requires the " \
                                         "computation of data_fun(x,sample) for the centering and data_fun(x,x) for " \
                                         "the normalization."

    def apply_transforms(self, data_fun, x=None, y=None, transforms: List = None) -> Tensor:
        if transforms is None:
            transforms = self._default_transforms

        node = self
        for transform in transforms:
            if transform not in node.children:
                if transform == _center:
                    child = _CenterTransform(explicit=self.explicit)
                elif transform == _normalize:
                    child = _NormalizeTransform(explicit=self.explicit)
                else:
                    raise NameError("Transform type not recognized.")
                node.add_child(child)
                node = child

        self._data_fun = data_fun
        if x is None and y is None:
            sol = node.data
        else:
            sol = node.data_oos(x=x, y=y)
        self._data_fun = None # to avoid blocking the destruction if necessary by the garbage collector
        return sol
