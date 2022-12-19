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

from kerch._logger import _Logger
from kerch.utils.tensor import equal
from kerch.utils.type import EPS
from kerch.utils.errors import DualError


class _Transform(_Logger, metaclass=ABCMeta):
    def __init__(self, explicit: bool, name: str, lighweight: bool = True, default_path: bool = False, **kwargs):
        super(_Transform, self).__init__(**kwargs)
        self._parent = None
        self._children: dict = {}
        self._name: str = name
        self._explicit: bool = explicit
        self._default: bool = False
        self._default_path: bool = default_path
        self._lightweight = lighweight

        # DATA
        self._data = None
        self._statistics = None
        self._statistics_oos = None

    def __str__(self):
        if self._default:
            return self._name + " (default)"
        return self._name

    @property
    def parent(self) -> Union[_Transform, None]:
        return self._parent

    @parent.setter
    def parent(self, val: _Transform):
        if self.parent is not None:
            self._log.debug(f"Overwriting parent of child {type(self)}.")
        self._parent = val

        if type(val) in self.parent.children:
            self._log.debug(f"Overwriting child of type {type(val)} in parent {type(self._parent)}.")
        self.parent.children[type(self)] = self

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
    def is_leaf(self) -> bool:
        return not bool(self.children)

    @property
    def is_root(self) -> bool:
        return self.parent is None

    # DATA
    @abstractmethod
    def _explicit_statistics(self, data):
        pass

    def _implicit_statistics(self, data, x=None):
        raise DualError

    @abstractmethod
    def _explicit_data(self):
        pass

    def _implicit_data(self):
        raise DualError

    @property
    def data(self) -> Tensor:
        if self._data is None:
            if self.explicit:
                data = self._explicit_data()
            else:
                data = self._implicit_data()

            if not self._lightweight or self.default:
                self._data = data
            return data
        return self._data

    def statistics(self, data=None) -> Tensor:
        if self._statistics is None:
            if data is None:
                data = self.parent.data
            if self.explicit:
                statistics = self._explicit_statistics(data=data)
            else:
                statistics = self._implicit_statistics(data=data)

            if not self._lightweight or self._default_path:
                self._statistics = statistics
        return self._statistics

    # OOS
    @abstractmethod
    def _explicit_statistics_oos(self, data, x=None):
        pass

    def _implicit_statistics_oos(self, data, x=None):
        raise DualError

    @abstractmethod
    def _explicit_data_oos(self, x=None):
        pass

    def _implicit_data_oos(self, x=None, y=None):
        raise DualError

    def statistics_oos(self, x=None, y=None, data=None) -> Union[Tensor, (Tensor, Tensor)]:
        if self.explicit:
            if self._statistics_oos is None:
                if data is None:
                    data = self.parent.data_oos(x=x)
                if x is None:
                    statistic = self.statistics()
                else:
                    statistic = self._explicit_statistics_oos(x=x, data=data)
                self._statistics_oos = statistic
            return self._statistics_oos
        else:  # implicit
            if self._statistics_oos is None:
                if data is None:
                    data = self.parent.data_oos(x=x, y=y)
                if x is None:
                    stat_x = self.statistics
                else:
                    stat_x = self._implicit_statistics_oos(x=x, data=data)
                if equal(x,y):
                    stat_y = stat_x
                elif y is None:
                    stat_y = self.statistics
                else:
                    stat_y = self._implicit_statistics_oos(x=y, data=data)
                self._statistics_oos = [stat_x, stat_y]
            return self._statistics_oos[0], self._statistics_oos[1]

    def data_oos(self, x=None, y=None) -> Tensor:
        if self.explicit:
            return self._explicit_data_oos(x=x)
        else:
            return self._implicit_data_oos(x=x, y=y)

    def clean_oos(self):
        self._statistics_oos = None


class _MeanCentering(_Transform):
    def __init__(self, explicit: bool, default_path: bool = False):
        super(_MeanCentering, self).__init__(explicit=explicit, name="Mean centering", default_path=default_path)

    def _explicit_statistics(self, data):
        return torch.mean(data, dim=0)

    def _implicit_statistics(self, data, x=None):
        mean = torch.mean(data, dim=1, keepdim=True)
        mean_tot = torch.mean(mean)
        return mean, mean_tot

    def _explicit_data(self):
        data = self.parent.data
        return data - self.statistics(data)

    def _implicit_data(self):
        mat = self.parent.data
        mean, mean_tot = self.statistics(mat)
        return mat - mean - mean.T + mean_tot

    def _explicit_statistics_oos(self, x=None, data=None):
        return self.statistics()

    def _implicit_statistics_oos(self, x=None, data=None):
        mat_x = self.parent.data_oos(x=x)
        return torch.mean(mat_x, dim=1, keepdim=True)

    def _explicit_data_oos(self, x=None):
        return self.parent.data_oos(x=x) - self.statistics_oos(x=x)

    def _implicit_data_oos(self, x=None, y=None):
        mean_x, mean_y = self.statistics_oos(x=x, y=y)
        mean_tot = self.statistics()[1]
        return self.parent.data_oos(x=x, y=y) - mean_x \
               - mean_y.T \
               + mean_tot


class _UnitSphereNormalization(_Transform):
    def __init__(self, explicit: bool, default_path: bool = False):
        super(_UnitSphereNormalization, self).__init__(explicit=explicit, name="Unit Sphere Normalization",
                                                       default_path=default_path)

    def _explicit_statistics(self, data):
        return torch.norm(data, dim=1, keepdim=True)

    def _implicit_statistics(self, data, x=None):
        if data.nelement() == 0:
            return self._implicit_self(x)
        else:
            return torch.sqrt(torch.diag(data))[:, None]

    def _explicit_data(self):
        data = self.parent.data
        norm = self.statistics(data)
        return data / torch.clamp(norm, min=EPS)

    def _implicit_data(self):
        data = self.parent.data
        norm = self.statistics(data)
        return data / torch.clamp(norm * norm.T, min=EPS)

    def _explicit_statistics_oos(self, x=None, data=None):
        return torch.norm(data, dim=1, keepdim=True)

    def _implicit_statistics_oos(self, x=None, data=None) -> Tensor:
        if data.nelement() == 0:
            d = self._implicit_self(x)
        else:
            d = torch.diag(data)[:, None]
        return torch.sqrt(d)

    def _explicit_data_oos(self, x=None):
        vec = self.parent.data_oos(x)
        norm = self.statistics_oos(x=x, data=vec)
        return vec / torch.clamp(norm, min=EPS)

    def _implicit_data_oos(self, x=None, y=None):
        mat = self.parent.data_oos(x=x, y=y)
        # avoid computing the full matrix and use the _implicit_self when possible
        norm_x, norm_y = self.statistics_oos(x=x, y=y, data=torch.empty(0))
        return mat / torch.clamp(norm_x * norm_y.T, min=EPS)

    def _implicit_self(self, x=None) -> Tensor:
        if isinstance(self.parent, TransformTree):
            return self.parent._implicit_self(x)
        else:
            return torch.diag(self.data_oos(x, x))[:, None]


class _MinMaxNormalization(_Transform):
    def __init__(self, explicit: bool, default_path: bool = False):
        super(_MinMaxNormalization, self).__init__(explicit=explicit,
                                                   name="Min Max Normalization", default_path=default_path)

    def _explicit_statistics(self, data):
        max_vec = torch.max(data, dim=0)
        if type(self.parent) is _MinimumCentering:
            return max_vec  # new min is 0
        else:
            min_vec = torch.min(data, dim=0)
            return max_vec - min_vec

    def _explicit_data(self):
        data = self.parent.data
        norm = self.statistics(data)
        return data / torch.clamp(norm, min=EPS)

    def _explicit_statistics_oos(self, x=None, data=None):
        return self.statistics

    def _explicit_data_oos(self, x=None):
        return self.parent.data_oos(x=x) / torch.clamp(self.statistics_oos(x=x), min=EPS)


class _MinimumCentering(_Transform):
    def __init__(self, explicit: bool, default_path: bool = False):
        super(_MinimumCentering, self).__init__(explicit=explicit,
                                                name="Minimum Centering", default_path=default_path)

    def _explicit_statistics(self, data):
        return torch.min(data, dim=0)

    def _explicit_data(self):
        data = self.parent.data
        return data - self.statistics(data)

    def _explicit_statistics_oos(self, x=None, data=None):
        return self.statistics

    def _explicit_data_oos(self, x=None):
        return self.parent.data_oos(x=x) - self.statistics_oos(x=x)


class _UnitVarianceNormalization(_Transform):
    def __init__(self, explicit: bool, default_path: bool = False):
        super(_UnitVarianceNormalization, self).__init__(explicit=explicit,
                                                         name="Unit Variance Normalization", default_path=default_path)

    def _explicit_statistics(self, data):
        return torch.std(data, dim=0)

    def _explicit_data(self):
        data = self.parent.data
        norm = self.statistics(data)
        return data / torch.clamp(norm, min=EPS)

    def _explicit_statistics_oos(self, x=None, data=None):
        return self.statistics

    def _explicit_data_oos(self, x=None):
        return self.parent.data_oos(x=x) / torch.clamp(self.statistics_oos(x=x), min=EPS)


all_transforms = {"normalize": _UnitSphereNormalization,  # for legacy
                  "center": _MeanCentering,  # for legacy
                  "unit_sphere_normalization": _UnitSphereNormalization,
                  "mean_centering": _MeanCentering,
                  "minimum_centering": _MinimumCentering,
                  "unit_variance_normalization": _UnitVarianceNormalization,
                  "min_max_normalization": _MinMaxNormalization,
                  "standardize": [_MeanCentering, _UnitVarianceNormalization],
                  "min_max_rescaling": [_MinimumCentering, _MinMaxNormalization]}


class TransformTree(_Transform):
    @staticmethod
    def beautify_transforms(transforms) -> Union[None, List[_Transform]]:
        if transforms is None:
            return None
        else:
            transform_classes = []
            for transform_name in transforms:
                transform_classes.append(all_transforms.get(
                    transform_name, NameError(f"Unrecognized transform key {transform_name}.")))

            # remove same following elements
            previous_item = None
            idx = 0
            for current_item in transform_classes:
                if current_item == previous_item:
                    transforms.pop(idx)
                else:
                    previous_item = current_item
                    idx = idx + 1

            return transform_classes

    def __init__(self, explicit: bool, data, default_transforms=None, implicit_self=None, **kwargs):
        super(TransformTree, self).__init__(explicit=explicit, name='base', **kwargs)

        if default_transforms is None:
            default_transforms = []

        self._default_transforms = TransformTree.beautify_transforms(default_transforms)

        # create default tree
        node = self
        for transform in self._default_transforms:
            child = transform(explicit=self.explicit, default_path=True)
            node.add_child(child)
            node = child
        self._default_node = node
        node.default = True

        self._base = data
        self._data = None
        self._data_oos = None

        self._implicit_self_fun = implicit_self

    # TODO: rewrite __str__ to show full tree

    @property
    def default_transforms(self) -> List:
        return self._default_transforms

    def _get_data(self) -> Union[Tensor, Parameter]:
        if callable(self._base):
            return self._base()
        return self._base

    def _explicit_statistics(self, data):
        return None

    def _implicit_statistics(self, data, x=None):
        return None

    def _explicit_data(self):
        if callable(self._base):
            return self._base()
        return self._base

    def _implicit_data(self):
        if callable(self._base):
            return self._base()
        return self._base

    def _implicit_self(self, x=None) -> Union[Tensor]:
        if self._implicit_self_fun is not None:
            return self._implicit_self_fun(x)
        if x is None:
            return torch.diag(self._implicit_data())[:, None]
        else:
            return torch.diag(self._implicit_data_oos(x, x))[:, None]

    @property
    def default_data(self) -> Tensor:
        return self._default_node.data

    @property
    def default_statistics(self) -> Tensor:
        return self._default_node.statistics()

    def _explicit_statistics_oos(self, data, x=None):
        pass

    def _implicit_statistics_oos(self, data, x=None):
        pass

    def _explicit_data_oos(self, x=None):
        if callable(self._data_oos):
            return self._data_oos(x)
        return self._data_oos

    def _implicit_data_oos(self, x=None, y=None):
        assert callable(self._data_oos), "data_fun should be callable in the implicit case as it requires the " \
                                         "computation of data_fun(x,sample) for the centering and data_fun(x,x) for " \
                                         "the normalization."
        return self._data_oos(x, y)

    def apply_transforms(self, data, x=None, y=None, transforms: List[str] = None) -> Tensor:
        transforms = TransformTree.beautify_transforms(transforms)
        if transforms is None:
            transforms = self._default_transforms

        tree_path = [self]
        for transform in transforms:
            if transform not in tree_path[-1].children:
                child = transform(explicit=self.explicit)
                tree_path[-1].add_child(child)
            else:
                child = tree_path[-1].children[transform]
            tree_path.append(child)

        if x is None and y is None:
            sol = tree_path[-1].data
        else:
            self._data_oos = data
            sol = tree_path[-1].data_oos(x=x, y=y)
            for node in tree_path:
                node.clean_oos()
            self._data_oos = None  # to avoid blocking the destruction if necessary by the garbage collector
        return sol
