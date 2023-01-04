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

import kerch
from kerch._logger import _Logger
from kerch.utils.tensor import equal
from kerch.utils.type import EPS
from kerch.utils.errors import BijectionError, ImplicitError


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
        self._sample = None
        self._statistics_sample = None
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
    def sample(self) -> Tensor:
        if self._sample is None:
            if self.explicit:
                data = self._explicit_sample()
            else:
                data = self._implicit_sample()

            if not self._lightweight or self.default:
                self._sample = data
            return data
        return self._sample

    def statistics_sample(self, sample=None) -> Tensor:
        if self._statistics_sample is None:
            if sample is None:
                sample = self.parent.sample
            if self.explicit:
                statistics = self._explicit_statistics(sample=sample)
            else:
                statistics = self._implicit_statistics(sample=sample)

            if not self._lightweight or self._default_path:
                self._statistics_sample = statistics
        return self._statistics_sample

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

    def statistics_oos(self, x=None, y=None, oos=None) -> Union[Tensor, (Tensor, Tensor)]:
        if self.explicit:
            if self._statistics_oos is None:
                if oos is None:
                    oos = self.parent.oos(x=x)
                if x is None:
                    statistic = self.statistics_sample()
                else:
                    statistic = self._explicit_statistics_oos(x=x, oos=oos)
                self._statistics_oos = statistic
            return self._statistics_oos
        else:  # implicit
            if self._statistics_oos is None:
                if oos is None:
                    oos = self.parent.oos(x=x, y=y)
                if x is None:
                    stat_x = self.statistics_sample
                else:
                    stat_x = self._implicit_statistics_oos(x=x, oos=oos)
                if equal(x,y):
                    stat_y = stat_x
                elif y is None:
                    stat_y = self.statistics_sample
                else:
                    stat_y = self._implicit_statistics_oos(x=y, oos=oos)
                self._statistics_oos = [stat_x, stat_y]
            return self._statistics_oos[0], self._statistics_oos[1]

    def oos(self, x=None, y=None) -> Tensor:
        if self.explicit:
            return self._explicit_oos(x=x)
        else:
            return self._implicit_oos(x=x, y=y)

    def clean_oos(self):
        self._statistics_oos = None

    def _revert(self, oos):
        if self.explicit:
            return self._revert_explicit(oos)
        else:
            return self._revert_implicit(oos)

    def _revert_explicit(self, oos):
        raise BijectionError

    def _revert_implicit(self, oos):
        raise ImplicitError


class _MeanCentering(_Transform):
    def __init__(self, explicit: bool, default_path: bool = False):
        super(_MeanCentering, self).__init__(explicit=explicit, name="Mean centering", default_path=default_path)

    def _explicit_statistics(self, sample):
        return torch.mean(sample, dim=0)

    def _implicit_statistics(self, sample, x=None):
        mean = torch.mean(sample, dim=1, keepdim=True)
        mean_tot = torch.mean(mean)
        return mean, mean_tot

    def _explicit_sample(self):
        sample = self.parent.sample
        return sample - self.statistics_sample(sample)

    def _implicit_sample(self):
        mat = self.parent.sample
        mean, mean_tot = self.statistics_sample(mat)
        return mat - mean - mean.T + mean_tot

    def _explicit_statistics_oos(self, x=None, oos=None):
        return self.statistics_sample()

    def _implicit_statistics_oos(self, x=None, oos=None):
        sample_x = self.parent.oos(x=x)
        return torch.mean(sample_x, dim=1, keepdim=True)

    def _explicit_oos(self, x=None):
        return self.parent.oos(x=x) - self.statistics_oos(x=x)

    def _implicit_oos(self, x=None, y=None):
        mean_x, mean_y = self.statistics_oos(x=x, y=y)
        mean_tot = self.statistics_sample()[1]
        return self.parent.oos(x=x, y=y) - mean_x \
               - mean_y.T \
               + mean_tot

    def _revert_explicit(self, oos):
        return oos + self.statistics_sample()


class _UnitSphereNormalization(_Transform):
    def __init__(self, explicit: bool, default_path: bool = False):
        super(_UnitSphereNormalization, self).__init__(explicit=explicit, name="Unit Sphere Normalization",
                                                       default_path=default_path)

    def _explicit_statistics(self, sample):
        return torch.norm(sample, dim=1, keepdim=True)

    def _implicit_statistics(self, sample, x=None):
        if sample.nelement() == 0:
            return self._implicit_self(x)
        else:
            return torch.sqrt(torch.diag(sample))[:, None]

    def _explicit_sample(self):
        sample = self.parent.sample
        norm = self.statistics_sample(sample)
        return sample / torch.clamp(norm, min=EPS)

    def _implicit_sample(self):
        sample = self.parent.sample
        norm = self.statistics_sample(sample)
        return sample / torch.clamp(norm * norm.T, min=EPS)

    def _explicit_statistics_oos(self, x=None, oos=None):
        return torch.norm(oos, dim=1, keepdim=True)

    def _implicit_statistics_oos(self, x=None, oos=None) -> Tensor:
        if oos.nelement() == 0:
            d = self._implicit_self(x)
        else:
            d = torch.diag(oos)[:, None]
        return torch.sqrt(d)

    def _explicit_oos(self, x=None):
        oos = self.parent.oos(x)
        norm = self.statistics_oos(x=x, oos=oos)
        return oos / torch.clamp(norm, min=EPS)

    def _implicit_oos(self, x=None, y=None):
        oos = self.parent.oos(x=x, y=y)
        # avoid computing the full matrix and use the _implicit_self when possible
        norm_x, norm_y = self.statistics_oos(x=x, y=y, oos=torch.empty(0))
        return oos / torch.clamp(norm_x * norm_y.T, min=EPS)

    def _implicit_self(self, x=None) -> Tensor:
        if isinstance(self.parent, TransformTree):
            return self.parent._implicit_self(x)[:, None]
        else:
            return torch.diag(self.oos(x, x))[:, None]


class _MinMaxNormalization(_Transform):
    def __init__(self, explicit: bool, default_path: bool = False):
        super(_MinMaxNormalization, self).__init__(explicit=explicit,
                                                   name="Min Max Normalization", default_path=default_path)

    def _explicit_statistics(self, sample):
        max_sample = torch.max(sample, dim=0).values
        if type(self.parent) is _MinimumCentering:
            return max_sample  # new min is 0
        else:
            min_sample = torch.min(sample, dim=0).values
            return max_sample - min_sample

    def _explicit_sample(self):
        sample = self.parent.sample
        norm = self.statistics_sample(sample)
        return sample / torch.clamp(norm, min=EPS)

    def _explicit_statistics_oos(self, x=None, oos=None):
        return self.statistics_sample()

    def _explicit_oos(self, x=None):
        return self.parent.oos(x=x) / torch.clamp(self.statistics_oos(x=x), min=EPS)

    def _revert_explicit(self, oos):
        return oos * torch.clamp(self.statistics_sample(), min=EPS)


class _MinimumCentering(_Transform):
    def __init__(self, explicit: bool, default_path: bool = False):
        super(_MinimumCentering, self).__init__(explicit=explicit,
                                                name="Minimum Centering", default_path=default_path)

    def _explicit_statistics(self, sample):
        return torch.min(sample, dim=0).values

    def _explicit_sample(self):
        sample = self.parent.sample
        return sample - self.statistics_sample(sample)

    def _explicit_statistics_oos(self, x=None, oos=None):
        return self.statistics_sample()

    def _explicit_oos(self, x=None):
        return self.parent.oos(x=x) - self.statistics_oos(x=x)

    def _revert_explicit(self, oos):
        return oos + self.statistics_sample()


class _UnitVarianceNormalization(_Transform):
    def __init__(self, explicit: bool, default_path: bool = False):
        super(_UnitVarianceNormalization, self).__init__(explicit=explicit,
                                                         name="Unit Variance Normalization", default_path=default_path)

    def _explicit_statistics(self, sample):
        return torch.std(sample, dim=0)

    def _explicit_sample(self):
        sample = self.parent.sample
        norm = self.statistics_sample(sample)
        return sample / torch.clamp(norm, min=EPS)

    def _explicit_statistics_oos(self, x=None, oos=None):
        return self.statistics_sample()

    def _explicit_oos(self, x=None):
        return self.parent.oos(x=x) / torch.clamp(self.statistics_oos(x=x), min=EPS)

    def _revert_explicit(self, sample):
        return sample * torch.clamp(self.statistics_sample(), min=EPS)


all_transforms = {"normalize": _UnitSphereNormalization,  # for legacy
                  "center": _MeanCentering,  # for legacy
                  "unit_sphere_normalization": _UnitSphereNormalization,
                  "mean_centering": _MeanCentering,
                  "minimum_centering": _MinimumCentering,
                  "unit_variance_normalization": _UnitVarianceNormalization,
                  "minmax_normalization": _MinMaxNormalization,
                  "standardize": [_MeanCentering, _UnitVarianceNormalization],
                  "minmax_rescaling": [_MinimumCentering, _MinMaxNormalization]}


class TransformTree(_Transform):
    @staticmethod
    def beautify_transforms(transforms) -> Union[None, List[_Transform]]:
        if transforms is None:
            return None
        else:
            transform_classes = []
            for transform_name in transforms:
                new_transform = all_transforms.get(
                    transform_name, NameError(f"Unrecognized transform key {transform_name}."))
                if isinstance(new_transform, Exception):
                    raise new_transform
                elif isinstance(new_transform, List):
                    for tr in new_transform:
                        transform_classes.append(tr)
                elif issubclass(new_transform, _Transform):
                    transform_classes.append(new_transform)
                else:
                    kerch._GLOBAL_LOGGER._log.error("Error while creating TransformTree list of transforms")


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

    def __init__(self, explicit: bool, sample, default_transforms=None, implicit_self=None, **kwargs):
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

        self._base = sample
        self._data = None
        self._data_oos = None

        self._implicit_self_fun = implicit_self

    def __repr__(self):
        output = "Transforms: \n"
        if len(self._default_transforms) == 0:
            return output + "\t" + "None (default)"
        for transform in self._default_transforms:
            output += "\t" + self.children[transform].__str__()
        return output

    @property
    def default_transforms(self) -> List:
        return self._default_transforms

    def _get_data(self) -> Union[Tensor, Parameter]:
        if callable(self._base):
            return self._base()
        return self._base

    def _explicit_statistics(self, sample):
        return None

    def _implicit_statistics(self, sample, x=None):
        return None

    def _explicit_sample(self):
        if callable(self._base):
            return self._base()
        return self._base

    def _implicit_sample(self):
        if callable(self._base):
            return self._base()
        return self._base

    def _implicit_self(self, x=None) -> Union[Tensor]:
        if self._implicit_self_fun is not None:
            return self._implicit_self_fun(x)
        if x is None:
            return torch.diag(self._implicit_sample())[:, None]
        else:
            return torch.diag(self._implicit_oos(x, x))[:, None]

    @property
    def default_sample(self) -> Tensor:
        return self._default_node.sample

    @property
    def default_statistics(self) -> Tensor:
        return self._default_node.statistics_sample()

    def _explicit_statistics_oos(self, oos, x=None):
        pass

    def _implicit_statistics_oos(self, oos, x=None):
        pass

    def _explicit_oos(self, x=None):
        if callable(self._data_oos):
            return self._data_oos(x)
        return self._data_oos

    def _implicit_oos(self, x=None, y=None):
        assert callable(self._data_oos), "data_fun should be callable in the implicit case as it requires the " \
                                         "computation of data_fun(x,sample) for the centering and data_fun(x,x) for " \
                                         "the normalization."
        return self._data_oos(x, y)

    def _revert_explicit(self, oos):
        return oos

    def _revert_implicit(self, oos):
        return oos

    def _create_tree(self, transforms: List[str] = None) -> List[_Transform]:
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
        return tree_path

    def apply(self, oos, x=None, y=None, transforms: List[str] = None) -> Tensor:
        tree_path = self._create_tree(transforms)
        if (not isinstance(oos, Tensor)) and x is None and y is None:
            return tree_path[-1].sample
        elif isinstance(oos, Tensor):
            x = 'oos1'
            y = 'oos2'

        self._data_oos = oos
        sol = tree_path[-1].oos(x=x, y=y)
        for node in tree_path:
            node.clean_oos()
        self._data_oos = None  # to avoid blocking the destruction if necessary by the garbage collector
        return sol

    def revert(self, oos, transforms: List[str] = None) -> Tensor:
        tree_path = self._create_tree(transforms)
        for transform in reversed(tree_path):
            oos = transform._revert(oos)
        return oos