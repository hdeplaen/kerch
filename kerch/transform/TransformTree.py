# coding=utf-8
from typing import Union, List

import kerch
import torch
from torch import Tensor
from torch.nn import Parameter

from .Transform import Transform
from .all import (UnitSphereNormalization,
                  MinimumCentering,
                  MeanCentering,
                  UnitVarianceNormalization,
                  MinMaxNormalization)


@kerch.utils.extend_docstring(Transform)
class TransformTree(Transform):
    r"""
    Creates a tree of transform for efficient computing, with cache management.

    :param explicit: True is the transform are to be computed in the explicit formulation, False instead.
    :param sample: Default sample on which the statistics are to be computed.
    :param default_transform: Optional default list of transform., defaults to None.
    :param diag_fun: Optional function handle to directly compute the diagonal of the implicit formulation to increase
        computation speed.

    :type explicit: bool
    :type sample: Tensor or function handle
    :type default_transform: List[str]
    :type diag_fun: Function handle
    """

    _all_transform = {"normalize": UnitSphereNormalization,  # for legacy
                      "center": MeanCentering,  # for legacy
                      "sphere": UnitSphereNormalization,
                      "min": MinimumCentering,
                      "variance": UnitVarianceNormalization,
                      "standard": [MeanCentering, UnitVarianceNormalization],
                      "minmax": MinMaxNormalization,
                      "unit_sphere_normalization": UnitSphereNormalization,
                      "mean_centering": MeanCentering,
                      "minimum_centering": MinimumCentering,
                      "unit_variance_normalization": UnitVarianceNormalization,
                      "minmax_normalization": MinMaxNormalization,
                      "standardize": [MeanCentering, UnitVarianceNormalization],
                      "minmax_rescaling": [MinimumCentering, MinMaxNormalization]}

    @staticmethod
    def beautify_transform(transform: list[str]) -> Union[None, List[Transform]]:
        r"""
        Creates a list of _Transform classes and removes duplicates.

        :param transform: list of the different transform.
        :type transform: List[str]
        """
        if transform is None:
            return None
        else:
            transform_classes = []
            for tr in transform:
                try:
                    if issubclass(tr, Transform):
                        transform_classes.append(tr)
                except TypeError:
                    new_transform = TransformTree._all_transform.get(
                        tr, NameError(f"Unrecognized transform key {tr}."))
                    if isinstance(new_transform, Exception):
                        raise new_transform
                    elif isinstance(new_transform, List):
                        for ntr in new_transform:
                            transform_classes.append(ntr)
                    elif issubclass(new_transform, Transform):
                        transform_classes.append(new_transform)
                    else:
                        kerch._GLOBAL_LOGGER._logger.error("Error while creating TransformTree list of transform")

            # remove same following elements
            previous_item = None
            idx = 0
            for current_item in transform_classes:
                if current_item == previous_item:
                    transform_classes.pop(idx)
                else:
                    previous_item = current_item
                    idx += 1

            return transform_classes

    def __init__(self, explicit: bool, sample, default_transform=None, diag_fun=None, **kwargs):
        super(TransformTree, self).__init__(explicit=explicit, name='base', **kwargs)

        if default_transform is None:
            default_transform = []

        self._default_transforms = TransformTree.beautify_transform(default_transform)

        # create default tree
        node = self
        for transform in self._default_transforms:
            offspring = transform(explicit=self.explicit, default_path=True, cache_level=self.cache_level)
            node.add_offspring(offspring)
            node = offspring
        self._default_node = node
        node.default = True

        self._base = sample
        self._data_oos = None

        self._diag_fun = diag_fun

    def __str__(self):
        output = "Transforms: \n"
        if len(self._default_transforms) == 0:
            return output + "\t" + "None (default)"
        node = self._default_node
        while not isinstance(node, TransformTree):
            output += "\t" + str(node)
            node = node.parent
        return output

    @property
    def default_transforms(self) -> List:
        r"""
        Default list of transforms to be applied.
        """
        return self._default_transforms

    @property
    def final_transform(self) -> type(Transform):
        r"""
        Final transform to be applied, which is the last element of
        :py:attr:`~kerch.transform.TransformTree.default_transforms`.
        """
        try:
            return self._default_transforms[-1]
        except IndexError:
            return TransformTree

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

    def _implicit_diag(self, x=None) -> Union[Tensor]:
        if self._diag_fun is not None:
            return self._diag_fun(x)
        if x is None:
            return torch.diag(self._implicit_sample())[:, None]
        else:
            return torch.diag(self._implicit_oos(x, x))[:, None]

    @property
    def projected_sample(self) -> Tensor:
        r"""
        Sample after transform. Retrieved from cache if relevant.
        """
        return self._default_node.sample

    def _explicit_statistics_oos(self, oos, x=None):
        pass

    def _implicit_statistics_oos(self, oos, x=None):
        pass

    def _explicit_oos(self, x=None):
        if callable(self._base):
            return self._base(x)
        return x

    def _implicit_oos(self, x=None, y=None):
        if callable(self._base):
            return self._base(x, y)
        raise NotImplementedError

    def _revert_explicit(self, oos):
        return oos

    def _revert_implicit(self, oos):
        return oos

    def _get_tree(self, transform: List[str] = None) -> List[Transform]:
        transform = TransformTree.beautify_transform(transform)
        if transform is None:
            transform = self._default_transforms

        tree_path = [self]
        for tr_class in transform:
            current_tr = tree_path[-1]
            if tr_class in current_tr.offspring:
                offspring = current_tr.offspring[tr_class]
            else:
                offspring = tr_class(explicit=self.explicit, cache_level=self.cache_level)
                current_tr.add_offspring(offspring)
            tree_path.append(offspring)
        return tree_path

    def apply(self, oos=None, x=None, y=None, transform: List[str] = None) -> Tensor:
        r"""
        Applies the transform to the value to out-of-sample data. Either value is a function handle and you can use
        x (explicit) and x, y (explicit) to specify the data. Either directly give a Tensor.

        .. warning::
            If value is a Tensor, some transform may not work in implicit formulation. For example, the unit sphere
            normalization requires k(x,x) for all out-of-sample points. Some combinations may be even more intricate.

        :param x: Relevant if using a function handle for value.
        :param y: Relevant if using a function handle for value in implicit mode.
        :param transform: Transforms to be used. If none are to be used, i.e. getting the raw data back, please
            specify [], not None, which will return the default transform used for the sample., defaults to None,
            i.e., the default transform.

        :type x: Tensor
        :type y: Tensor
        :type transform: List[str]
        """
        tree_path = self._get_tree(transform)
        sol = tree_path[-1].oos(x=x, y=y)
        self._clean_cache()
        return sol

    def revert(self, value, transform: List[str] = None) -> Tensor:
        r"""
        Reverts the transform (runs the tree backwards) to the value to out-of-sample data.

        :param value: Out-of-sample data.
        :param transform: Transforms to be used. If none are to be used, i.e. getting the raw data back, please
            specify [], not None, which will return the default transform used for the sample., defaults to None,
            i.e., the default transform.

        :type value: Tensor
        :type transform: List[str]
        """

        tree_path = self._get_tree(transform)
        for transform in reversed(tree_path):
            value = transform._revert(value)
        return value
