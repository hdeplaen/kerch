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

    all_transform = {"normalize": UnitSphereNormalization,  # for legacy
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
    def beautify_transform(transform) -> Union[None, List[Transform]]:
        r"""
        Creates a list of _Transform classes and removes duplicates.

        :param transform: list of the different transform.
        :type transform: List[str]
        """
        if transform is None:
            return None
        else:
            transform_classes = []
            for transform in transform:
                try:
                    if issubclass(transform, Transform):
                        transform_classes.append(transform)
                except TypeError:
                    new_transform = TransformTree.all_transform.get(
                        transform, NameError(f"Unrecognized transform key {transform}."))
                    if isinstance(new_transform, Exception):
                        raise new_transform
                    elif isinstance(new_transform, List):
                        for tr in new_transform:
                            transform_classes.append(tr)
                    elif issubclass(new_transform, Transform):
                        transform_classes.append(new_transform)
                    else:
                        kerch._GLOBAL_LOGGER._log.error("Error while creating TransformTree list of transform")

            # remove same following elements
            previous_item = None
            idx = 0
            for current_item in transform_classes:
                if current_item == previous_item:
                    transform.pop(idx)
                else:
                    previous_item = current_item
                    idx = idx + 1

            return transform_classes

    def __init__(self, explicit: bool, sample, default_transform=None, diag_fun=None, **kwargs):
        super(TransformTree, self).__init__(explicit=explicit, name='base', **kwargs)

        if default_transform is None:
            default_transform = []

        self._default_transform = TransformTree.beautify_transform(default_transform)

        # create default tree
        node = self
        for transform in self._default_transform:
            offspring = transform(explicit=self.explicit, default_path=True)
            node.add_offspring(offspring)
            node = offspring
        self._default_node = node
        node.default = True

        self._base = sample
        self._data = None
        self._data_oos = None

        self._diag_fun = diag_fun

    def __str__(self):
        output = "Transforms: \n"
        if len(self._default_transform) == 0:
            return output + "\t" + "None (default)"
        node = self._default_node
        while not isinstance(node, TransformTree):
            output += "\t" + str(node)
            node = node.parent
        return output

    @property
    def default_transform(self) -> List:
        r"""
        Default list of transform to be applied.
        """
        return self._default_transform

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
        if x == 'oos1':
            return self._data_oos
        else:
            assert callable(self._data_oos), 'data_oos should be callable when providing an x which is not None.'
            return self._data_oos(x)

    def _implicit_oos(self, x=None, y=None):
        if x == 'oos1' and y == 'oos2':
            return self._data_oos
        else:
            assert callable(self._data_oos), "data_oos should be callable in the implicit case as it requires the " \
                                             "computation of data_fun(x,sample) for the centering and data_fun(x,x) for " \
                                             "the normalization."
            return self._data_oos(x, y)

    def _revert_explicit(self, oos):
        return oos

    def _revert_implicit(self, oos):
        return oos

    def _get_tree(self, transform: List[str] = None) -> List[Transform]:
        transform = TransformTree.beautify_transform(transform)
        if transform is None:
            transform = self._default_transform

        tree_path = [self]
        for transform in transform:
            if transform not in tree_path[-1].offspring:
                offspring = transform(explicit=self.explicit)
                tree_path[-1].add_offspring(offspring)
            else:
                offspring = tree_path[-1].offspring[transform]
            tree_path.append(offspring)
        return tree_path

    def apply(self, oos=None, x=None, y=None, transform: List[str] = None) -> Tensor:
        r"""
        Applies the transform to the value to out-of-sample data. Either value is a function handle and you can use
        x (explicit) and x, y (explicit) to specify the data. Either directly give a Tensor.

        .. warning::
            If value is a Tensor, some transform may not work in implicit formulation. For example, the unit sphere
            normalization requires k(x,x) for all out-of-sample points. Some combinations may be even more intricate.

        :param oos: Out-of-sample data, defaults to the function handle used for the sample.
        :param x: Relevant if using a function handle for value.
        :param y: Relevant if using a function handle for value in implicit mode.
        :param transform: Transforms to be used. If none are to be used, i.e. getting the raw data back, please
            specify [], not None, which will return the default transform used for the sample., defaults to None,
            i.e., the default transform.

        :type oos: function handle or Tensor
        :type x: Tensor
        :type y: Tensor
        :type transform: List[str]
        """

        if oos is None:
            if callable(self._base):
                oos = self._base
            else:
                self._log.error("No out-of-sample provided and the default one is not a function handle.")

        tree_path = self._get_tree(transform)
        if (not isinstance(oos, Tensor)) and x is None and y is None:
            return tree_path[-1].sample
        elif isinstance(oos, Tensor):
            x = 'oos1'
            y = 'oos2'

        self._data_oos = oos
        sol = tree_path[-1].oos(x=x, y=y)
        for node in tree_path:
            node._clean_cache()
        self._data_oos = None  # to avoid blocking the destruction if necessary by the garbage collector
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
