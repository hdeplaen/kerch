from typing import Union, List

import kerch
import torch
from torch import Tensor
from torch.nn import Parameter
from ._Projection import _Projection

from ._MinimumCentering import _MinimumCentering
from ._MeanCentering import _MeanCentering
from ._Sphere import _UnitSphereNormalization
from ._Variance import _UnitVarianceNormalization
from ._MinMaxNormalization import _MinMaxNormalization


@kerch.utils.extend_docstring(_Projection)
class ProjectionTree(_Projection):
    r"""
    Creates a tree of projection for efficient computing, with cache management.

    :param explicit: True is the projection are to be computed in the explicit formulation, False instead.
    :param sample: Default sample on which the statistics are to be computed.
    :param default_projections: Optional default list of projection., defaults to None.
    :param diag_fun: Optional function handle to directly compute the diagonal of the implicit formulation to increase
        computation speed.

    :type explicit: bool
    :type sample: Tensor or function handle
    :type default_projections: List[str]
    :type diag_fun: Function handle
    """

    all_projections = {"normalize": _UnitSphereNormalization,  # for legacy
                       "center": _MeanCentering,  # for legacy
                       "sphere": _UnitSphereNormalization,
                       "min": _MinimumCentering,
                       "variance": _UnitVarianceNormalization,
                       "standard": [_MeanCentering, _UnitVarianceNormalization],
                       "minmax": _MinMaxNormalization,
                       "unit_sphere_normalization": _UnitSphereNormalization,
                       "mean_centering": _MeanCentering,
                       "minimum_centering": _MinimumCentering,
                       "unit_variance_normalization": _UnitVarianceNormalization,
                       "minmax_normalization": _MinMaxNormalization,
                       "standardize": [_MeanCentering, _UnitVarianceNormalization],
                       "minmax_rescaling": [_MinimumCentering, _MinMaxNormalization]}

    @staticmethod
    def beautify_projections(projections) -> Union[None, List[_Projection]]:
        r"""
        Creates a list of _Projection classes and removes duplicates.

        :param projections: list of the different projection.
        :type projections: List[str]
        """
        if projections is None:
            return None
        else:
            projection_classes = []
            for projection in projections:
                try:
                    if issubclass(projection, _Projection):
                        projection_classes.append(projection)
                except TypeError:
                    new_projection = ProjectionTree.all_projections.get(
                        projection, NameError(f"Unrecognized projection key {projection}."))
                    if isinstance(new_projection, Exception):
                        raise new_projection
                    elif isinstance(new_projection, List):
                        for tr in new_projection:
                            projection_classes.append(tr)
                    elif issubclass(new_projection, _Projection):
                        projection_classes.append(new_projection)
                    else:
                        kerch._GLOBAL_LOGGER._log.error("Error while creating ProjectionTree list of projection")

            # remove same following elements
            previous_item = None
            idx = 0
            for current_item in projection_classes:
                if current_item == previous_item:
                    projections.pop(idx)
                else:
                    previous_item = current_item
                    idx = idx + 1

            return projection_classes

    def __init__(self, explicit: bool, sample, default_projections=None, diag_fun=None, **kwargs):
        super(ProjectionTree, self).__init__(explicit=explicit, name='base', **kwargs)

        if default_projections is None:
            default_projections = []

        self._default_projections = ProjectionTree.beautify_projections(default_projections)

        # create default tree
        node = self
        for projection in self._default_projections:
            offspring = projection(explicit=self.explicit, default_path=True)
            node.add_offspring(offspring)
            node = offspring
        self._default_node = node
        node.default = True

        self._base = sample
        self._data = None
        self._data_oos = None

        self._diag_fun = diag_fun

    def __str__(self):
        output = "Projections: \n"
        if len(self._default_projections) == 0:
            return output + "\t" + "None (default)"
        node = self._default_node
        while not isinstance(node, ProjectionTree):
            output += "\t" + str(node)
            node = node.parent
        return output

    @property
    def default_projections(self) -> List:
        r"""
        Default list of projection to be applied.
        """
        return self._default_projections

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
        Sample after projection. Retrieved from cache if relevant.
        """
        return self._default_node.sample

    def _explicit_statistics_oos(self, oos, x=None):
        pass

    def _implicit_statistics_oos(self, oos, x=None):
        pass

    def _explicit_oos(self, x=None):
        if callable(self._data_oos):
            return self._data_oos(x)
        return self._data_oos

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

    def _get_tree(self, projections: List[str] = None) -> List[_Projection]:
        projections = ProjectionTree.beautify_projections(projections)
        if projections is None:
            projections = self._default_projections

        tree_path = [self]
        for projection in projections:
            if projection not in tree_path[-1].offspring:
                offspring = projection(explicit=self.explicit)
                tree_path[-1].add_offspring(offspring)
            else:
                offspring = tree_path[-1].offspring[projection]
            tree_path.append(offspring)
        return tree_path

    def apply(self, oos=None, x=None, y=None, projections: List[str] = None) -> Tensor:
        r"""
        Applies the projection to the value to out-of-sample data. Either value is a function handle and you can use
        x (explicit) and x, y (explicit) to specify the data. Either directly give a Tensor.

        .. warning::
            If value is a Tensor, some projection may not work in implicit formulation. For example, the unit sphere
            normalization requires k(x,x) for all out-of-sample points. Some combinations may be even more intricate.

        :param oos: Out-of-sample data, defaults to the function handle used for the sample.
        :param x: Relevant if using a function handle for value.
        :param y: Relevant if using a function handle for value in implicit mode.
        :param projections: Projections to be used. If none are to be used, i.e. getting the raw data back, please
            specify [], not None, which will return the default projection used for the sample., defaults to None,
            i.e., the default projection.

        :type oos: function handle or Tensor
        :type x: Tensor
        :type y: Tensor
        :type projections: List[str]
        """

        if oos is None:
            if callable(self._base):
                oos = self._base
            else:
                self._log.error("No out-of-sample provided and the default one is not a function handle.")

        tree_path = self._get_tree(projections)
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

    def revert(self, value, projections: List[str] = None) -> Tensor:
        r"""
        Reverts the projection (runs the tree backwards) to the value to out-of-sample data.

        :param value: Out-of-sample data.
        :param projections: Projections to be used. If none are to be used, i.e. getting the raw data back, please
            specify [], not None, which will return the default projection used for the sample., defaults to None,
            i.e., the default projection.

        :type value: Tensor
        :type projections: List[str]
        """

        tree_path = self._get_tree(projections)
        for projection in reversed(tree_path):
            value = projection._revert(value)
        return value
