"""
File containing the abstract kernel classes.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

from abc import ABCMeta, abstractmethod
from typing import List, Union

import torch
from torch import Tensor

from .. import utils
from ._Base import _Base
from ..projection import ProjectionTree
from ..projection._Sphere import _UnitSphereNormalization


@utils.extend_docstring(_Base)
class _Projected(_Base, metaclass=ABCMeta):
    r"""
    :param kernel_projections: A list composed of the elements `'normalize'` or `'center'`. For example a centered
        cosine kernel which is centered and normalized in order to get a covariance matrix for example can be obtained
        by invoking a linear kernel with `default_projections = ['normalize', 'center', 'normalize']` or just a cosine
        kernel with `default_projections = ['center', 'normalize']`. Redundancy is automatically handled., defaults
        to `[]`.
    :type kernel_projections: List[str]
    """

    @abstractmethod
    @utils.kwargs_decorator({
        "kernel_projections": [],
        "center": False,
        "normalize": False
    })
    def __init__(self, **kwargs):
        super(_Projected, self).__init__(**kwargs)

        projections = kwargs["kernel_projections"]

        # LEGACY SUPPORT
        if kwargs["center"]:
            self._log.warning("Argument center kept for legacy and will be removed in a later version. Please use the "
                              "more versatile kernel_projections parameter instead.")
            projections.append("mean_centering")
        if kwargs["normalize"]:
            self._log.warning("Argument normalize kept for legacy and will be removed in a later version. Please use "
                              "the more versatile kernel_projections parameter instead.")
            projections.append("unit_sphere_normalization")

        self._default_kernel_projections = self._simplify_projections(projections)

    @property
    def kernel_projections(self) -> ProjectionTree:
        r"""
        Default projection performed on the kernel
        """
        if self.explicit:
            return self._explicit_projection
        else:
            return self._implicit_projection

    @property
    def _naturally_centered(self) -> bool:
        return False

    @property
    def _naturally_normalized(self) -> bool:
        return False

    @property
    def _required_projections(self) -> Union[List, None]:
        return None

    def _simplify_projections(self, projections=None) -> List:
        if projections is None:
            projections = []

        # add requirements
        if self._required_projections is not None:
            projections.append(self._required_projections)

        # remove same following elements
        projections = ProjectionTree.beautify_projections(projections)

        # remove unnecessary operation if kernel does it by default
        try:
            if self._naturally_normalized and projections[0] == _UnitSphereNormalization:
                projections.pop(0)
        except IndexError:
            pass

        return projections

    def _get_projections(self, projections=None) -> List:
        if projections is None:
            return self._default_kernel_projections
        return self._simplify_projections(projections)

    @property
    def centered(self) -> bool:
        r"""
        Returns if the kernel is centered in the feature maps.
        """
        if not self._naturally_centered:
            try:
                return self._default_kernel_projections[0] == 'center'
            except IndexError:
                return False
        return True

    @property
    def normalized(self) -> bool:
        r"""
        Returns if the kernel is normalized in the feature maps.
        """
        if not self._naturally_normalized:
            try:
                return self._default_kernel_projections[0] == 'normalize'
            except IndexError:
                return False
        return True

    @property
    def _explicit_projection(self) -> ProjectionTree:
        def fun():
            return ProjectionTree(explicit=True,
                                  sample=self._explicit_with_none,
                                  default_projections=self._default_kernel_projections,
                                  cache_level=self._cache_level)
        return self._get("explicit_projection", "oblivious", fun)

    @property
    def _implicit_projection(self) -> ProjectionTree:
        def fun():
            return ProjectionTree(explicit=False,
                                  sample=self._implicit_with_none,
                                  default_projections=self._default_kernel_projections,
                                  cache_level=self._cache_level,
                                  implicit_self=self._implicit_self)
        return self._get("implicit_projection", "oblivious", fun)

    def _phi(self):
        r"""
        Returns the explicit feature map with default centering and normalization. If already computed, it is
        recovered from the cache.
        """
        def fun():
            return self._explicit_projection.projected_sample
        return self._get("phi", "lightweight", fun)

    def _C(self) -> Tensor:
        r"""
        Returns the covariance matrix with default centering and normalization. If already computed, it is recovered
        from the cache.
                """
        if "C" not in self._cache:
            scale = 1 / self.num_idx
            phi = self._phi()
            self._cache["C"] = scale * phi.T @ phi
        return self._cache["C"]

    def _K(self, explicit=None, overwrite: bool = False) -> Tensor:
        r"""
        Returns the kernel matrix with default centering and normalization. If already computed, it is recovered from
        the cache.

        :param explicit: Specifies whether the explicit or implicit formulation has to be used. Always uses
            the explicit if available.
        :param overwrite: By default, the kernel matrix is recovered from cache if already computed. Force overwrites
            this if True., defaults to False
        """
        def fun(explicit):
            if explicit is None: explicit = self.explicit
            if explicit:
                phi = self._phi()
                return phi @ phi.T
            else:
                return self._implicit_projection.projected_sample
        return self._get("K", "lightweight", lambda: fun(explicit), force=False, overwrite=overwrite)

    # ACCESSIBLE METHODS
    def phi(self, x=None, projections=None) -> Tensor:
        r"""
        Returns the explicit feature map :math:`\phi(\cdot)` of the specified points.

        :param x: The datapoints serving as input of the explicit feature map. If `None`, the sample will be used.,
            defaults to `None`
        :type x: Tensor(,dim_input), optional
        :raises: ExplicitError
        """
        x = utils.castf(x)
        projections = self._get_projections(projections)
        return self._explicit_projection.apply(oos=self._explicit_with_none, x=self.project_sample(x), projections=projections)

    def k(self, x=None, y=None, explicit=None, projections=None) -> Tensor:
        """
        Returns a kernel matrix, either of the sample, either out-of-sample, either fully out-of-sample.

        .. math::
            K = [k(x_i,y_j)]_{i,j=1}^{N,M},

        with :math:`\{x_i\}_{i=1}^N` the out-of-sample points (`x`) and :math:`\{y_i\}_{j=1}^N` the sample points
        (`y`).

        .. note::
            In the case of centered kernels, this computation is more expensive as it requires to _center according to
            the sample data, which implies computing a statistic on the out-of-sample kernel matrix and thus
            also computing it.

        :param x: Out-of-sample points (first dimension). If `None`, the default sample will be used., defaults to `None`
        :param y: Out-of-sample points (second dimension). If `None`, the default sample will be used., defaults to `None`

        :type x: Tensor(N,dim_input), optional
        :type y: Tensor(M,dim_input), optional

        :return: Kernel matrix
        :rtype: Tensor(N,M)

        :raises: ExplicitError
        """
        # if x is None and y is None:
        #     return self.K
        # in order to get the values in the correct format (e.g. coming from numpy)
        if explicit is None:
            explicit = self.explicit

        x = utils.castf(x)
        y = utils.castf(y)
        projections = self._get_projections(projections)
        if explicit:
            phi_x = self._explicit_projection.apply(x=self.project_sample(x),
                                                    projections=projections)
            if utils.equal(x, y):
                phi_y = phi_x
            else:
                phi_y = self._explicit_projection.apply(y=self.project_sample(y),
                                                        projections=projections)
            return phi_x @ phi_y.T
        else: # implicit
            if utils.equal(x, y):
                x = self.project_sample(x)
                return self._implicit_projection.apply(x=x,
                                                       y=x,
                                                       projections=projections)
            else:
                return self._implicit_projection.apply(x=self.project_sample(x),
                                                       y=self.project_sample(y),
                                                       projections=projections)

    def c(self, x=None, projections=None) -> Tensor:
        r"""
        Out-of-sample explicit matrix.

        .. math::
            C = \frac1M\sum_{i}^{M} \phi(x_i)\phi(x_i)^\top.

        :param x: Out-of-sample points (first dimension). If `None`, the default sample will be used., defaults to `None`
        :type x: Tensor(N,dim_input), optional

        :return: Explicit matrix
        :rtype: Tensor(dim_feature,dim_feature)
        """

        projections = self._get_projections(projections)
        phi = self._explicit_projection.apply(oos=self._explicit_with_none, x=self.project_sample(x), projections=projections)
        scale = 1 / x.shape[0]
        return scale * phi.T @ phi

    def forward(self, x, representation="dual") -> Tensor:
        """
        Passes datapoints through the kernel.

        :param x: Datapoints to be passed through the kernel.
        :param representation: Chosen representation. If `dual`, an out-of-sample kernel matrix is returned. If
            `primal` is specified, it returns the explicit feature map., defaults to `dual`

        :type x: Tensor(,dim_input)
        :type representation: str, optional

        :return: Out-of-sample kernel matrix or explicit feature map depending on `representation`.

        :raises: RepresentationError
        """

        def primal(x):
            return self.phi(x)

        def dual(x):
            return self.k(x)

        switcher = {"primal": primal,
                    "dual": dual}

        fun = switcher.get(representation, utils.RepresentationError)
        return fun(x)

    def cov(self, x=None) -> Tensor:
        r"""
        Returns the covariance matrix fo the provided input.

        :param x: Out-of-sample points (first dimension). If `None`, the default sample will be used., defaults to `None`
        :type x: Tensor(N,dim_input), optional

        :return: Covariance matrix
        :rtype: Tensor(dim_feature, dim_feature)
        """
        projections = self._default_kernel_projections
        try:
            if not (self._default_kernel_projections[-1] == 'center'):
                projections.append('center')
        except IndexError:
            projections.append('center')

        projections = self._simplify_projections(projections)
        return self.c(x=x, projections=projections)

    def corr(self, x=None) -> Tensor:
        """
        Returns the correlation matrix fo the provided input.

        :param x: Out-of-sample points (first dimension). If `None`, the default sample will be used., defaults to `None`
        :type x: Tensor(N,dim_input), optional

        :return: Correlation matrix
        :rtype: Tensor(dim_feature, dim_feature)
        """
        cov = self.cov(x=x)
        var = torch.sqrt(torch.diag(cov))[:, None]
        return cov / (var * var.T)
