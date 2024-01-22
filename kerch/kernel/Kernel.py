# coding=utf-8
"""
File containing the abstract kernel classes with transform.

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
from ._BaseKernel import _BaseKernel
from ..transform import TransformTree
from ..transform.all.Sphere import UnitSphereNormalization


@utils.extend_docstring(_BaseKernel)
class Kernel(_BaseKernel, metaclass=ABCMeta):
    r"""
    :param kernel_transform: A list composed of the elements `'normalize'` or `'center'`. For example a centered
        cosine kernel which is centered and normalized in order to get a covariance matrix for example can be obtained
        by invoking a linear kernel with `default_transform = ['normalize', 'center', 'normalize']` or just a cosine
        kernel with `default_transform = ['center', 'normalize']`. Redundancy is automatically handled., defaults
        to `[]`.
    :type kernel_transform: List[str]
    """

    @abstractmethod
    @utils.kwargs_decorator({
        "center": False,
        "normalize": False
    })
    def __init__(self, *args, **kwargs):
        super(Kernel, self).__init__(*args, **kwargs)

        kernel_transform = kwargs.pop('kernel_transform', [])

        # LEGACY SUPPORT
        if kwargs["center"]:
            self._log.warning("Argument center kept for legacy and will be removed in a later version. Please use the "
                              "more versatile kernel_transform parameter instead.")
            kernel_transform.append("mean_centering")
        if kwargs["normalize"]:
            self._log.warning("Argument normalize kept for legacy and will be removed in a later version. Please use "
                              "the more versatile kernel_transform parameter instead.")
            kernel_transform.append("unit_sphere_normalization")

        self._default_kernel_transform = self._simplify_transform(kernel_transform)

    @property
    def kernel_transform(self) -> TransformTree:
        r"""
        Default transform performed on the kernel
        """
        if self.explicit:
            return self._kernel_explicit_transform
        else:
            return self._kernel_implicit_transform

    @property
    def hparams(self) -> dict:
        return {'Default kernel transforms': self._default_kernel_transform, **super(Kernel, self).hparams}

    @property
    def _naturally_centered(self) -> bool:
        return False

    @property
    def _naturally_normalized(self) -> bool:
        return False

    @property
    def _required_transform(self) -> Union[List, None]:
        return None

    def _simplify_transform(self, transform=None) -> List:
        if transform is None:
            transform = []

        # add requirements
        if self._required_transform is not None:
            transform.append(self._required_transform)

        # remove same following elements
        transform = TransformTree.beautify_transform(transform)

        # remove unnecessary operation if kernel does it by default
        try:
            if self._naturally_normalized and transform[0] == UnitSphereNormalization:
                transform.pop(0)
        except IndexError:
            pass

        return transform

    def _get_transform(self, transform=None) -> List:
        if transform is None:
            return self._default_kernel_transform
        return self._simplify_transform(transform)

    @property
    def centered(self) -> bool:
        r"""
        Returns if the kernel is centered in the feature maps.
        """
        if not self._naturally_centered:
            try:
                return self._default_kernel_transform[0] == 'center'
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
                return self._default_kernel_transform[0] == 'normalize'
            except IndexError:
                return False
        return True

    @property
    def _kernel_explicit_transform(self) -> TransformTree:
        def fun():
            return TransformTree(explicit=True,
                                  sample=self._explicit_with_none,
                                  default_transform=self._default_kernel_transform,
                                  cache_level=self._cache_level)
        return self._get("kernel_explicit_transform", level_key="kernel_explicit_transform", fun=fun)

    @property
    def _kernel_implicit_transform(self) -> TransformTree:
        def fun():
            return TransformTree(explicit=False,
                                  sample=self._implicit_with_none,
                                  default_transform=self._default_kernel_transform,
                                  cache_level=self._cache_level,
                                  implicit_self=self._implicit_self)
        return self._get("kernel_implicit_transform", level_key="kernel_implicit_transform", fun=fun)

    def _phi(self):
        r"""
        Returns the explicit feature map with default centering and normalization. If already computed, it is
        recovered from the cache.
        """
        def fun():
            self._check_sample()
            return self._kernel_explicit_transform.projected_sample
        return self._get("phi", level_key="sample_phi", fun=fun)

    def _C(self) -> Tensor:
        r"""
        Returns the covariance matrix with default centering and normalization. If already computed, it is recovered
        from the cache.
                """
        def fun():
            scale = 1 / self.num_idx
            phi = self._phi()
            return scale * phi.T @ phi
        return self._get("C", level_key="sample_C", fun=fun)

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
            self._check_sample()
            if explicit is None: explicit = self.explicit
            if explicit:
                phi = self._phi()
                return phi @ phi.T
            else:
                return self._kernel_implicit_transform.projected_sample
        return self._get("K", level_key="sample_K", fun=lambda: fun(explicit), overwrite=overwrite)

    # ACCESSIBLE METHODS
    def phi(self, x=None, transform=None) -> Tensor:
        r"""
        Returns the explicit feature map :math:`\phi(\cdot)` of the specified points.

        :param x: The datapoints serving as input of the explicit feature map. If `None`, the sample will be used.,
            defaults to `None`
        :type x: Tensor(,dim_input), optional
        :raises: ExplicitError
        """
        x = utils.castf(x)
        transform = self._get_transform(transform)
        return self._kernel_explicit_transform.apply(oos=self._explicit_with_none, x=self.project_input(x), transform=transform)

    def k(self, x=None, y=None, explicit=None, transform=None) -> Tensor:
        r"""
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
        transform = self._get_transform(transform)
        if explicit:
            phi_x = self._kernel_explicit_transform.apply(x=self.project_input(x),
                                                          transform=transform)
            if utils.equal(x, y):
                phi_y = phi_x
            else:
                phi_y = self._kernel_explicit_transform.apply(y=self.project_input(y),
                                                              transform=transform)
            return phi_x @ phi_y.T
        else: # implicit
            if utils.equal(x, y):
                x = self.project_input(x)
                return self._kernel_implicit_transform.apply(x=x,
                                                             y=x,
                                                             transform=transform)
            else:
                return self._kernel_implicit_transform.apply(x=self.project_input(x),
                                                             y=self.project_input(y),
                                                             transform=transform)

    def c(self, x=None, transform=None) -> Tensor:
        r"""
        Out-of-sample explicit matrix.

        .. math::
            C = \frac1M\sum_{i}^{M} \phi(x_i)\phi(x_i)^\top.

        :param x: Out-of-sample points (first dimension). If `None`, the default sample will be used., defaults to `None`
        :type x: Tensor(N,dim_input), optional

        :return: Explicit matrix
        :rtype: Tensor(dim_feature,dim_feature)
        """

        transform = self._get_transform(transform)
        phi = self._kernel_explicit_transform.apply(oos=self._explicit_with_none, x=self.project_input(x), transform=transform)
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

        :param x: Out-of-sample points (first dimension). If `None`, the default sample will be used.
            Defaults to `None`.
        :type x: Tensor[N, dim_input], optional

        :return: Covariance matrix
        :rtype: Tensor[dim_feature, dim_feature]
        """
        transform = self._default_kernel_transform.copy()
        try:
            if not (self._default_kernel_transform[-1] == 'center'):
                transform.append('center')
        except IndexError:
            transform.append('center')

        transform = self._simplify_transform(transform)
        return self.c(x=x, transform=transform)

    def corr(self, x=None) -> Tensor:
        """
        Returns the correlation matrix fo the provided input.

        :param x: Out-of-sample points (first dimension). If `None`, the default sample will be used., defaults to `None`
        :type x: Tensor[N, dim_input], optional

        :return: Correlation matrix
        :rtype: Tensor[dim_feature, dim_feature]
        """
        cov = self.cov(x=x)
        var = torch.sqrt(torch.diag(cov))[:, None]
        return cov / (var * var.T)
