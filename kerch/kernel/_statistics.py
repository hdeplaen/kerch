"""
File containing the abstract kernel classes.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

from abc import ABCMeta, abstractmethod
from typing import Union, List

from torch import Tensor
import torch

from .. import utils
from ._Base import _Base
from ._transforms import TransformTree


@utils.extend_docstring(_Base)
class _Statistics(_Base, metaclass=ABCMeta):
    r"""
    :param center: `True` if any implicit feature or kernel is must be centered, `False` otherwise. The _center
        is always performed relative to a statistic on the sample., defaults to `False`
    :param normalize: `True` if any implicit feature or kernel is must be normalized, `False` otherwise. The _center
        is always performed relative to a statistic on the sample., defaults to `False`
    :param ligthweight: During the computation of the statistics for centering and normalization, intermediate values
        are computed. It `True`, the model only keeps the necessary final statistics for the centering and
        normalization specified during construction. All other values are either never computed either discarded if
        required at some point. If asking for out-of-sample without the default centering and construction values, the
        statistics would then be computed and immediately discarded. Thye would have to be computed over and over again
        if repetively required. However, provided the same values are used as the one specified in the construction,
        this is the most efficient, not computing or keeping anything unnecessary. This parameter controls a
        time-memory trade-off., defaults to `True`
    :type center: bool, optional
    :type normalize: bool, optional
    :type ligthweight: bool, optional
    """

    @abstractmethod
    @utils.kwargs_decorator({
        "lightweight": True,
    })
    def __init__(self, **kwargs):
        super(_Statistics, self).__init__(**kwargs)
        self._lightweight = kwargs["lightweight"]

    def _get_transforms(self, transforms=None) -> List:
        return []

    @property
    def _explicit_statistics(self) -> TransformTree:
        if "explicit" not in self._cache:
            self._cache["explicit"] = TransformTree(explicit=True,
                                                    data=self._explicit,
                                                    default_transforms=self._default_transforms,
                                                    lighweight=self._lightweight)
        return self._cache["explicit"]

    @property
    def _implicit_statistics(self) -> TransformTree:
        if "implicit" not in self._cache:
            self._cache["implicit"] = TransformTree(explicit=False,
                                                    data=self._implicit,
                                                    default_transforms=self._default_transforms,
                                                    lighweight=self._lightweight)
        return self._cache["implicit"]

    def _phi(self):
        r"""
        Returns the explicit feature map with default centering and normalization. If already computed, it is
        recovered from the cache.
        """
        if "phi" not in self._cache:
            self._cache["phi"] = self._explicit_statistics.default_data
        return self._cache["phi"]

    def _C(self) -> Tensor:
        r"""
        Returns the covariance matrix with default centering and normalization. If already computed, it is recovered
        from the cache.
                """
        if "C" not in self._cache:
            phi = self._phi()
            self._cache["C"] = phi.T @ phi
        return self._cache["C"]

    def _K(self, explicit=None, force: bool = False) -> Tensor:
        r"""
        Returns the kernel matrix with default centering and normalization. If already computed, it is recovered from
        the cache.

        :param explicit: Specifies whether the explicit or implicit formulation has to be used. Always uses the
            the explicit if available.
        :param force: By default, the kernel matrix is recovered from cache if already computed. Force overwrites
            this if True., defaults to False
        """
        if "K" not in self._cache or force:
            if explicit is None: explicit = self.explicit
            if explicit:
                phi = self._phi()
                self._cache["K"] = phi @ phi.T
            else:
                self._cache["K"] = self._explicit_statistics.default_data
        return self._cache["K"]

    # ACCESSIBLE METHODS
    def phi(self, x=None, transforms = None) -> Tensor:
        r"""
        Returns the explicit feature map :math:`\phi(\cdot)` of the specified points.

        :param x: The datapoints serving as input of the explicit feature map. If `None`, the sample will be used.,
            defaults to `None`
        :type x: Tensor(,dim_input), optional
        :raises: PrimalError
        """
        x = utils.castf(x)
        transforms = self._get_transforms(transforms)
        return self._explicit_statistics.apply_transforms(data_fun=self._explicit, x=x, transforms=transforms)

    def k(self, x=None, y=None, explicit=None, transforms=None) -> Tensor:
        """
        Returns a kernel matrix, either of the sample, either out-of-sample, either fully out-of-sample.

        .. math::
            K = [k(x_i,y_j)]_{i,j=1}^{N,M},

        with :math:`\{x_i\}_{i=1}^N` the out-of-sample points (`x`) and :math:`\{y_i\}_{j=1}^N` the sample points
        (`y`).

        .. note::
            In the case of centered kernels, this computation is more expensive as it requires to _center according to
            the sample dataset, which implies computing a statistic on the out-of-sample kernel matrix and thus
            also computing it.

        :param x: Out-of-sample points (first dimension). If `None`, the default sample will be used., defaults to `None`
        :param y: Out-of-sample points (second dimension). If `None`, the default sample will be used., defaults to `None`

        :type x: Tensor(N,dim_input), optional
        :type y: Tensor(M,dim_input), optional

        :return: Kernel matrix
        :rtype: Tensor(N,M)

        :raises: PrimalError
        """
        # if x is None and y is None:
        #     return self.K
        # in order to get the values in the correct format (e.g. coming from numpy)
        x = utils.castf(x)
        y = utils.castf(y)
        transforms = self._get_transforms(transforms)
        return self._implicit_statistics.apply_transforms(data_fun=self._implicit, x=x, y=y, transforms=transforms)

    def c(self, x=None, y=None, transforms=None) -> Tensor:
        r"""
        Out-of-sample covariance matrix.
        """

        transforms = self._get_transforms(transforms)
        phi_x = self._explicit_statistics.apply_transforms(data_fun=self._explicit, x=x, transforms=transforms)
        if x == y:
            phi_y = phi_x
        else:
            phi_y = self._explicit_statistics.apply_transforms(data_fun=self._explicit, x=y, transforms=transforms)
        return phi_x.T @ phi_y

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

        fun = switcher.get(representation, utils.model.RepresentationError)
        return fun(x)

    def Cov(self, x=None, y=None) -> Tensor:
        transforms = self._default_transforms
        if transforms[-1] == "normalize":
            if transforms[-2] != "center":
                transforms = transforms.append("center").append("normalize")
        elif transforms[-1] == "center":
            transforms = transforms.append("normalize")
        else:
            transforms = transforms.append("center").append("normalize")

        return self.c(x=x, y=y, transforms=transforms)
