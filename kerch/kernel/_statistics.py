"""
File containing the abstract kernel classes.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

from abc import ABCMeta, abstractmethod
from typing import List

import torch
from torch import Tensor

from .. import utils
from ._base import _Base
from kerch._transforms import TransformTree, _UnitSphereNormalization


@utils.extend_docstring(_Base)
class _Statistics(_Base, metaclass=ABCMeta):
    r"""
    :param kernel_transforms: A list composed of the elements `'normalize'` or `'center'`. For example a centered
        cosine kernel which is centered and normalized in order to get a covariance matrix for example can be obtained
        by invoking a linear kernel with `default_transforms = ['normalize', 'center', 'normalize']` or just a cosine
        kernel with `default_transforms = ['center', 'normalize']`. Redundancy is automatically handled., defaults
        to `[]`.
    :param ligthweight: During the computation of the statistics for centering and normalization, intermediate values
        are computed. It `True`, the model only keeps the necessary final statistics for the centering and
        normalization specified during construction. All other values are either never computed either discarded if
        required at some point. If asking for out-of-sample without the default centering and construction values, the
        statistics would then be computed and immediately discarded. They would have to be computed over and over again
        if repetively required. However, provided the same values are used as the one specified in the construction,
        this is the most efficient, not computing or keeping anything unnecessary. This parameter controls a
        time-memory trade-off., defaults to `True`
    :name kernel_transforms: List[str]
    :name ligthweight: bool, optional
    """

    @abstractmethod
    @utils.kwargs_decorator({
        "lightweight": True,
        "kernel_transforms": [],
    })
    def __init__(self, **kwargs):
        super(_Statistics, self).__init__(**kwargs)

        self._required_transforms = None
        self._naturally_centered = False
        self._naturally_normalized = False

        self._default_kernel_transforms = self._simplify_transforms(kwargs["kernel_transforms"])
        self._lightweight = kwargs["lightweight"]

    @property
    def kernel_transforms(self) -> TransformTree:
        r"""
        Default transforms performed on the kernel
        """
        if self.explicit:
            return self._explicit_statistics
        else:
            return self._implicit_statistics

    def _simplify_transforms(self, transforms=None) -> List:
        if transforms is None:
            transforms = []

        # add requirements
        if self._required_transforms is not None:
            transforms.append(self._required_transforms)

        # remove same following elements
        TransformTree.beautify_transforms(transforms)

        # remove unnecessary operation if kernel does it by default
        try:
            if self._naturally_normalized and transforms[0] == _UnitSphereNormalization:
                transforms.pop(0)
        except IndexError:
            pass

        return transforms

    def _get_transforms(self, transforms=None) -> List:
        if transforms is None:
            return self._default_kernel_transforms
        return self._simplify_transforms(transforms)

    @property
    def centered(self) -> bool:
        r"""
        Returns if the kernel is centered in the feature maps.
        """
        if not self._naturally_centered:
            try:
                return self._default_kernel_transforms[0] == 'center'
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
                return self._default_kernel_transforms[0] == 'normalize'
            except IndexError:
                return False
        return True

    @property
    def _explicit_statistics(self) -> TransformTree:
        if "explicit" not in self._cache:
            self._cache["explicit"] = TransformTree(explicit=True,
                                                    data=self._explicit,
                                                    default_transforms=self._default_kernel_transforms,
                                                    lighweight=self._lightweight)
        return self._cache["explicit"]

    @property
    def _implicit_statistics(self) -> TransformTree:
        if "implicit" not in self._cache:
            self._cache["implicit"] = TransformTree(explicit=False,
                                                    data=self._implicit,
                                                    default_transforms=self._default_kernel_transforms,
                                                    lighweight=self._lightweight,
                                                    implicit_self=self._implicit_self)
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
                self._cache["K"] = self._implicit_statistics.default_data
        return self._cache["K"]

    # ACCESSIBLE METHODS
    def phi(self, x=None, transforms=None) -> Tensor:
        r"""
        Returns the explicit feature map :math:`\phi(\cdot)` of the specified points.

        :param x: The datapoints serving as input of the explicit feature map. If `None`, the sample will be used.,
            defaults to `None`
        :name x: Tensor(,dim_input), optional
        :raises: PrimalError
        """
        x = utils.castf(x)
        transforms = self._get_transforms(transforms)
        return self._explicit_statistics.apply(data=self._explicit, x=self.transform_sample(x), transforms=transforms)

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

        :name x: Tensor(N,dim_input), optional
        :name y: Tensor(M,dim_input), optional

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
        if self.explicit:
            phi_x = self._explicit_statistics.apply(data=self._explicit, x=self.transform_sample(x), transforms=transforms)
            if utils.equal(x, y):
                phi_y = phi_x
            else:
                phi_y = self._explicit_statistics.apply(data=self._explicit, x=self.transform_sample(x), transforms=transforms)
            return phi_x @ phi_y.T
        else:
            if utils.equal(x, y):
                x = self.transform_sample(x)
                return self._implicit_statistics.apply(data=self._implicit, x=x, y=x, transforms=transforms)
            else:
                return self._implicit_statistics.apply(data=self._implicit, x=self.transform_sample(x),
                                                       y = self.transform_sample(y), transforms=transforms)

    def c(self, x=None, y=None, transforms=None) -> Tensor:
        r"""
        Out-of-sample covariance matrix.
        """

        transforms = self._get_transforms(transforms)
        phi_x = self._explicit_statistics.apply(data=self._explicit, x=self.transform_sample(x), transforms=transforms)
        if torch.equal(x,y):
            phi_y = phi_x
        else:
            phi_y = self._explicit_statistics.apply(data=self._explicit, x=self.transform_sample(y), transforms=transforms)
        return phi_x.T @ phi_y

    def forward(self, x, representation="dual") -> Tensor:
        """
        Passes datapoints through the kernel.

        :param x: Datapoints to be passed through the kernel.
        :param representation: Chosen representation. If `dual`, an out-of-sample kernel matrix is returned. If
            `primal` is specified, it returns the explicit feature map., defaults to `dual`

        :name x: Tensor(,dim_input)
        :name representation: str, optional

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
        transforms = self._default_kernel_transforms
        try:
            if not (self._default_kernel_transforms[-1] == 'normalize'
                    and (self._default_kernel_transforms[-2] == 'center'
                         or self._naturally_centered)):
                transforms.append('center')
                transforms.append('normalize')
        except IndexError:
            transforms.append('center')
            transforms.append('normalize')

        transforms = self._simplify_transforms(transforms)
        return self.c(x=x, y=y, transforms=transforms)
