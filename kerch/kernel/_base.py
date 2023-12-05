"""
File containing the abstract kernel classes.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch
from torch import Tensor
from abc import ABCMeta, abstractmethod

from .. import utils
from .._sample import _Sample


@utils.extend_docstring(_Sample)
class _Base(_Sample, metaclass=ABCMeta):
    def __init__(self, **kwargs):
        super(_Base, self).__init__(**kwargs)
        self._log.debug("Initializing " + str(self))

    @abstractmethod
    def __str__(self):
        pass

    def __repr__(self):
        return self.__str__()

    # PROPERTIES
    @property
    @abstractmethod
    def explicit(self) -> bool:
        r"""
        True if the method has an explicit formulation, False otherwise.
        """
        pass

    @property
    def hparams(self):
        r"""
        Dictionnary containing the hyper-parameters and their values. This can be relevant for monitoring.
        """
        return {}

    @property
    def params(self) -> dict:
        r"""
        Dictionnary containing the parameters and their values. This can be relevant for monitoring.
        """
        return {}

    @property
    @abstractmethod
    def dim_feature(self) -> int:
        r"""
        Returns the dimension of the explicit feature map if it exists.
        """
        pass

    # def merge_idxs(self, **kwargs):
    #     raise NotImplementedError
    #     # self.dmatrix()
    #     # return torch.nonzero(torch.triu(self.dmatrix()) > (1 - kwargs["mtol"]), as_tuple=False)
    #
    # def merge(self, idxs):
    #     raise NotImplementedError
    #     # # suppress added up kernel
    #     # self._Sample = (self._Sample.gather(dim=0, index=idxs[:, 1]) +
    #     #                 self._Sample.gather(dim=0, index=idxs[:, 0])) / 2
    #
    #     self.dmatrix()
    #     # suppress added up kernel entries in the kernel matrix
    #     self._Cache["K"].gather(dim=0, index=idxs[:, 1], out=self._Cache["K"])
    #     self._Cache["K"].gather(dim=1, index=idxs[:, 1], out=self._Cache["K"])
    #
    # def reduce(self, idxs):
    #     raise NotImplementedError
    #     self._Sample.gather(dim=0, index=idxs, out=self._Sample)

    ###################################################################################################
    ################################### MATHS ARE HERE ################################################
    ###################################################################################################

    @abstractmethod
    def _implicit(self, x, y) -> Tensor:
        pass

    def _implicit_with_none(self, x=None, y=None) -> Tensor:
        # implicit raw
        if x is None:
            x = self.current_sample
        if y is None:
            y = self.current_sample
        return self._implicit(x, y)

    def _implicit_self(self, x=None):
        K = self._implicit_with_none(x, x)
        return torch.diag(K)

    @abstractmethod
    def _explicit(self, x) -> Tensor:
        pass

    def _explicit_with_none(self, x=None):
        # explicit raw
        if x is None:
            x = self.current_sample
        return self._explicit(x)

    def _phi(self):
        r"""
        Returns the explicit feature map with default centering and normalization. If already computed, it is
        recovered from the cache.

        :param overwrite: By default, the feature map is recovered from cache if already computed. Force overwrites
            this if True., defaults to False.
        """
        return self._get("phi", "lightweight", self.explicit)

    def _C(self) -> Tensor:
        r"""
        Returns the covariance matrix with default centering and normalization. If already computed, it is recovered
        from the cache.

        :param overwrite: By default, the covariance matrix is recovered from cache if already computed. Force overwrites
            this if True., defaults to False
                """
        def fun():
            scale = 1 / self.num_idx
            phi = self._phi()
            return scale * phi.T @ phi
        return self._get("C", "lightweight", fun)

    def _K(self, explicit=None, force : bool=False) -> Tensor:
        r"""
        Returns the kernel matrix with default centering and normalization. If already computed, it is recovered from
        the cache.

        :param explicit: Specifies whether the explicit or implicit formulation has to be used. Always uses the
            the explicit if available.
        :param force: By default, the kernel matrix is recovered from cache if already computed. Force overwrites
            this if True., defaults to False
        """
        def fun(explicit):
            if explicit is None: explicit = self.explicit
            if explicit:
                phi = self._phi()
                return phi @ phi.T
            else:
                return self._explicit_with_none()
        return self._get("K", "lightweight", lambda: fun(explicit))

    # ACCESSIBLE METHODS
    def phi(self, x=None) -> Tensor:
        r"""
        Returns the explicit feature map :math:`\phi(\cdot)` of the specified points.

        :param x: The datapoints serving as input of the explicit feature map. If `None`, the sample will be used.,
            defaults to `None`
        :type x: Tensor(,dim_input), optional
        :raises: ExplicitError
        """
        if x is None:
            return self._phi()

        x = utils.castf(x)
        x = self.project_sample(x)
        return self._explicit_with_none(x)

    def k(self, x=None, y=None, explicit=None) -> Tensor:
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

        :raises: ExplicitError
        """
        # if x is None and y is None:
        #     return self.K

        if x is None and y is None:
            return self._K(explicit=self.explicit)

        # in order to get the values in the correct format (e.g. coming from numpy)
        x = utils.castf(x)
        y = utils.castf(y)
        x = self.project_sample(x)
        y = self.project_sample(y)
        return self._implicit_with_none(x, y)

    def c(self, x=None) -> Tensor:
        r"""
        Out-of-sample explicit matrix.

        .. math::
            C = \frac1M\sum_{i}^{M} \phi(x_i)\phi(x_i)^\top.

        :param x: Out-of-sample points (first dimension). If `None`, the default sample will be used., defaults to `None`
        :type x: Tensor(N,dim_input), optional

        :return: Covariance matrix
        :rtype: Tensor(dim_feature,dim_feature)
        """

        phi = self.phi(x)
        scale = 1 / x.shape[0]
        return scale * phi.T @ phi

    def forward(self, x, representation="implicit") -> Tensor:
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

        def explicit(x):
            return self.phi(x)

        def implicit(x):
            return self.k(x)

        switcher = {"explicit": explicit,
                    "implicit": implicit}

        fun = switcher.get(representation, utils.model.RepresentationError)
        return fun(x)

    @property
    def K(self) -> Tensor:
        r"""
        Returns the kernel matrix on the sample dataset. Same result as calling :py:func:`k()`, but faster.
        It is loaded from memory if already computed and unchanged since then, to avoid re-computation when reccurently
        called.

        .. math::
            K_{ij} = k(x_i,x_j).
        """
        return self._K(explicit=self.explicit)

    # @property
    # def C(self) -> Tensor:
    #     r"""
    #     Returns the explicit matrix on the sample datapoints.
    #
    #     .. math::
    #         C = \frac1N\sum_i^N \phi(x_i)\phi(x_i)^\top.
    #     """
    #     return self._C()

    # @property
    # def Phi(self) -> Tensor:
    #     r"""
    #     Returns the explicit feature map :math:`\phi(\cdot)` of the sample datapoints. Same as calling
    #     :py:func:`phi()`, but faster.
    #     It is loaded from memory if already computed and unchanged since then, to avoid re-computation when reccurently
    #     called.
    #     """
    #     return self._phi()

    def implicit_preimage(self, k_coefficient: Tensor, method: str = 'knn', **kwargs):
        # DEFENSIVE
        k_coefficient = utils.castf(k_coefficient)

        if torch.all(k_coefficient < 0):
            self._log.warning(f"The argument k_coefficient contains negative values, which should never be the case by "
                              f"definition of a RKHS.")

        # PRE-IMAGE
        match method.lower():
            case 'knn':
                from ..preimage import knn
                return knn(k_coefficient, self, **kwargs)
            case 'smoother':
                from ..preimage import smoother
                return smoother(k_coefficient, self, **kwargs)
            case 'iterative':
                from ..preimage import iterative
                return iterative(k_coefficient, self, **kwargs)
            case _:
                raise AttributeError('Unknown or non-implemented preimage method.')

    @abstractmethod
    def explicit_preimage(self, phi: Tensor):
        pass
