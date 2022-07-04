"""
File containing the abstract kernel classes.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch
from typing import Iterator

from abc import ABCMeta, abstractmethod
from torch import Tensor

from .. import utils
from .._sample import _Sample


@utils.extend_docstring(_Sample)
class base(_Sample, metaclass=ABCMeta):
    r"""
    :param center: `True` if any implicit feature or kernel is must be centered, `False` otherwise. The center
        is always performed relative to a statistic on the sample., defaults to `False`
    :param normalize: `True` if any implicit feature or kernel is must be normalized, `False` otherwise. The center
        is always performed relative to a statistic on the sample., defaults to `False`
    :type center: bool, optional
    :type normalize: bool, optional
    """

    @abstractmethod
    @utils.kwargs_decorator({
        "center": False,
        "normalize": False
    })
    def __init__(self, **kwargs):
        super(base, self).__init__(**kwargs)
        self._log.debug("Initializing " + str(self))

        ## CENTERING
        self._center = kwargs["center"]

        ## NORMALIZATION
        normalize = kwargs["normalize"]
        if normalize is True or normalize is False:
            self._eps = 1.e-8
            self._normalize_requested = normalize
        else:
            self._eps = normalize
            self._normalize_requested = True

        # It may be that some kernels are naturally normalized and don't need the additional computation
        self._normalize = self._normalize_requested

    @abstractmethod
    def __str__(self):
        pass

    def __repr__(self):
        return self.__str__()

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

    @property
    def center(self) -> bool:
        r"""
        Indicates if the kernel has to be centered. Changing this value leads to a recomputation of the statistics.
        """
        return self._center

    @center.setter
    def center(self, val: bool):
        self._center = val
        self._reset()

    @property
    def normalize(self) -> bool:
        r"""
        Indicates if the kernel has to be normalized. Changing this value leads to a recomputation of the statistics.
        """
        return self._normalize

    @normalize.setter
    def normalize(self, val: bool):
        self._normalize = val
        self._reset()

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
    def _implicit(self, x=None, y=None):
        # implicit without center
        if x is None:
            x = self.current_sample
        if y is None:
            y = self.current_sample
        return x, y

    def _implicit_self(self, x=None):
        K = self._implicit(x, x)
        return torch.diag(K)

    @abstractmethod
    def _explicit(self, x=None):
        # explicit without center
        if x is None:
            x = self.current_sample
        return x

    def _compute_K(self, implicit=False):
        """
        Computes the dual matrix, also known as the kernel matrix.
        Its size is len(idx_kernels) * len(idx_kernels).

        :param idx_kernels: Index of the support vectors used to compute the kernel matrix. If nothing is provided, the kernel uses all_kernels of them.
        :return: Kernel matrix.
        """
        if self._empty_sample:
            self._log.warning('No sample dataset. Please assign a sample dataset or specify the dimensions of the '
                              'sample dataset to initialize random values before computing kernel values.')
            return None

        if "K" not in self._cache:
            self._log.debug("Computing kernel matrix and dual statistics.")
            if implicit:
                phi = self.phi()
                self._cache["K"] = phi @ phi.T
            else:
                self._cache["K"] = self._implicit()

                # centering in the implicit case happens ad hoc
                if self._center:
                    self._cache["K_mean"] = torch.mean(self._cache["K"], dim=1, keepdim=True)
                    self._cache["K_mean_tot"] = torch.mean(self._cache["K"], dim=(0, 1))
                    self._cache["K"] = self._cache["K"] - self._cache["K_mean"] \
                                       - self._cache["K_mean"].T \
                                       + self._cache["K_mean_tot"]
                if self._normalize:
                    self._cache["K_norm"] = torch.sqrt(torch.diag(self._cache["K"]))[:, None]
                    K_norm = self._cache["K_norm"] * self._cache["K_norm"].T
                    self._cache["K"] = self._cache["K"] / torch.clamp(K_norm, min=self._eps)

        return self._cache["K"]

    def _compute_C(self):
        """
        Computes the primal matrix, i.e. correlation between the different outputs.
        Its size is output * output.
        """
        if self._empty_sample:
            self._log.warning('No sample dataset. Please specify a sample dataset or the dimensions of the sample '
                              'dataset to initialize random values before computing kernel values.')
            return None

        if "C" not in self._cache:
            self._log.debug("Computing explicit feature map and primal statistics.")
            self._cache["phi"] = self._explicit()

            if self._center:
                self._cache["phi_mean"] = torch.mean(self._cache["phi"], dim=0)
                self._cache["phi"] = self._cache["phi"] - self._cache["phi_mean"]
            if self._normalize:
                self._cache["phi_norm"] = torch.norm(self._cache["phi"], dim=1, keepdim=True)
                self._cache["phi"] = self._cache["phi"] / self._cache["phi_norm"]
            self._cache["C"] = self._cache["phi"].T @ self._cache["phi"]
        return self._cache["C"], self._cache["phi"]

    def phi(self, x=None, center=None, normalize=None) -> Tensor:
        r"""
        Returns the explicit feature map :math:`\phi(\cdot)` of the specified points.

        :param x: The datapoints serving as input of the explicit feature map. If `None`, the sample will be used.,
            defaults to `None`
        :type x: Tensor(,dim_input), optional
        :param center: Returns if the matrix has to be centered or not. If None, then the default value used during
            construction is used., defaults to None
        :param normalize: Returns if the matrix has to be normalized or not. If None, then the default value used during
            construction is used., defaults to None
        :type center: bool, optional
        :type normalize: bool, optional
        :raises: PrimalError
        """

        # if x is None, phi(x) for x in the sample is returned.
        if center is None:
            center = self._center
        if normalize is None:
            normalize = self._normalize

        # check is statistics are available if required
        if center or normalize or x is None:
            if self._compute_C() is None:
                self._log.error('Impossible to compute statistics on the sample (probably due to an undefined sample.')
                raise Exception

        x = utils.castf(x)
        phi = self._explicit(x)
        if center:
            phi = phi - self._cache["phi_mean"]
        if normalize:
            phi_norm = torch.norm(phi, dim=1, keepdim=True)
            phi = phi / torch.clamp(phi_norm, min=self._eps)

        return phi

    def k(self, x=None, y=None, implicit=False, center=None, normalize=None) -> Tensor:
        """
        Returns a kernel matrix, either of the sample, either out-of-sample, either fully out-of-sample.

        .. math::
            K = [k(x_i,y_j)]_{i,j=1}^{N,M},

        with :math:`\{x_i\}_{i=1}^N` the out-of-sample points (`x`) and :math:`\{y_i\}_{j=1}^N` the sample points
        (`y`).

        .. note::
            In the case of centered kernels, this computation is more expensive as it requires to center according to
            the sample dataset, which implies computing a statistic on the out-of-sample kernel matrix and thus
            also computing it.

        :param x: Out-of-sample points (first dimension). If `None`, the default sample will be used., defaults to `None`
        :param y: Out-of-sample points (second dimension). If `None`, the default sample will be used., defaults to `None`

        :type x: Tensor(N,dim_input), optional
        :type y: Tensor(M,dim_input), optional

        :param center: Returns if the matrix has to be centered or not. If None, then the default value used during
            construction is used., defaults to None
        :param normalize: Returns if the matrix has to be normalized or not. If None, then the default value used during
            construction is used., defaults to None
        :type center: bool, optional
        :type normalize: bool, optional

        :return: Kernel matrix
        :rtype: Tensor(N,M)

        :raises: PrimalError
        """
        # if x is None and y is None:
        #     return self.K

        if center is None:
            center = self._center
        if normalize is None:
            normalize = self._normalize

        x = utils.castf(x)
        y = utils.castf(y)

        # if any computation on the sample is required
        if center or normalize or x is None or y is None:
            if implicit and self._compute_C() is None:
                self._log.error('Impossible to compute statistics on the sample (probably due to an undefined sample.')
                raise Exception
            elif not implicit and self._compute_K() is None:
                self._log.error('Impossible to compute statistics on the sample (probably due to an undefined sample.')
                raise Exception

        # now that we know that the sample prerequisites are met, we can compute the OOS.
        if implicit:
            phi_sample = self.phi(y, center, normalize)
            phi_oos = self.phi(x, center, normalize)
            return phi_oos @ phi_sample.T
        else:
            K = self._implicit(x, y)
            if center:
                if x is not None:
                    K_x_sample = self._implicit(x)
                    m_x_sample = torch.mean(K_x_sample, dim=1, keepdim=True)
                else:
                    m_x_sample = self._cache["K_mean"]

                if y is not None:
                    K_y_sample = self._implicit(y)
                    m_y_sample = torch.mean(K_y_sample, dim=1, keepdim=True)
                else:
                    m_y_sample = self._cache["K_mean"]

                K = K - m_x_sample \
                    - m_y_sample.T \
                    + self._cache["K_mean_tot"]
            if normalize:
                if x is None:
                    n_x = self._cache["K_norm"]
                else:
                    diag_K_x = self._implicit_self(x)[:, None]
                    if center:
                        diag_K_x = diag_K_x - 2 * m_x_sample + self._cache["K_mean_tot"]
                    n_x = torch.sqrt(diag_K_x)

                if y is None:
                    n_y = self._cache["K_norm"]
                else:
                    diag_K_y = self._implicit_self(y)[:, None]
                    if center:
                        diag_K_y = diag_K_y - 2 * m_y_sample + self._cache["K_mean_tot"]
                    n_y = torch.sqrt(diag_K_y)

                K_norm = n_x * n_y.T
                K = K / torch.clamp(K_norm, min=self._eps)

            return K

    def c(self, x=None, y=None, center=None, normalize=None) -> Tensor:
        r"""
        Out-of-sample covariance matrix.

        .. note::
            A centered and normalized covariance matrix is a correlation matrix.
        """
        return self.phi(x, center=center, normalize=normalize).T \
               @ self.phi(y, center=center, normalize=normalize)

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

    @property
    def K(self) -> Tensor:
        r"""
        Returns the kernel matrix on the sample dataset. Same result as calling :py:func:`k()`, but faster.
        It is loaded from memory if already computed and unchanged since then, to avoid re-computation when reccurently
        called.

        .. math::
            K_{ij} = k(x_i,x_j).
        """
        return self._compute_K()

    @property
    def C(self) -> Tensor:
        r"""
        Returns the covariance matrix on the sample datapoints.

        .. math::
            C = \frac1N\sum_i^N \phi(x_i)\phi(x_i)^\top.
        """
        return self._compute_C()[0]

    @property
    def phi_sample(self) -> Tensor:
        r"""
        Returns the explicit feature map :math:`\phi(\cdot)` of the sample datapoints. Same as calling
        :py:func:`phi()`, but faster.
        It is loaded from memory if already computed and unchanged since then, to avoid re-computation when reccurently
        called.
        """
        return self._compute_C()[1]

