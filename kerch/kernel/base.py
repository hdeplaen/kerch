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
    :type lightweight: bool, optional
    """

    @abstractmethod
    @utils.kwargs_decorator({
        "center": False,
        "normalize": False,
        "lightweight": True,
    })
    def __init__(self, **kwargs):
        super(base, self).__init__(**kwargs)
        self._log.debug("Initializing " + str(self))
        self._lightweight = kwargs["lightweight"]

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

    # PROPERTIES
    @property
    @abstractmethod
    def explicit(self) -> bool:
        r"""
        True if the method has an explicit formulation, False otherwise.
        """
        pass

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
    def sphere(self) -> bool:
        return self._sphere

    @sphere.setter
    def sphere(self, val: bool):
        self._sphere = val

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

    # STATISTICS AND CACHE
    def _implicit_statistics(self,
                             sphere : bool,
                             center : bool,
                             normalize : bool,
                             always_get_raw : bool = False) -> None:
        """
        Computes the dual matrix, also known as the kernel matrix.
        Its size is len(idx_kernels) * len(idx_kernels).

        :param idx_kernels: Index of the support vectors used to compute the kernel matrix. If nothing is provided, the kernel uses all_kernels of them.
        :return: Kernel matrix.
        """
        if self._empty_sample:
            self._log.error('No sample dataset. Please assign a sample dataset or specify the dimensions of the '
                              'sample dataset to initialize random values before computing kernel values.')
            raise Exception

        self._log.debug("Computing kernel matrix and dual statistics.")

        def _get_K_raw() -> Tensor:
            if "K_raw" not in self._cache:
                self._cache["K_raw"] = self._implicit()
            return self._cache["K_raw"]

        if always_get_raw:
            _get_K_raw()

        # to avoid redundancy (see later)
        if normalize and not center:
            sphere = True
            normalize = False

        if sphere:
            if "K_norm" not in self._cache:
                self._cache["K_norm"] = torch.sqrt(torch.diag(_get_K_raw()))[:, None]
        # center the mapping
        if center:
            if sphere:
                if "K_sphere_mean" not in self._cache:
                    self._cache["K_sphere"] = _get_K_raw() / torch.clamp(
                        self._cache["K_norm"] * self._cache["K_norm"].T, min=self._eps)
                    self._cache["K_sphere_mean"] = torch.mean(self._cache["K_sphere"], dim=1, keepdim=True)
                    self._cache["K_sphere_mean_tot"] = torch.mean(self._cache["K_sphere_mean"], dim=0)
            else:
                if "K_mean" not in self._cache:
                    self._cache["K_mean"] = torch.mean(_get_K_raw(), dim=1, keepdim=True)
                    self._cache["K_mean_tot"] = torch.mean(self._cache["K_mean"], dim=0)
        # normalize the mapping (only relevant if centered, otherwise can already be done with sphere)
        if normalize:
            if center:
                if sphere:
                    if "K_sphere_centered_norm" not in self._cache:
                        self._cache["K_sphere_centered"] = self._cache["K_sphere"] \
                                                           - self._cache["K_sphere_mean"] \
                                                           - self._cache["K_sphere_mean"].T \
                                                           + self._cache["K_sphere_mean_tot"]
                        self._cache["K_sphere_centered_norm"] = torch.sqrt(
                            torch.diag(self._cache["K_sphere_centered"]))[:, None]
                else:
                    if "K_centered_norm" not in self._cache:
                        self._cache["K_centered"] = _get_K_raw() \
                                                    - self._cache["K_mean_sphere"] \
                                                    - self._cache["K_mean_sphere"].T \
                                                    + self._cache["K_mean_tot"]
                        self._cache["K_centered_norm"] = torch.sqrt(torch.diag(self._cache["K_centered"]))[:, None]
        else:
        # other cases should never happen
            raise Exception("Internal implementation error: this case should never happen.")

        if self._lightweight and not always_get_raw:
            self._remove_from_cache("K_raw")

    def _explicit_statistics(self,
                             sphere : bool,
                             center : bool,
                             normalize : bool,
                             always_get_raw : bool = False):
        """
        Computes the primal matrix, i.e. correlation between the different outputs.
        Its size is output * output.
        """
        if self._empty_sample:
            self._log.error('No sample dataset. Please specify a sample dataset or the dimensions of the sample '
                              'dataset to initialize random values before computing kernel values.')
            raise Exception

        self._log.debug("Computing explicit feature map and primal statistics.")

        def _get_phi_raw() -> Tensor:
            if "phi_raw" not in self._cache:
                self._cache["phi_raw"] = self._explicit()
            return self._cache["phi_raw"]

        if always_get_raw:
            _get_phi_raw()

        # this case is redundant (see below)
        if normalize and not center:
            sphere = True
            normalize = False

        # project the mapping onto a sphere (normalization at the source)
        if sphere:
            if "phi_norm" not in self._cache:
                self._cache["phi_norm"] = torch.norm(_get_phi_raw(), dim=1, keepdim=True)
        # center the mapping
        if center:
            if sphere:
                if "phi_sphere_mean" not in self._cache:
                    self._cache["phi_sphere"] = _get_phi_raw() / torch.clamp(self._cache["phi_norm"], min=self._eps)
                    self._cache["phi_sphere_mean"] = torch.mean(self._cache["phi_sphere"], dim=0)
            else:
                if "phi_mean" not in self._cache:
                    self._cache["phi_sphere_mean"] = torch.mean(_get_phi_raw(), dim=0)
                    self._cache["phi_centered"] = _get_phi_raw() - self._cache["phi_mean_sphere"]
        # normalize the mapping (only relevant if centered, otherwise can already be done with sphere)
        if normalize:
            if center:
                if sphere:
                    if "phi_sphere_centered_norm" not in self._cache:
                        self._cache["phi_sphere_centered"] = self._cache["phi_sphere"] - self._cache["phi_sphere_mean"]
                        self._cache["phi_sphere_centered_norm"] = torch.norm(self._cache["phi_sphere_centered"], dim=1, keepdim=True)
                else:
                    if "phi_centered_norm" not in self._cache:
                        self._cache["phi_centered"] = _get_phi_raw() - self._cache["phi_mean_sphere"]
                        self._cache["phi_centered_norm"] = torch.norm(self._cache["phi_centered"], dim=1, keepdim=True)
            else:
                # other cases should never happen
                raise Exception("Internal implementation error: this case should never happen.")

        if self._lightweight and not always_get_raw:
            self._remove_from_cache("phi_raw")

    def _phi(self, force: bool = False):
        r"""
        Returns the explicit feature map with default centering and normalization. If already computed, it is
        recovered from the cache.

        :param force: By default, the feature map is recovered from cache if already computed. Force overwrites
            this if True., defaults to False.
        """
        if "phi" not in self._cache or force:
            self._explicit_statistics(sphere=self._sphere,
                                      center=self._center,
                                      normalize=self._normalize,
                                      always_get_raw=True)
            if self._sphere and not self._center and not self._normalize:
                if "phi_sphere" not in self._cache:
                    self._cache["phi_sphere"] = self._cache["phi_raw"] / torch.clamp(
                        self._cache["phi_norm"], min=self._eps)
                self._cache["phi"] = self._cache["phi_sphere"]

            elif self._sphere and self._center and not self._normalize:
                if "phi_sphere_centered" not in self._cache:
                    self._cache["phi_sphere_centered"] = self._cache["phi_sphere"] - self._cache["phi_sphere_mean"]
                self._cache["phi"] = self._cache["phi_sphere_centered"]

            elif self._sphere and not self._center and self._normalize:
                # this case is redundant and should not happen
                if "phi_sphere" not in self._cache:
                    self._cache["phi_sphere"] = self._cache["phi_raw"] / torch.clamp(
                        self._cache["phi_norm"], min=self._eps)
                self._cache["phi"] = self._cache["phi_sphere"]

            elif self._sphere and self._center and self._normalize:
                self._cache["phi"] = self._cache["phi_sphere_centered"] / torch.clamp(
                    self._cache["phi_sphere_centered_norm"], min = self._eps)

            elif not self._sphere and not self._center and not self._normalize:
               self._cache["phi"] = self._cache["phi_raw"]

            elif not self._sphere and self._center and not self._normalize:
                if "phi_centered" not in self._cache:
                    self._cache["phi_centered"] = self._cache["phi"] - self._cache["phi_mean"]
                self._cache["phi"] = self._cache["phi_centered"]

            elif not self._sphere and not self._center and self._normalize:
                # this case is redundant and should not happen
                if "phi_sphere" not in self._cache:
                    self._cache["phi_sphere"] = self._cache["phi_raw"] / torch.clamp(
                        self._cache["phi_norm"], min=self._eps)
                self._cache["phi"] = self._cache["phi_sphere"]

            elif not self._sphere and self._center and self._normalize:
                self._cache["phi"] = self._cache["phi_centered"] / torch.clamp(
                    self._cache["phi_centered_norm"], min = self._eps)

            self._lighten_statistics()
        return self._cache["phi"]

    def _C(self, force: bool = False) -> Tensor:
        r"""
        Returns the covariance matrix with default centering and normalization. If already computed, it is recovered
        from the cache.

        :param force: By default, the covariance matrix is recovered from cache if already computed. Force overwrites
            this if True., defaults to False
                """
        if "C" not in self._cache or force:
            phi = self._phi(force=force)
            self._cache["C"] = phi.T @ phi
        return self._cache["C"]

    def _K(self, explicit=None, force: bool = False) -> Tensor:
        r"""
        Returns the kernel matrix with default centering and normalization. If already computed, it is recovered from
        the cache.

        :param explicit: Specifies whether the explicit or implicit formulation has to be used. Always uses the
            explicit if available.
        :param force: By default, the kernel matrix is recovered from cache if already computed. Force overwrites
            this if True. This can be relevant in the case where it can be computed both implicitly and explicitly.,
            defaults to False
        """
        if explicit is None: explicit = self.explicit
        if explicit:
            phi = self._phi(force)
            if "K" not in self._cache or force:
                self._cache["K"] = phi @ phi.T
        else:
            self._implicit_statistics(sphere = self._sphere,
                                      center=self._center,
                                      normalize=self._normalize,
                                      always_get_raw=True)

            if self._sphere and self._center and self._normalize:
                self._cache["K"] = self._cache["K_sphere_centered"] / torch.clamp(
                    self._cache["K_sphere_centered_norm"] * self._cache["K_sphere_centered_norm"].T, min = self._eps)
            elif self._sphere and self._center and not self._normalize:
                if "K_sphere_centered" not in self._cache:
                    self._cache["K_sphere_centered"] = self._cache["K_sphere"] \
                                                       - self._cache["K_sphere_mean"] \
                                                       - self._cache["K_sphere_mean"].T \
                                                       + self._cache["K_sphere_mean_tot"]
                self._cache["K"] = self._cache["K_sphere_centered"]
            elif self._sphere and not self._center and self._normalize:
                # redundant
                self._cache["K"] = self._cache["K_sphere_centered"] / torch.clamp(
                    self._cache["K_sphere_centered_norm"] * self._cache["K_sphere_centered_norm"].T, min=self._eps)
            elif self._sphere and not self._center and not self._normalize:
                if "K_sphere" not in self._cache:
                    self._cache["K_sphere"] = self._cache["K_raw"] / torch.clamp(
                        self._cache["K_norm"] * self._cache["K_norm"].T, min=self._eps)
            elif not self._sphere and self._center and self._normalize:
                self._cache["K"] = self._cache["K_centered"] / torch.clamp(
                    self._cache["K_centered_norm"] * self._cache["K_centered_norm"].T, min = self._eps)
            elif not self._sphere and self._center and not self._normalize:
                if "K_centered" not in self._cache:
                    self._cache["K_centered"] = self._cache["K_raw"] \
                                                - self._cache["K_mean_sphere"] \
                                                - self._cache["K_mean_sphere"].T \
                                                + self._cache["K_mean_tot"]
                self._cache["K"] = self._cache["K_centered"]
            elif not self._sphere and not self._center and self._normalize:
                # redundant
                self._cache["K"] = self._cache["K_sphere_centered"] / torch.clamp(
                    self._cache["K_sphere_centered_norm"] * self._cache["K_sphere_centered_norm"].T, min=self._eps)
            elif not self._sphere and not self._center and not self._normalize:
                self._cache["K"] = self._cahce["K_raw"]
            else:
                # should never happen
                raise Exception("Internal implementation error. This case should not happen.")

            self._lighten_statistics()
        return self._cache["K"]

    def _lighten_statistics(self) -> None:
        r"""
        Removes cache elements to keep the model lightweight
        """
        if self._lightweight:
            self._remove_from_cache("phi_raw")
            self._remove_from_cache("K_raw")
            if not self._sphere:
                self._remove_from_cache("phi_norm")
                self._remove_from_cache("K_norm")
            if not self._center:
                self._remove_from_cache("phi_centered")
                self._remove_from_cache("phi_mean")
                self._remove_from_cache("phi_sphere_centered")
                self._remove_from_cache("phi_sphere_mean")
                self._remove_from_cache("K_centered")
                self._remove_from_cache("K_mean")
                self._remove_from_cache("K_mean_tot")
                self._remove_from_cache("K_sphere_centered")
                self._remove_from_cache("K_sphere_mean")
                self._remove_from_cache("K_sphere_mean_tot")
            if not self._normalize:
                self._remove_from_cache("phi_centered_norm")
                self._remove_from_cache("K_centered_norm")
                self._remove_from_cache("phi_sphere_centered_norm")
                self._remove_from_cache("K_sphere_centered_norm")

    # ACCESSIBLE METHODS
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
        if center is None: center = self._center
        if normalize is None: normalize = self._normalize

        # if the default value is required and the default centering and normalization are asked, it can directly
        # be recovered from the cache (and added to the cache if not already computed).
        if x is None and center == self._center and normalize == self._normalize:
            return self._phi()

        x = utils.castf(x)
        phi = self._explicit(x)

        # check that statistics are available
        self._explicit_statistics(center, normalize)

        if center and not normalize:
            phi = phi - self._cache["phi_mean"]
        elif normalize:
            phi_norm = torch.norm(phi, dim=1, keepdim=True)
            phi = phi / torch.clamp(phi_norm, min=self._eps)
            if center:
                phi = phi - self._cache["phi_normalized_mean"]

        self._lighten_statistics()
        return phi

    def k(self, x=None, y=None, explicit=None, center=None, normalize=None) -> Tensor:
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

        # recover default values
        if explicit is None: explicit = self.explicit
        if center is None: center = self._center
        if normalize is None: normalize = self._normalize

        # if the default value is required and the default centering and normalization are asked, it can directly
        # be recovered from the cache (and added to the cache if not already computed).
        if x is None and y is None and center == self._center and normalize == self._normalize:
            return self._K(explicit)

        # in order to get the values in the correct format (e.g. coming from numpy)
        x = utils.castf(x)
        y = utils.castf(y)

        # now that we know that the sample prerequisites are met, we can compute the OOS.
        if explicit:
            phi_sample = self.phi(y, center, normalize)
            phi_oos = self.phi(x, center, normalize)
            K = phi_oos @ phi_sample.T
        else:
            self._implicit_statistics(center=center, normalize=normalize)
            K = self._implicit(x, y)

            if center and not normalize:
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
            elif normalize:
                if x is None:
                    n_x = self._cache["K_norm"]
                else:
                    diag_K_x = self._implicit_self(x)[:, None]
                    n_x = torch.sqrt(diag_K_x)
                if y is None:
                    n_y = self._cache["K_norm"]
                else:
                    diag_K_y = self._implicit_self(x)[:, None]
                    n_y = torch.sqrt(diag_K_y)
                K_norm = n_x * n_y.T
                K = K / torch.clamp(K_norm, min=self._eps)
                if center:
                    if x is None:
                        m_norm_x = self._cache["K_normalized_mean"]
                    else:
                        K_x_sample = self._implicit(x)
                        K_x_sample_norm = n_x * self._cache["K_norm"].T
                        K_x_sample_normalized = K_x_sample / torch.clamp(K_x_sample_norm, min=self._eps)
                        m_norm_x = torch.mean(K_x_sample_normalized, dim=1, keepdim=True)
                    if y is None:
                        m_norm_y = self._cache["K_normalized_mean"]
                    else:
                        K_y_sample = self._implicit(x)
                        K_y_sample_norm = n_y * self._cache["K_norm"].T
                        K_y_sample_normalized = K_y_sample / torch.clamp(K_y_sample_norm, min=self._eps)
                        m_norm_y = torch.mean(K_y_sample_normalized, dim=1, keepdim=True)
                    K = K - m_norm_x \
                        - m_norm_y.T \
                        + self._cache["K_normalized_mean_tot"]

        self._lighten_statistics()
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
        return self._K(explicit=self.explicit)

    @property
    def C(self) -> Tensor:
        r"""
        Returns the covariance matrix on the sample datapoints.

        .. math::
            C = \frac1N\sum_i^N \phi(x_i)\phi(x_i)^\top.
        """
        return self._C()

    @property
    def Phi(self) -> Tensor:
        r"""
        Returns the explicit feature map :math:`\phi(\cdot)` of the sample datapoints. Same as calling
        :py:func:`phi()`, but faster.
        It is loaded from memory if already computed and unchanged since then, to avoid re-computation when reccurently
        called.
        """
        return self._phi()
