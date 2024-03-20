# coding=utf-8
from __future__ import annotations

import torch
from torch import Tensor as T
from typing import Optional, Iterator
from abc import ABCMeta, abstractmethod

from ._Level import _Level
from .. import utils


class _PPCA(_Level, metaclass=ABCMeta):
    @utils.kwargs_decorator({'use_mean': False,
                             'feature_noise': None})
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_mean = kwargs["use_mean"]
        self.feature_noise = kwargs["feature_noise"]
        self._vals = torch.nn.Parameter(torch.empty(0, dtype=utils.FTYPE),
                                        requires_grad=False)
        self._parameter_related_cache = [*self._parameter_related_cache,
                                         "_B_primal", "_B_dual", "_Inv_primal", "_Inv_dual"]

    @property
    def use_mean(self) -> bool:
        return self._use_mean

    @use_mean.setter
    def use_mean(self, val: bool) -> None:
        self._use_mean = val

    @property
    def vals(self) -> T:
        r"""
        Eigenvalues of the model. The model has to be fitted for these values to exist.
        """
        return self._vals

    @property
    def sqrt_vals(self) -> T:
        return self._get(key="sqrt_vals", fun=lambda: torch.sqrt(self.vals))

    @vals.setter
    def vals(self, val):
        val = utils.castf(val, tensor=False, dev=self._vals.device)
        self._vals.data = val

    def total_variance(self, as_tensor=False, normalize=True, representation=None) -> float | T:
        r"""
        Total variance contained in the feature map. In primal formulation,
        this is given by :math:`\DeclareMathOperator{\tr}{tr}\tr(C)`, where :math:`C = \sum\phi(x)\phi(x)^\top` is
        the covariance matrix on the sample. In dual, this is given by :math:`\DeclareMathOperator{\tr}{tr}\tr(K)`,
        where :math:`K_{ij} = k(x_i,x_j)` is the kernel matrix on the sample.

        :param as_tensor: Indicated whether the variance has to be returned as a float or a torch.Tensor., defaults
            to ``False``
        :type as_tensor: bool, optional

        .. warning::
            For this value to strictly be interpreted as a variance, the corresponding kernel (or feature map)
            has to be normalized. In that case however, the total variance will amount to the dimension of the feature
            map in primal and the number of datapoints in dual.
        """
        representation = utils.check_representation(representation=representation, default=self._representation)
        level_key = "KPCA_total_variance_default_representation" if representation == self._representation \
            else "KPCA_total_variance_other_representation"
        if representation == 'primal':
            var = self._get("total_variance_primal", level_key=level_key, fun=lambda: torch.trace(self.C))
        else:
            var = self._get("total_variance_dual", level_key=level_key, fun=lambda: torch.trace(self.K))
        if normalize:
            var /= self.num_idx
        if as_tensor:
            return var
        return var.detach().cpu().numpy()

    def model_variance(self, as_tensor=False, normalize=True) -> float | T:
        r"""
        Total variance learnt by the model given by the sum of the eigenvalues.

        :param as_tensor: Indicated whether the variance has to be returned as a float or a torch.Tensor., defaults
            to ``False``
        :type as_tensor: bool, optional

        .. warning::
            For this value to strictly be interpreted as a variance, the corresponding kernel (or feature map)
            has to be normalized. We refer to the remark of ``total_variance``.
        """
        var = torch.sum(self.vals)
        if normalize:
            var /= self.num_idx
        if as_tensor:
            return var
        return var.detach().cpu().numpy()

    def _reset_dual(self) -> None:
        super(_PPCA, self)._reset_dual()
        self._remove_from_cache(["total_variance_primal", "total_variance_dual"])

    def relative_variance(self, as_tensor=False) -> float | T:
        r"""
        Relative variance learnt by the model given by ```model_variance``/``total_variance``.
        This number is always comprised between 0 and 1 and avoids any considerations on normalization.

        :param as_tensor: Indicated whether the variance has to be returned as a float or a torch.Tensor., defaults
            to ``False``
        :type as_tensor: bool, optional
        """
        var = self.model_variance(as_tensor=as_tensor, normalize=False) / \
              self.total_variance(as_tensor=as_tensor, normalize=False)
        return var

    @property
    def feature_noise(self) -> float:
        r"""
        Sigma value, acts as a regularization parameter.
        """
        return self._feature_noise

    @feature_noise.setter
    def feature_noise(self, val: Optional[float]) -> None:
        if val is None:
            self._feature_noise = None
        else:
            self._feature_noise = float(val)

    @property
    def mu(self) -> torch.nn.Parameter:
        return self._mu

    @mu.setter
    def mu(self, val: torch.Tensor) -> None:
        self._mu.data = utils.castf(val, dev=self.sample.device, tensor=True)

    ########################################################################################################
    @property
    @torch.no_grad()
    def _B_primal(self) -> T:
        def compute() -> T:
            return torch.cholesky(self.weight @ self.weight.T + self.feature_noise ** 2 * self._I_primal)

        return self._get(key="_B_primal", level_key="PPCA_B_primal", fun=compute)

    @_B_primal.setter
    def _B_primal(self, val: T) -> None:
        self._get(key="_B_primal", level_key="PPCA_B_primal", force=True, fun=lambda: val)

    @property
    @torch.no_grad()
    def _B_dual(self) -> T:
        def compute() -> T:
            return torch.cholesky(self.hidden.T @ self.hidden + self.feature_noise ** 2 * torch.inv(self.K))

        return self._get(key="_B_dual", level_key="PPCA_B_dual", fun=compute)

    @_B_dual.setter
    def _B_dual(self, val: T) -> None:
        self._get(key="_B_dual", level_key="PPCA_B_dual", force=True, fun=lambda: val)

    @property
    @torch.no_grad()
    def _Inv_primal(self) -> T:
        def compute() -> T:
            return torch.inv(self.weight.T @ self.weight + self.feature_noise ** 2 * self._I_primal)

        return self._get(key="_Inv_primal", level_key="PPCA_Inv_primal", fun=compute)

    @_Inv_primal.setter
    @torch.no_grad()
    def _Inv_primal(self, val: T) -> None:
        self._get(key="_Inv_primal", level_key="PPCA_Inv_primal", fun=lambda: val, overwrite=True)

    @property
    @torch.no_grad()
    def _Inv_dual(self) -> T:
        def compute() -> T:
            return torch.inv(self.hidden.T @ self.K @ self.hidden + self.feature_noise ** 2 * self._I_dual)

        return self._get(key="_Inv_dual", level_key="PPCA_Inv_dual", fun=compute)

    @_Inv_dual.setter
    @torch.no_grad()
    def _Inv_dual(self, val: T) -> None:
        self._get(key="_Inv_dual", level_key="PPCA_Inv_dual", fun=lambda: val, overwrite=True)

    ########################################################################################################################

    @torch.no_grad()
    def h_map(self, phi: Optional[T] = None, k: Optional[T] = None) -> T:
        r"""
        Draws a `h` given the maximum a posteriori of the distribution. By choosing the input, you either
        choose a primal or dual representation.

        :param phi: Primal representation.
        :param k: Dual representation.
        :type phi: Tensor[N, dim_input], optional
        :type k: Tensor[N, num_idx], optional
        :return: MAP of h given phi or k.
        :rtype: Tensor[N, dim_output]
        """

        if phi is not None and k is None:
            return phi @ self.weight.T @ self._Inv_primal
        if phi is None and k is not None:
            return k @ self.hidden.T @ self._Inv_dual
        else:
            raise AttributeError("One and only one attribute phi or k has to be specified.")

    @torch.no_grad()
    def phi_map(self, h: T) -> T:
        r"""
        Maximum a posteriori of phi given h.

        :param h: Latent representation.
        :type h: Tensor[N, dim_output]
        :return: MAP of phi given h.
        :rtype: Tensor[N, dim_input]
        """
        if self.use_mean:
            return h @ self.weight.T + self.mu
        return h @ self.weight.T

    @torch.no_grad()
    def k_map(self, h: T) -> T:
        r"""
        Maximum a posteriori of k given h.

        :param h: Latent representation.
        :type h: Tensor[N, dim_output]
        :return: MAP of k given h.
        :rtype: Tensor[N, num_idx]
        """
        if self.use_mean:
            raise NotImplementedError
        return h @ self.hidden.T @ self.K

    @torch.no_grad()
    def draw_h(self, num: int = 1) -> T:
        r"""
        Draws a h given its prior distribution.

        :param num: Number of h to be sampled, defaults to 1.
        :type num: int, optional
        :return: Latent representation.
        :rtype: Tensor[num, dim_output]
        """
        return torch.randn((num, self.dim_output), device=self.sample.device, dtype=utils.FTYPE)

    @torch.no_grad()
    def draw_phi(self, num: int = 1, posterior: bool = True) -> T:
        r"""
        Draws a primal representation phi given its posterior distribution.

        :param posterior: Indicates whether phi has to be drawn from its posterior distribution or its conditional
            given the prior of h. Defaults to True.
        :param num: Number of phi to be sampled, defaults to 1.
        :type num: int, optional
        :type posterior: bool, optional
        :return: Primal representation.
        :rtype: Tensor[num, dim_input]
        """
        if posterior:
            u = torch.randn((num, self.dim_feature), dtype=utils.FTYPE, device=self.sample.device)
            if self.use_mean:
                return u @ self._B_primal + self.mu
            return u @ self._B_primal
        h = self.draw_h(num)
        return self.phi_map(h)

    @torch.no_grad()
    def draw_k(self, num: int = 1, posterior: bool = False) -> T:
        r"""
        Draws a dual representation k given its posterior distribution.

        :param posterior: Indicates whether phi has to be drawn from its posterior distribution or its conditional
            given the prior of h. Defaults to True.
        :param num: Number of k to be sampled, defaults to 1.
        :type num: int, optional
        :type posterior: bool, optional
        :return: Dual representation.
        :rtype: Tensor[num, num_idx]
        """
        if posterior:
            u = torch.randn((num, self.num_idx), dtype=utils.FTYPE, device=self.sample.device)
            if self.use_mean:
                raise NotImplementedError
            return u @ self._B_dual
        h = self.draw_h(num)
        return self.k_map(h)

    ################################################################################################################

    def loss(self, representation=None) -> T:
        return torch.tensor(0., dtype=utils.FTYPE)

    def _stiefel_parameters(self, recurse=True) -> Iterator[torch.nn.Parameter]:
        super(_PPCA, self)._euclidean_parameters(recurse)
        if self._representation == 'primal':
            if self._primal_param_exists:
                yield self._primal_param
        else:
            if self._hidden_exists:
                yield self._dual_param
