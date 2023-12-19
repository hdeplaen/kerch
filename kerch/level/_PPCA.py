import torch
from torch import Tensor as T
from typing import Optional
from abc import ABCMeta, abstractmethod

from ._Level import _Level


class _PPCA(_Level, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def sigma(self) -> float:
        r"""
        Sigma value, acts as a regularization parameter.
        """
        return self._sigma

    @sigma.setter
    def sigma(self, val: float) -> None:
        self._sigma = float(val)

    @property
    def mu(self) -> torch.nn.Parameter:
        return self._mu

    @mu.setter
    def mu(self, val: torch.Tensor) -> None:
        self._mu.data = val.to(self._mu.device())

    @abstractmethod
    def _solve_primal(self) -> None:
        pass

    @abstractmethod
    def _solve_dual(self) -> None:
        pass

    ########################################################################################################################
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
            Inv_reg = self._get(key="Inv_reg_primal",
                                level="normal",
                                fun=lambda: torch.inv(self.weight.T @ self.weight + self.sigma ** 2 * self._I_primal))
            return phi @ self.weight @ Inv_reg
        if phi is None and k is not None:
            Inv_reg = self._get(key="Inv_reg_dual",
                                level='normal',
                                fun=lambda: torch.inv(
                                    self.hidden.T @ self.K @ self.hidden + self.sigma ** 2 * self._I_dual))
            return k @ self.hidden @ Inv_reg
        else:
            raise AttributeError("One and only one attribute phi or k has to be specified.")

    def phi_map(self, h: T) -> T:
        r"""
        Maximum a posteriori of phi given h.
        :param h: Latent representation.
        :type h: Tensor[N, dim_output]
        :return: MAP of phi given h.
        :rtype: Tensor[N, dim_input]
        """
        return h @ self.weight + self.mu

    def k_map(self, h: T) -> T:
        r"""
        Maximum a posteriori of k given h.
        :param h: Latent representation.
        :type h: Tensor[N, dim_output]
        :return: MAP of k given h.
        :rtype: Tensor[N, num_idx]
        """
        return h @ self.hidden @ self.K

    def draw_h(self, num: int = 1) -> T:
        r"""
        Draws a h given its prior distribution.
        :param num: Number of h to be sampled, defaults to 1.
        :type num: int, optional
        :return: Latent representation.
        :rtype: Tensor[num, dim_output]
        """
        return torch.randn((num, self.dim_output), device=self.sample.device())

    def draw_phi(self, num: int = 1) -> T:
        r"""
        Draws a primal representation phi given its posterior distribution.
        :param num: Number of phi to be sampled, defaults to 1.
        :type num: int, optional
        :return: Primal representation.
        :rtype: Tensor[num, dim_input]
        """
        h = self.draw_h(num)
        return self.phi_map(h)

    def draw_k(self, num: int = 1) -> T:
        r"""
        Draws a dual representation k given its posterior distribution.
        :param num: Number of k to be sampled, defaults to 1.
        :type num: int, optional
        :return: Dual representation.
        :rtype: Tensor[num, num_idx]
        """
        h = self.draw_h(num)
        return self.k_map(h)

    ################################################################################################################

    def _euclidean_parameters(self, recurse=True):
        super(_PPCA, self)._euclidean_parameters(recurse)
        if self._representation == 'primal':
            if self._weight_exists:
                yield self._weight
        else:
            if self._hidden_exists:
                yield self._hidden
