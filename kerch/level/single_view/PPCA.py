from typing import Optional
import torch
from torch import Tensor as T

from .._PPCA import _PPCA
from .Level import Level
from ... import utils

class PPCA(_PPCA, Level):
    def __init__(self, *args, **kwargs):
        super(PPCA, self).__init__(*args, **kwargs)
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
            return (phi - self.mu) @ self.weight @ torch.diag(1 / self.vals) * self.num_idx
        if phi is None and k is not None:
            return k @ self.hidden @ torch.diag(1 / self.vals) * self.num_idx
        else:
            raise AttributeError("One and only one attribute phi or k has to be specified.")

##############################################################################################################
    @torch.no_grad()
    def _solve_primal(self) -> None:
        C = self.C

        if self.dim_output is None:
            self._dim_output = self.dim_feature
        elif self.dim_output > self.dim_feature:
            self._log.warning(f"In primal, the output dimension {self.dim_output} (the number of "
                              f"eigenvectors) must not exceed the feature dimension {self.dim_feature} (the dimension "
                              f"of the correlation matrix to be diagonalized). As this is the case here, the output "
                              f"dimension is reduced to {self.dim_feature}.")
            self.dim_output = self.dim_feature

        v, w = utils.eigs(C, k=self.dim_output, psd=True)

        if self.dim_output == self.num_idx:
            self.sigma = 0.
        else:
            self.sigma = (torch.trace(C) - torch.sum(v)) / (self.num_idx * (self.num_idx - self.dim_output))
        self.vals = v
        self.weight = w @ torch.diag(torch.sqrt(v / self.num_idx - self.sigma ** 2))

    @torch.no_grad()
    def _solve_dual(self) -> None:
        K = self.K

        if self.dim_output is None:
            self._dim_output = self.num_idx
        elif self.dim_output > self.num_idx:
            self._log.warning(f"In dual, the output dimension {self.dim_output} (the number of "
                              f"eigenvectors) must not exceed the number of samples {self.num_idx} (the dimension "
                              f"of the kernel matrix to be diagonalized). As this is the case here, the output "
                              f"dimension is reduced to {self.num_idx}.")
            self.dim_output = self.num_idx

        v, h = utils.eigs(K, k=self.dim_output, psd=True)

        if self.dim_output == self.num_idx:
            self.sigma = 0.
        else:
            self.sigma = (torch.trace(K) - torch.sum(v)) / (self.num_idx * (self.num_idx - self.dim_output))
        self.vals = v
        self.hidden = h @ torch.diag(torch.sqrt(1 / self.num_idx - self.sigma ** 2 / v))
        pass
