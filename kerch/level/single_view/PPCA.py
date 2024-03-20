# coding=utf-8
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
    def _solve_primal(self) -> None:
        C = self.C

        if self.dim_output is None:
            self._dim_output = self.dim_feature
        elif self.dim_output > self.dim_feature:
            self._logger.warning(f"In primal, the output dimension {self.dim_output} (the number of "
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
            self._logger.warning(f"In dual, the output dimension {self.dim_output} (the number of "
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

    @property
    def H(self) -> T:
        return self.dual_param

    @property
    def W(self) -> T:
        return self.primal_param @ torch.diag(self.sqrt_vals)
