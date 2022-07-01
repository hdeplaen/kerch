import torch
from torch import Tensor as T

from ._level import _Level
from kerch import utils


class _KPCA(_Level):
    r"""
    Kernel Principal Component Analysis.
    """

    @utils.extend_docstring(_Level)
    @utils.kwargs_decorator({})
    def __init__(self, *args, **kwargs):
        super(_KPCA, self).__init__(*args, **kwargs)
        self._vals = torch.nn.Parameter(torch.empty(0, dtype=utils.FTYPE),
                                        requires_grad=False)

    @property
    def vals(self) -> T:
        return self._vals.data

    @vals.setter
    def vals(self, val):
        val = utils.castf(val, tensor=False, dev=self._vals.device)
        self._vals.data = val

    ######################################################################################

    def _solve_primal(self, target: T = None) -> None:
        if self.dim_output is None:
            self._dim_output = self.num_idx

        C = self.C
        v, w = utils.eigs(C, k=self.dim_output, psd=True)

        self.weight = w
        self.vals = v

    def _solve_dual(self, target: T = None) -> None:
        if self.dim_output is None:
            self._dim_output = self.num_idx

        K = self.K
        v, h = utils.eigs(K, k=self.dim_output, psd=True)

        self.hidden = h
        self.vals = v

    ######################################################################################

    def _primal_obj(self, x=None) -> T:
        P = self.weight @ self.weight.T  # primal projector
        R = self._I_primal - P  # reconstruction
        C = self.c(x)  # covariance
        return torch.norm(R * C)  # reconstruction error on the covariance

    def _dual_obj(self, x=None) -> T:
        P = self.hidden @ self.hidden.T  # dual projector
        R = self._I_dual - P  # reconstruction
        K = self.k(x)  # kernel matrix
        return torch.norm(R * K)  # reconstruction error on the kernel

    ######################################################################################

    def solve(self, sample=None, target=None, representation=None) -> None:
        # KPCA models don't require the target to be defined. This is verified.
        if target is not None:
            self._log.warning("The target value is discarded when fitting a KPCA model.")
        return _Level.solve(self,
                            sample=sample,
                            target=None,
                            representation=representation)
