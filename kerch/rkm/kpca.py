import torch
from torch import Tensor as T

from .level import Level
from kerch import utils


class KPCA(Level):
    r"""
    Kernel Principal Component Analysis.
    """

    @utils.extend_docstring(Level)
    @utils.kwargs_decorator({})
    def __init__(self, **kwargs):
        super(KPCA, self).__init__(**kwargs)
        self._vals = torch.nn.Parameter(torch.empty(0, dtype=utils.FTYPE),
                                        requires_grad=False)

    @property
    def vals(self) -> T:
        return self._vals.data

    @vals.setter
    def vals(self, val):
        val = utils.castf(val, tensor=False, dev=self._vals.device)
        self._vals.data = val

    def _solve_primal(self,target: T=None) -> None:
        pass

    def _solve_dual(self,target: T=None) -> None:
        if self.dim_output is None:
            self._dim_output = self.num_idx

        K = self.kernel.K
        v, h = utils.eigs(K, k=self.dim_output, psd=True)

        self.hidden = h
        self.vals = v

    def solve(self, sample=None, target=None, representation=None) -> None:
        # KPCA models don't require the target to be defined. This is verified.
        if target is not None:
            self._log.warning("The target value is discarded when fitting a KPCA model.")
        return super(KPCA, self).solve(sample=sample,
                                       target=None,
                                       representation=representation)

    def interpolate(self):
        pass