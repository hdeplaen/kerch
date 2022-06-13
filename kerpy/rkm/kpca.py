import torch
from torch import Tensor as T

from .level import level
from kerpy import utils


class kpca(level):
    r"""
    Kernel Principal Component Analysis.
    """

    @utils.extend_docstring(level)
    @utils.kwargs_decorator({})
    def __init__(self, **kwargs):
        super(kpca, self).__init__(**kwargs)

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
        return super(kpca, self).solve(sample=sample,
                                        target=None,
                                        representation=representation)