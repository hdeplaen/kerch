import torch
from torch import Tensor as T

from .level import level
from kerpy import utils


class kpca(level):
    r"""
    Kernel Principal Component Analaysis.
    """

    @utils.extend_docstring(level)
    @utils.kwargs_decorator({})
    def __init__(self, **kwargs):
        super(kpca, self).__init__(**kwargs)

    def _solve_primal(self, sample: T, target: T) -> None:
        pass

    def _solve_dual(self, sample: T, target: T) -> None:
        pass


    def solve(self, sample=None, target=None, representation=None) -> None:
        # KPCA models don't require the target to be defined. This is verified.
        if target is not None:
            self._log.warning("The target value is discarded when fitting a KPCA model.")
        return super(kpca, self).solve(sample=sample,
                                        target=None,
                                        representation=representation)