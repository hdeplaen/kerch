import torch
from torch import Tensor as T

from .level import Level
from ._kpca import _KPCA
from kerch import utils


class KPCA(_KPCA, Level):
    r"""
    Kernel Principal Component Analysis.
    """

    @utils.extend_docstring(_KPCA)
    @utils.extend_docstring(Level)
    @utils.kwargs_decorator({})
    def __init__(self, *args, **kwargs):
        super(KPCA, self).__init__(*args, **kwargs)

    def __str__(self):
        return "KPCA with " + Level.__str__(self)

    def reconstruct(self, x=None, representation=None):
        representation = utils.check_representation(representation, self._representation, self)
        if representation == 'primal':
            phi = self.phi(x)
            U = self.weight
            R = U @ U.T
            return phi @ R
        else:
            K = self.k(x)
            H = self.hidden
            R = H @ H.T
            return K @ R
