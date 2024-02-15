# coding=utf-8
import torch
from torch import Tensor as T

from .Level import Level
from .._KPCA import _KPCA
from ...utils import check_representation, extend_docstring, kwargs_decorator


class KPCA(_KPCA, Level):
    r"""
    Kernel Principal Component Analysis.
    """

    @extend_docstring(_KPCA)
    @extend_docstring(Level)
    @kwargs_decorator({})
    def __init__(self, *args, **kwargs):
        super(KPCA, self).__init__(*args, **kwargs)
        if not self.centered:
            self._logger.warning("The used feature map/kernel is not centered. "
                                 "This may work, but contradicts with the classical definition of KPCA.")


    def __str__(self):
        return "KPCA with " + Level.__str__(self)

    @property
    @torch.no_grad()
    def K_reconstructed(self) -> T:
        H = self.hidden
        D = torch.diag(self.vals)
        return H @ D @ H.T

    @property
    @torch.no_grad()
    def C_reconstructed(self) -> T:
        U = self.weight
        D = torch.diag(self.vals)
        return U @ D @ U.T


    def reconstruct(self, x=None, representation=None):
        representation = check_representation(representation, self._representation, self)
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

    def _update_dual_from_primal(self):
        self.hidden = self._forward(representation='primal') @ torch.diag(1 / self.vals)
