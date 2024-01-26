# coding=utf-8
"""
File containing the RBF kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch

from .. import utils
from .exponential import Exponential


@utils.extend_docstring(Exponential)
class Laplacian(Exponential):
    r"""
    Laplacian kernel.

    .. math::
        k(x,y) = \exp\left( -\frac{\lVert x-y \rVert_2}{2\sigma^2} \right).


    .. note::
        The difference with the RBF kernel is in the squaring or not of the distance inside the exponential.

    """

    def __init__(self, *args, **kwargs):
        super(Laplacian, self).__init__(*args, **kwargs)

    def __str__(self):
        if self._sigma_defined:
            return f"Laplacian kernel (sigma: {str(self.sigma)})"
        return f"Laplacian kernel (sigma undefined)"

    @property
    def hparams_fixed(self):
        return {"Kernel": "Laplacian", **super(Laplacian, self).hparams_fixed}

    def _dist(self, x, y):
        x = x.T[:, :, None]
        y = y.T[:, None, :]

        diff = x - y
        D = torch.sum(diff * diff, dim=0, keepdim=False)
        return torch.sqrt(D)
