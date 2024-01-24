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
class RBF(Exponential):
    r"""
    RBF kernel (radial basis function).

    .. math::
        k(x,y) = \exp\left( -\frac{\lVert x-y \rVert_2^2}{2\sigma^2} \right).

    """

    def __init__(self, *args, **kwargs):
        super(RBF, self).__init__(*args, **kwargs)

    def __str__(self):
        if self._sigma_defined:
            return f"RBF kernel (sigma: {str(self.sigma)})"
        return f"RBF kernel (sigma undefined)"


    @property
    def hparams(self):
        return {"Kernel": "RBF", **super(RBF, self).hparams}

    def _dist(self, x, y):
        x = x.T[:, :, None]
        y = y.T[:, None, :]

        diff = x - y
        return torch.sum(diff * diff, dim=0, keepdim=False)


