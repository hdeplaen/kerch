"""
File containing the RBF kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch

from .. import utils
from ._exponential import _Exponential


@utils.extend_docstring(_Exponential)
class RBF(_Exponential):
    r"""
    RBF kernel (radial basis function).

    .. math::
        k1(x,y) = \exp\left( -\frac{\lVert x-y \rVert_2^2}{2\sigma^2} \right).

    """

    def __init__(self, **kwargs):
        super(RBF, self).__init__(**kwargs)

    def __str__(self):
        if self._sigma is None:
            return f"RBF kernel (sigma undefined)"
        return f"RBF kernel (sigma: {str(self.sigma)})"

    @property
    def hparams(self):
        return {"Kernel": "RBF", **super(RBF, self).hparams}

    def _dist(self, x, y):
        x = x.T[:, :, None]
        y = y.T[:, None, :]

        diff = x - y
        return torch.sum(diff * diff, dim=0, keepdim=False)
