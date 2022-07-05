"""
File containing the linear kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch

from .. import utils
from .explicit import explicit, base


@utils.extend_docstring(base)
class linear(explicit):
    r"""
    Linear kernel.

    .. math::
        k(x,y) = x^\top x.

    To this kernel also corresponds the explicit finite dimensional feature map :math:`\phi(x)=x`.
    """

    @utils.kwargs_decorator({})
    def __init__(self, **kwargs):
        super(linear, self).__init__(**kwargs)
        if self.normalize == True:
            self._log.info("A normalized linear kernel also corresponds to a cosine kernel.")

    def __str__(self):
        return "linear kernel"

    @property
    def dim_feature(self) -> int:
        return self.dim_input

    def hparams(self):
        return {"Kernel": "Linear", **super(linear, self).hparams}

    def _explicit(self, x=None):
        return super(linear, self)._explicit(x)
