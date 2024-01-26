# coding=utf-8
"""
File containing the linear kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch

from .. import utils
from .explicit import Explicit


@utils.extend_docstring(Explicit)
class Linear(Explicit):
    r"""
    Linear kernel.

    .. math::
        k(x,y) = x^\top y.

    To this kernel also corresponds the explicit finite dimensional feature map :math:`\phi(x)=x`.
    """

    @utils.kwargs_decorator({})
    def __init__(self, *args, **kwargs):
        super(Linear, self).__init__(*args, **kwargs)
        if self.normalized:
            self._logger.info("A normalized linear kernel also corresponds to a cosine kernel.")

    def __str__(self):
        return "linear kernel"

    @property
    def dim_feature(self) -> int:
        return self.dim_input

    def hparams_fixed(self):
        return {"Kernel": "Linear", **super(Linear, self).hparams_fixed}

    def _explicit(self, x):
        return super(Linear, self)._explicit(x)

    def _explicit_preimage(self, phi) -> torch.Tensor:
        return phi
