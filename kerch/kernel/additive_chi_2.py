# coding=utf-8
"""
File containing the RBF kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: May 2022
"""

import torch

from .. import utils
from .implicit import Implicit, Kernel


@utils.extend_docstring(Kernel)
class AdditiveChi2(Implicit):
    r"""
    Additive Chi Squared kernel. Often used in computer vision.

    .. math::
        k(x,y) = \sum_i \frac{2x_i y_i}{x_i + y_i}.

    """

    def __init__(self, *args, **kwargs):
        super(AdditiveChi2, self).__init__(*args, **kwargs)

    def __str__(self):
        return f"Additive Chi Squared kernel."

    @property
    def hparams_variable(self):
        return {}

    @property
    def hparams_fixed(self):
        return {"Kernel": "Additive Chi Squared", **super(AdditiveChi2, self).hparams_fixed}

    def _implicit(self, x, y):
        x = x.T[:, :, None]
        y = y.T[:, None, :]

        prod = x * y
        sum = torch.clamp(x + y, min=utils.EPS)
        output = torch.sum(2 * prod / sum, dim=0, keepdim=True)

        return output.squeeze(0)
