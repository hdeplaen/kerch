# coding=utf-8
"""
File containing the RBF kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch

from ... import utils
from ..distance.select import Select


@utils.extend_docstring(Select)
class Exponential(Select):
    r"""
    :param squared: Boolean indicating whether the norm in the exponential is squared. Defaults to ``True``.
    :type squared: bool
    """

    def __init__(self, *args, **kwargs):
        super(Exponential, self).__init__(*args, **kwargs)
        self._squared = kwargs.pop('squared', True)

    def __str__(self):
        return f"exponential kernel {Select.__str__(self)}"

    @property
    def hparams_fixed(self) -> dict:
        return {"Kernel": "Exponential",
                "Squared exp. distance": self._squared,
                **super(Exponential, self).hparams_fixed}

    @property
    def squared(self) -> bool:
        r"""
        Boolean indicating whether the norm in the exponential is squared.
        """
        return self._squared

    @property
    def _naturally_normalized(self) -> bool:
        # _Exponential kernels are always naturally normalized
        return True

    def _implicit(self, x, y):
        fact = -.5
        if self._squared:
            d = self._square_dist_sigma(x, y)
        else:
            d = self._dist_sigma(x, y)
        return torch.exp(torch.mul(d, fact))

    def _implicit_self(self, x=None):
        if x is None:
            x = self.current_sample_projected
        return torch.ones(x.shape[0], dtype=utils.FTYPE, device=x.device)
