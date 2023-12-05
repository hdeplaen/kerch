"""
File containing the Wasserstein Exponential Kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch

from .. import utils
from ._exponential import _Exponential


@utils.extend_docstring(_Exponential)
class Wasserstein(_Exponential):
    r"""
    Wasserstein Exponential Kernel.

    .. math::
        k(x,y) = \exp\left( -\frac{\mathcal{W}^2(x,y)}{2\sigma^2} \right),

    with the Wasserstein distance given by

    .. math::
        \mathcal{W}^2(x,y) = .
    """

    @utils.kwargs_decorator({'cost':'euclidean'})
    def __init__(self, **kwargs):
        super(Wasserstein, self).__init__(**kwargs)
        self.cost = kwargs['cost']

    def __str__(self):
        if self._sigma is None:
            return f"Wasserstein Exponential Kernel (sigma undefined)"
        return f"Wasserstein Exponential Kernel (sigma: {str(self.sigma)})"

    @property
    def hparams(self):
        return {"Kernel": "Wasserstein Exponential Kernel", **super(Wasserstein, self).hparams}

    @property
    def cost(self) -> torch.Tensor:
        return self._cost

    @cost.setter
    def cost(self, val: torch.Tensor) -> None:
        self._cost = val

    def _dist(self, x, y):
        # TODO
        raise NotImplementedError
