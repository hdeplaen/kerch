# coding=utf-8
"""
File containing the Wasserstein Exponential Kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch

from ... import utils
from kerch.kernel.statistics.exponential import Exponential


@utils.extend_docstring(Exponential)
class Wasserstein(Exponential):
    r"""
    Wasserstein Exponential Kernel.

    .. math::
        k(x,y) = \exp\left( -\frac{\mathcal{W}^2(x,y)}{2\sigma^2} \right),

    with the Wasserstein distance given by

    .. math::
        \mathcal{W}^2(x,y) = .
    """

    @utils.kwargs_decorator({'cost': 'euclidean',
                             'reg': 1e-1})
    def __init__(self, *args, **kwargs):
        super(Wasserstein, self).__init__(*args, **kwargs)
        self.cost = kwargs['cost']
        self.reg = kwargs['reg']

    def __str__(self):
        if self._sigma is None:
            return f"Wasserstein Exponential Kernel (sigma undefined)"
        return f"Wasserstein Exponential Kernel (sigma: {str(self.sigma)})"

    @property
    def hparams_fixed(self):
        return {"Kernel": "Wasserstein Exponential Kernel", **super(Wasserstein, self).hparams_fixed}

    @property
    def cost(self) -> str:
        return self._cost

    @cost.setter
    def cost(self, val: str) -> None:
        self._cost = val

    @property
    def reg(self) -> float:
        return self._reg

    @reg.setter
    def reg(self, val: float) -> None:
        self._reg = float(val)

    def _square_dist(self, x, y) -> torch.Tensor:
        return self._get("W_dist", level_key="Wasserstein_kernel_dist", fun=lambda: self._wass_dist(x,y))

    def _wass_dist(self, x, y) -> torch.Tensor:
        raise NotImplementedError
