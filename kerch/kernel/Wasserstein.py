"""
File containing the Wasserstein Exponential Kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch

from .. import utils
from ._Exponential import _Exponential


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
    def hparams(self):
        return {"Kernel": "Wasserstein Exponential Kernel", **super(Wasserstein, self).hparams}

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

    def _dist(self, x, y) -> torch.Tensor:
        return self._get("W_dist", "lightweight", lambda: self._wass_dist(x,y))

    def _wass_dist(self, x, y) -> torch.Tensor:
        raise NotImplementedError
