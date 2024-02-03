import torch
from torch import Tensor

from ..distance.select_distance import SelectDistance
from ...utils import extend_docstring
from math import sqrt

@extend_docstring(SelectDistance)
class Silverman(SelectDistance):
    r"""
    Logistic kernel.

    .. math::
        k(x,y) = \exp\left( - \frac{d(x,y)}{\sqrt{2}\sigma}\right) \sin\left(\frac{d(x,y)}{\sqrt{2}\sigma} + \frac{\pi}{4}\right).


    """
    def __init__(self, *args, **kwargs):
        super(Silverman, self).__init__(*args, **kwargs)

    def __str__(self):
        return 'Silverman kernel'

    @property
    def _naturally_normalized(self) -> bool:
        # Silverman kernels are always naturally normalized
        return True

    @property
    def hparams_fixed(self) -> dict:
        return {'Kernel': 'Silverman',
                **super(Silverman, self).hparams_variable}

    def _implicit(self, x, y) -> Tensor:
        fact_sin = 0.25 * torch.pi
        fact_d = sqrt(.5)
        d = torch.mul(fact_d, self._dist_sigma(x, y))
        exp_d = torch.exp(-d)
        sin_d = torch.sin(fact_sin + d)
        return torch.mul(exp_d, sin_d)
