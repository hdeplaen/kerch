import torch
from torch import Tensor

from ..distance.select_distance import SelectDistance
from ...utils import extend_docstring

@extend_docstring(SelectDistance)
class Logistic(SelectDistance):
    r"""
    Logistic kernel.

    .. math::
        k(x,y) = \frac{4}{\exp\left(d(x,y) \sigma \right) + 2 + \exp\left(-d(x,y) / \sigma \right)}.


    """
    def __init__(self, *args, **kwargs):
        super(Logistic, self).__init__(*args, **kwargs)

    def __str__(self):
        return 'Logistic kernel'

    @property
    def hparams_fixed(self) -> dict:
        return {'Kernel': 'Logistic',
                **super(Logistic, self).hparams_variable}

    def _implicit(self, x, y) -> Tensor:
        d = self._dist_sigma(x, y)
        denominator = torch.exp(d) + torch.exp(-d) + 2.
        return torch.div(4., denominator)
