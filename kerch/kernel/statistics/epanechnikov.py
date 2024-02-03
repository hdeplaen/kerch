import torch
from torch import Tensor

from ..distance.select import Select
from ...utils import extend_docstring

@extend_docstring(Select)
class Epanechnikov(Select):
    r"""
    Epanechnikov (parabolic) kernel.

    .. math::
        k(x,y) = %
        \begin{cases}
        1 - \left(\frac{d(x,y)}{\sigma}\right)^2 & \text{for } \frac{d(x,y)}{\sigma} \leq 1, \\
        0 & \text{otherwise}.
        \end{cases}


    """
    def __init__(self, *args, **kwargs):
        super(Epanechnikov, self).__init__(*args, **kwargs)

    def __str__(self):
        return 'Epanechnikov/parabolic kernel'

    @property
    def _naturally_normalized(self) -> bool:
        # Epanechnikov kernels are always naturally normalized
        return True

    @property
    def hparams_fixed(self) -> dict:
        return {'Kernel': 'Epanechnikov',
                **super(Epanechnikov, self).hparams_variable}

    def _implicit(self, x, y) -> Tensor:
        d = self._dist_sigma(x, y)
        return (d <= 1) * (1 - d ** 2)
