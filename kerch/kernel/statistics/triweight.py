import torch
from torch import Tensor

from ..distance.select_distance import SelectDistance
from ...utils import extend_docstring

@extend_docstring(SelectDistance)
class Triweight(SelectDistance):
    r"""
    Triweight kernel.

    .. math::
        k(x,y) = %
        \begin{cases}
        \left(1 - \left(\frac{d(x,y)}{\sigma}\right)^2\right)^3 & \text{for } \frac{d(x,y)}{\sigma} \leq 1, \\
        0 & \text{otherwise}.
        \end{cases}


    """
    def __init__(self, *args, **kwargs):
        super(Triweight, self).__init__(*args, **kwargs)

    def __str__(self):
        return 'triweight kernel'

    @property
    def _naturally_normalized(self) -> bool:
        # Triweight kernels are always naturally normalized
        return True

    @property
    def hparams_fixed(self) -> dict:
        return {'Kernel': 'Triweight',
                **super(Triweight, self).hparams_variable}

    def _implicit(self, x, y) -> Tensor:
        d = self._dist_sigma(x, y)
        return (d <= 1) * (1 - d ** 2) ** 3
