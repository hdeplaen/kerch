import torch
from torch import Tensor

from ..distance.select_distance import SelectDistance
from ...utils import extend_docstring

@extend_docstring(SelectDistance)
class Uniform(SelectDistance):
    r"""
    Uniform (window) kernel.

    .. math::
        k(x,y) = %
        \begin{cases}
        1 & \text{for } \frac{d(x,y)}{\sigma} \leq 1, \\
        0 & \text{otherwise}.
        \end{cases}


    """
    def __init__(self, *args, **kwargs):
        super(Uniform, self).__init__(*args, **kwargs)

    def __str__(self):
        return 'uniform kernel'

    @property
    def _naturally_normalized(self) -> bool:
        # Uniform kernels are always naturally normalized
        return True

    @property
    def hparams_fixed(self) -> dict:
        return {'Kernel': 'Uniform',
                **super(Uniform, self).hparams_variable}

    def _implicit(self, x, y) -> Tensor:
        d = self._dist_sigma(x, y)
        return (d <= 1).type(d.dtype)
