"""
File containing the linear kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import rkm
import rkm.model as mdl

class LinearKernel(mdl.kernel.Kernel):
    """
    Linear kernel class
    k(x,y) = < x,y >.
    """

    @rkm.kwargs_decorator({})
    def __init__(self, **kwargs):
        """
        no specific parameters to the linear kernel
        """
        super(LinearKernel, self).__init__(**kwargs)

    def __str__(self):
        return "linear kernel"

    def hparams(self):
        return {"Kernel": "Linear", **super(LinearKernel, self).hparams}

    def _implicit(self, x):
        return x @ self.kernels[self._idx_kernels,:].t()

    def _explicit(self, x):
        return x
