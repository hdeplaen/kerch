"""
File containing the linear kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import rkm.src
import rkm.src.model.kernel.ExplicitKernel as ExplicitKernel


class LinearKernel(ExplicitKernel.ExplicitKernel):
    """
    Linear kernel class
    k(x,y) = < x,y >.
    """

    @rkm.src.kwargs_decorator({})
    def __init__(self, **kwargs):
        """
        no specific parameters to the linear kernel
        """
        super(LinearKernel, self).__init__(**kwargs)

    def __str__(self):
        return "linear kernel"

    def hparams(self):
        return {"Kernel": "Linear", **super(LinearKernel, self).hparams}