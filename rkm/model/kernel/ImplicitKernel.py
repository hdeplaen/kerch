"""
File containing the implicit kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import rkm
import rkm.model as mdl

class ImplicitKernel(mdl.kernel.Kernel):
    """
    Implicit kernel class, parametrized by a neural network.
    k(x,y) = NN( [x, y] ).
    """

    @rkm.kwargs_decorator(
        {"network": None})
    def __init__(self, **kwargs):
        """
        :param network: torch.nn.Module explicit kernel
        """
        super(ImplicitKernel, self).__init__(**kwargs)
        self._network = kwargs["network"]
        assert self._network is not None, "Network module must be specified."

    def __str__(self):
        return "implicit kernel"

    @property
    def hparams(self):
        return {"Kernel": "Implicit", **super(ImplicitKernel, self).hparams}

    def implicit(self, x):
        return self._network(x, self.kernels(self._idx_kernels))

    def explicit(self, x, idx_kernels=None):
        raise mdl.PrimalError
