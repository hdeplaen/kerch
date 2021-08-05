"""
File containing the explicit kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import rkm
import rkm.model as mdl

class ExplicitKernel(mdl.kernel.Kernel):
    """
    Explicit kernel class, parametrized by a neural network.
    k(x,y) = < NN(x), NN(y) >.
    """

    @rkm.kwargs_decorator(
        {"network": None, "kernels_trainable": False})
    def __init__(self, **kwargs):
        """
        :param network: torch.nn.Module explicit kernel
        :param kernels_trainable: True if support vectors / kernel are trainable (default False)
        """
        super(ExplicitKernel, self).__init__(**kwargs)
        self._network = kwargs["network"]
        assert self._network is not None, "Network module must be specified."

    def __str__(self):
        return "explicit kernel"

    def hparams(self):
        return {"Kernel": "Explicit", **super(ExplicitKernel, self).hparams}

    def implicit(self, x):
        x = self.explicit(x)
        y = self.explicit(self.kernels(self._idx_kernels))
        return x @ y.t()

    def explicit(self, x, idx_kernels=None):
        return self._network(x)
