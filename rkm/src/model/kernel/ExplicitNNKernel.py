"""
File containing the explicit kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import rkm.src
import rkm.src.model.kernel.ExplicitKernel as ExplicitKernel

class ExplicitNNKernel(ExplicitKernel.ExplicitKernel):
    """
    Explicit kernel class, parametrized by a neural network.
    k(x,y) = < NN(x), NN(y) >.
    """

    @rkm.src.kwargs_decorator(
        {"network": None, "kernels_trainable": False})
    def __init__(self, **kwargs):
        """
        :param network: torch.nn.Module explicit kernel
        :param kernels_trainable: True if support vectors / kernel are trainable (default False)
        """
        super(ExplicitNNKernel, self).__init__(**kwargs)
        self._network = kwargs["network"]
        assert self._network is not None, "Network module must be specified."

    def __str__(self):
        return "explicit kernel"

    def hparams(self):
        return {"Kernel": "Explicit", **super(ExplicitNNKernel, self).hparams}

    def _explicit(self, x=None):
        x = super(ExplicitNNKernel, self)._explicit(x)
        return self._network(x)
