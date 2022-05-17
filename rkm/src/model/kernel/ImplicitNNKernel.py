"""
File containing the implicit kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import rkm.src
import rkm.src.model.kernel.ImplicitKernel as ImplicitKernel

class ImplicitNNKernel(ImplicitKernel.ImplicitKernel):
    """
    Implicit kernel class, parametrized by a neural network.
    k(x,y) = NN( [x, y] ).
    """

    @rkm.src.kwargs_decorator(
        {"network": None})
    def __init__(self, **kwargs):
        """
        :param network: torch.nn.Module explicit kernel
        """
        super(ImplicitNNKernel, self).__init__(**kwargs)
        self._network = kwargs["network"]
        assert self._network is not None, "Network module must be specified."

    def __str__(self):
        return "implicit kernel"

    @property
    def hparams(self):
        return {"Kernel": "Implicit", **super(ImplicitNNKernel, self).hparams}

    def _implicit(self, x_oos=None, x_sample=None):
        raise NotImplementedError

        # x_oos, x_sample = super(ImplicitKernel, self)._implicit(x_oos, x_sample)
        # return self._network(x_oos, x_sample)
