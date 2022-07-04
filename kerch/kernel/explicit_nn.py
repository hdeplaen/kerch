"""
File containing the explicit kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

from .. import utils
from .explicit import explicit, base
import torch



@utils.extend_docstring(base)
class explicit_nn(explicit):
    r"""
    Implicit kernel class, parametrized by a neural network.

    .. math::
        k(x,y) = NN\left(x\right)^\top NN\left(y\right).


    In other words, we have

    .. math::
        \phi(x) = NN\left(x\right)

    :param network: Network to be used.
    :type network: torch.nn.Module
    """

    @utils.kwargs_decorator(
        {"network": None, "kernels_trainable": False})
    def __init__(self, **kwargs):
        """
        :param network: torch.nn.Module explicit kernel
        :param kernels_trainable: True if support vectors / kernel are trainable (default False)
        """
        super(explicit_nn, self).__init__(**kwargs)
        self._network: torch.nn.Module = kwargs["network"]
        assert self._network is not None, "Network module must be specified."

    def __str__(self):
        return "explicit kernel"

    def hparams(self):
        return {"Kernel": "Explicit", **super(explicit_nn, self).hparams}

    def _explicit(self, x=None):
        x = super(explicit_nn, self)._explicit(x)
        return self._network(x)

    def _euclidean_parameters(self, recurse=True):
        yield from self._network.parameters()
        super(explicit_nn, self)._euclidean_parameters(recurse)
