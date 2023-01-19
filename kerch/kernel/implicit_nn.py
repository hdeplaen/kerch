"""
File containing the implicit kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

from .. import utils
from ._implicit import _Implicit, _Projected
import torch


@utils.extend_docstring(_Projected)
class ImplicitNN(_Implicit):
    r"""
    _Implicit kernel class, parametrized by a neural network.

    .. math::
        k(x,y) = NN\left( [x, y] \right).


    .. warning::
        This kernel is not positive semi-definite in the general case. This is only possible if a specific choice of
        neural network is provided.


    :param network: Network to be used.
    :type network: torch.nn.Module
    """

    @utils.kwargs_decorator(
        {"network": None})
    def __init__(self, **kwargs):
        """
        :param network: torch.nn.Module explicit kernel
        """
        super(ImplicitNN, self).__init__(**kwargs)
        self._network: torch.nn.Module = kwargs["network"]
        assert isinstance(self._network, torch.nn.Module), "Torch network module must be specified."

    def __str__(self):
        return "implicit kernel"

    @property
    def hparams(self):
        return {"Kernel": "_Implicit", **super(ImplicitNN, self).hparams}

    def _implicit(self, x, y):
        raise NotImplementedError

        # x, y = super(ImplicitKernel, self)._implicit(x, y)
        # return self._encoder(x, y)

    def _euclidean_parameters(self, recurse=True):
        yield from self._network.parameters()
        yield from super(ImplicitNN, self)._euclidean_parameters(recurse)
