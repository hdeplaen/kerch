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
        {"encoder": None,
         "decoder": None,
         "kernels_trainable": False})
    def __init__(self, **kwargs):
        """
        :param encoder: torch.nn.Module explicit kernel
        :param kernels_trainable: True if support vectors / kernel are trainable (default False)
        """
        super(explicit_nn, self).__init__(**kwargs)

        self._encoder: torch.nn.Module = kwargs["encoder"]
        assert self._encoder is not None, "Encoder module must be specified."
        assert isinstance(self._encoder, torch.nn.Module), "Encoder must be an instance of torch.nn.Module."

        self._decoder: torch.nn.Module = kwargs["decoder"]
        assert isinstance(self._decoder, torch.nn.Module) or self._decoder is None, "If specified, the decoder must " \
                                                                                    "be an instance of torch.nn.Module."

    def __str__(self):
        return "explicit kernel"

    def hparams(self):
        return {"Kernel": "Explicit", **super(explicit_nn, self).hparams}

    @property
    def encoder(self) -> torch.nn.Module:
        return self._encoder

    @property
    def decoder(self) -> torch.nn.Module:
        return self._decoder

    def _explicit(self, x=None):
        x = super(explicit_nn, self)._explicit(x)
        return self._encoder(x)

    def _euclidean_parameters(self, recurse=True):
        yield from self._encoder.parameters()
        if self._decoder is not None:
            yield self._decoder.parameters()
        super(explicit_nn, self)._euclidean_parameters(recurse)

    def phi_pinv(self, phi=None, centered=None, normalized=None) -> torch.Tensor:
        if self._decoder is None:
            self._log.error("No decoder provided for pseudo-inversion of a neural-network based "
                            "explicit feature map.")
            raise Exception
        phi = super(explicit_nn, self).phi_pinv(phi, centered, normalized)
        return self._decoder(phi)
