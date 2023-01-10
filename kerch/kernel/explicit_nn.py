"""
File containing the explicit kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

from .. import utils
from ._explicit import _Explicit, _Statistics
import torch


@utils.extend_docstring(_Statistics)
class ExplicitNN(_Explicit):
    r"""
    _Implicit kernel class, parametrized by a neural network.

    .. math::
        k1(x,y) = NN\left(x\right)^\top NN\left(y\right).


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
        super(ExplicitNN, self).__init__(**kwargs)

        self._encoder: torch.nn.Module = kwargs["encoder"]
        assert self._encoder is not None, "Encoder module must be specified."
        assert isinstance(self._encoder, torch.nn.Module), "Encoder must be an instance of torch.nn.Module."

        self._decoder: torch.nn.Module = kwargs["decoder"]
        assert isinstance(self._decoder, torch.nn.Module) or self._decoder is None, "If specified, the decoder must " \
                                                                                    "be an instance of torch.nn.Module."

        self._nn_loss_func = kwargs["nn_loss_func"]

    def __str__(self):
        return "explicit kernel"

    def hparams(self):
        return {"Kernel": "_Explicit", **super(ExplicitNN, self).hparams}

    @property
    def encoder(self) -> torch.nn.Module:
        return self._encoder

    @property
    def decoder(self) -> torch.nn.Module:
        return self._decoder

    def _explicit(self, x=None):
        x = super(ExplicitNN, self)._explicit(x)
        return self._encoder(x)

    def _euclidean_parameters(self, recurse=True):
        yield from self._encoder.parameters()
        if self._decoder is not None:
            yield self._decoder.parameters()
        super(ExplicitNN, self)._euclidean_parameters(recurse)

    def _explicit_preimage(self, phi) -> torch.Tensor:
        if self._decoder is None:
            self._log.error("No decoder provided for pseudo-inversion of a neural-network based "
                            "explicit feature map.")
            raise Exception
        return self._decoder(phi)

