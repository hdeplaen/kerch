# coding=utf-8
"""
File containing the explicit kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

from typing import Iterator
from .. import utils
from .explicit import Explicit, Kernel
import torch
import lazy_loader


@utils.extend_docstring(Kernel)
class ExplicitNN(Explicit):
    r"""
    Explicit kernel class, parametrized by a neural network.

    .. math::
        k(x,y) = NN\left(x\right)^\top NN\left(y\right).


    In other words, we have

    .. math::
        \phi(x) = NN\left(x\right)

    :param network: Network to be used.
    :type network: torch.nn.Module
    """

    @utils.kwargs_decorator(
        {"kernels_trainable": False})
    def __init__(self, *args, **kwargs):
        """
        :param encoder: torch.nn.Module explicit kernel
        :param decoder: torch.nn.Module explicit kernel pseudo-inverse
        :param kernels_trainable: True if support vectors / kernel are trainable (default False)
        :param recon_loss_fun: Instance of the reconstruction loss function for the encoder/decoder pair.
            Defaults to torch.nn.MSELoss(reduction='mean').
        """
        self._encoder = None
        self._decoder = None

        super(ExplicitNN, self).__init__(*args, **kwargs)

        self._encoder = kwargs.pop('encoder', None)
        assert self._encoder is not None, "The argument encoder must be specified."
        assert isinstance(self._encoder, torch.nn.Module), "Encoder must be an instance of torch.nn.Module."

        self._decoder = kwargs.pop('decoder', None)
        assert isinstance(self._decoder, torch.nn.Module) or self._decoder is None, "If specified, the decoder must " \
                                                                                    "be an instance of torch.nn.Module."

        self._recon_loss_func = kwargs.pop('recon_loss_fun', torch.nn.MSELoss()) # reduction='mean' is the default

    def __str__(self):
        if self._encoder is not None:
            encoder = f"encoder: {self.encoder.__class__.__name__}"
        else:
            encoder = 'encoder undefined'
        if self._decoder is not None:
            decoder = f"decoder: {self.decoder.__class__.__name__}"
        else:
            decoder = "decoder undefined"
        return f"explicit kernel ({encoder}, {decoder})"

    def hparams(self):
        return {"Kernel": "Explicit Neural Network Based", **super(ExplicitNN, self).hparams}

    @property
    def encoder(self) -> torch.nn.Module:
        return self._encoder

    @property
    def decoder(self) -> torch.nn.Module:
        if self._decoder is None:
            raise utils.NotInitializedError(cls=self, message="No decoder provided. This is necessary for the "
                                                              "pseudo-inversion of a neural-network based explicit "
                                                              "feature map.")
        return self._decoder

    def decode(self, x=None) -> torch.Tensor:
        decoded = self.decoder(self(x))
        return self.sample_transform.revert(decoded)

    def _explicit(self, x):
        return self._encoder(x)

    def loss(self) -> float:
        recon = self.decode()
        return self._recon_loss_func(self.current_sample, recon)

    def _euclidean_parameters(self, recurse=True) -> Iterator[torch.nn.Parameter]:
        yield from self._encoder.parameters()
        if self._decoder is not None:
            yield from self._decoder.parameters()
        super(ExplicitNN, self)._euclidean_parameters(recurse)

    def _explicit_preimage(self, phi) -> torch.Tensor:
        View = lazy_loader.load('..level.single_view.View', error_on_import=True)
        if not isinstance(self, View.View):
            return self.decoder(phi)
        else:
            raise utils.KerchError('The decoder is not a pre-image of the explicit representation, but of the model '
                                   'image itself. Hence the pre-image cannot be computed directly on an explicit '
                                   'representation. You may directly access the decoder member of this instance and '
                                   'compute it yourself if you nevertheless wish to perform this operation.')

