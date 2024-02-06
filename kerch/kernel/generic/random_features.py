# coding=utf-8
"""
File containing the RFF kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: May 2022
"""

import torch
from math import sqrt
from typing import Union

from ... import utils
from ..explicit import Explicit, Kernel


@utils.extend_docstring(Kernel)
class RandomFeatures(Explicit):
    r"""
    Random Features kernel.

    .. math::
        \phi(x) = \frac{1}{\sqrt{d}} \left(\begin{array}{c}
            \sigma(w_1^{\top}x) \\
            \sigma(w_2^{\top}x) \\
            \vdots \\
            \sigma(w_d^{\top}x) \\
        \end{array}\right)

    with :math:`w_1, \ldots, w_d \sim \mathcal{N}(0,I_{\texttt{dim_input}})` and :math:`\texttt{dim_feature} = d`.

    .. note::
        Provided :math:`d \geq \texttt{dim_input}`, the map guarantees :math:`x = \phi^\dag \circ \phi \circ x`.
        The opposite :math:`d \leq \texttt{dim_input}` guarantees :math:`x = \phi \circ \phi^\dag \circ x`. The
        bijection is guaranteed if :math:`d = \texttt{dim_input}`.

    :param num_weights: Number of weights :math:`d` sampled for the Random Features kernel., defaults to 1.
    :type num_weights: int, optional
    :param weights: _Explicit values for the weights may be provided instead of automatically sampling them with the
        provided `num_weights`., defaults to `None`.
    :type weights: Tensor(num_weights, dim_input), optional
    :param weights_trainable: Specifies if the weights are to be considered as trainable parameters during
        backpropagation., default to `False`.
    :type weights_trainable: bool, optional
    """


    def __init__(self, *args, **kwargs):
        super(RandomFeatures, self).__init__(*args, **kwargs)
        self._weights = torch.nn.Parameter(torch.empty(0, dtype=utils.FTYPE),
                                           kwargs.pop('weights_trainable', False))

        weights = kwargs.pop('weights', None)
        if weights is None:
            self.num_weights = int(kwargs.pop('num_weights', 1))
        else:
            self.weights = weights

    @property
    def _weights_exists(self) -> bool:
        r"""
            Returns if this RFF has weights initialized.
        """
        return self._weights.nelement() != 0

    @property
    def num_weights(self) -> int:
        r"""
            Weight dimension :math:`d`, which is half of `dim_feature`.
        """
        return self._weights.shape[0]

    @num_weights.setter
    def num_weights(self, val: int = 1):
        self.weights = torch.nn.init.normal_(torch.empty((val, self.dim_input),
                                                         dtype=utils.FTYPE,
                                                         device=self._weights.device,
                                                         requires_grad=self.weights_trainable))

    @property
    def weights(self) -> Union[torch.nn.Parameter, None]:
        """
            Tensor parameter containing the :math:`w_1, \ldots, w_d`. The first dimension is :math:`d` and the second
            `dim_input`.
        """
        if self._weights_exists:
            return self._weights
        return None

    @weights.setter
    def weights(self, val=None):
        if val is not None:
            assert val.shape[1] == self.dim_input, f"Sample dimension {self.dim_input} incompatible with the " \
                                                   f"supplied weights {val.shape[1]} (dim 1)."
            if isinstance(val, torch.nn.Parameter):
                self._weights = val
            else:
                val = utils.castf(val, tensor=True, dev=self._weights.device)
                self._weights.data = val

                # zeroing the gradients if relevant
                if self.weights_trainable and self._weights.grad is not None:
                    self._weights.grad.x_original.zero_()
            self._logger.debug("The weights has been (re)initialized")
        else:
            self._weights = torch.nn.Parameter(torch.empty(0, dtype=utils.FTYPE),
                                               self.weights_trainable)
            self._logger.info("The weights is unset.")

    @property
    def weights_trainable(self) -> bool:
        r"""
            True if the weights is considered for training during backpropagation, False if considered as a constant.
        """
        return self._weights.requires_grad

    @weights_trainable.setter
    def weights_trainable(self, val: bool = False):
        self._weights.requires_grad = val

    @property
    def dim_feature(self) -> int:
        r"""
        Dimension of the explicit feature map :math:`\texttt{dim_feature} = d`.
        """
        return self.num_weights

    def __str__(self):
        return "Random Features kernel"

    @property
    def hparams_variable(self):
        return {"Dimension": self.num_weights}

    @property
    def hparams_fixed(self):
        return {"Kernel": "Random Features", **super(RandomFeatures, self).hparams_fixed}

    def _explicit_preimage(self, phi) -> torch.Tensor:
        phi *= sqrt(self.num_weights)
        weights_pinv = self._get(key="RSF_weights_piv", level_key="_rsf_piv",
                                 fun=lambda: torch.linalg.pinv(self.weights).T)
        return torch.special.logit(phi, eps=1.e-8) @ weights_pinv

    def _explicit(self, x):
        wx = x @ self.weights.T
        dim_inv_sqrt = 1 / sqrt(self.num_weights)
        return dim_inv_sqrt * torch.sigmoid(wx)

    def after_step(self) -> None:
        self._remove_from_cache('RSF_weights_piv')