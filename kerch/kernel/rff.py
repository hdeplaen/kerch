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

from .. import utils
from .explicit import explicit, base


@utils.extend_docstring(explicit)
class rff(explicit):
    r"""
    Random Fourier Features kernel.

    .. math::
        \phi(x) = \frac{1}{\sqrt{d}} \left(\begin{array}{cc}
            \cos(w_1^{\top}x) \\
            \vdots \\
            \cos(w_d^{\top}x) \\
            \sin(w_1^{\top}x) \\
            \vdots \\
            \sin(w_d^{\top}x) \\
        \end{array}\right)

    with :math:`w_1, \ldots, w_d \sim \mathcal{N}(0,I_{\texttt{dim_input}})` and :math:`\texttt{dim_feature} = 2d`.

    In the limit of :math:`d \rightarrow +\infty`, we recover the RBF kernel with unity bandwidth :math:`\sigma = 1`:

    .. math::
        k(x,y) = \phi(x)^{\top}\phi(y) = \exp\left( -\frac{1}{2}\lVert x-y \rVert_2^2 \right)

    :param num_weights: Number of weights :math:`d` sampled for the RFF., defaults to 1.
    :type num_weights: int, optional
    :param weights: Explicit values for the weights may be provided instead of automatically sampling them with the
        provided `num_weights`., defaults to `None`.
    :type weights: Tensor(num_weights, dim_input), optional
    :param weights_trainable: Specifies if the weights are to be considered as trainable parameters during
        backpropagation., default to `False`.
    :type weights_trainable: bool, optional

    """

    @utils.kwargs_decorator(
        {"num_weights": 1,
         "weights": None,
         "weights_trainable": False})
    def __init__(self, **kwargs):
        super(rff, self).__init__(**kwargs)
        self._weights = torch.nn.Parameter(torch.empty(0, dtype=utils.FTYPE),
                                         kwargs["weights_trainable"])

        if kwargs["weights"] is None:
            self.num_weights = kwargs["num_weights"]
        else:
            self.weights = kwargs["weights"]

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
                    self._weights.grad.data.zero_()
            self._log.debug("The weights has been (re)initialized")
        else:
            self._weights = torch.nn.Parameter(torch.empty(0, dtype=utils.FTYPE),
                                             self.weights_trainable)
            self._log.info("The weights is unset.")

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
        Dimension of the explicit feature map :math:`\texttt{dim_feature} = 2d`.
        """
        return 2 * self.num_weights

    def __str__(self):
        return "RFF kernel"

    @property
    def params(self):
        return {"Dimension": self.num_weights}

    @property
    def hparams(self):
        return {"Kernel": "Random Fourier Features", **super(rff, self).hparams}

    def _explicit(self, x=None):
        x = super(rff, self)._explicit(x)
        wx = x @ self.weights.T
        dim_inv_sqrt = 1 / sqrt(self.num_weights)
        return dim_inv_sqrt * torch.cat((torch.cos(wx),torch.sin(wx)), dim=1)
