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
        \phi(x) = \frac{1}{\sqrt{d}} \left(\begin{array}[c]
            \cos(w_1^{\top}x) \\
            \vdots \\
            \cos(w_d^{\top}x) \\
            \sin(w_1^{\top}x) \\
            \vdots \\
            \sin(w_d^{\top}x) \\
        \end{array}\right)

    with :math:`w_1, \ldots, w_d \sim \mathcal{N}(0,I_{\texttt{dim_input}})` and :math:`\texttt{dim_feature} = 2d`.

    .. info::
        In the limit of :math:`d \rightarrow +\infty`, we recover the RBF kernel :math:`\sigma = 1`:
        .. math::
            k(x,y) = \phi(x)^{\top}\phi(y) = \exp\left( -\frac{1}{2}\lVert x-y \rVert_2^2 \right)
    """

    @utils.kwargs_decorator(
        {"dim_weight": 1,
         "weight": None,
         "weight_trainable": False})
    def __init__(self, **kwargs):
        super(rff, self).__init__(**kwargs)
        self._weight = torch.nn.Parameter(torch.empty(0, dtype=utils.FTYPE),
                                         kwargs["weight_trainable"])

        if kwargs["weight"] is None:
            self.dim_weight = kwargs["dim_weight"]
        else:
            self.weight = kwargs["weight"]

    @property
    def _weight_exists(self) -> bool:
        r"""
        Returns if this RFF has initialized weighteterd.
        """
        return self._weight.nelement() != 0

    @property
    def dim_weight(self) -> int:
        r"""
            Weight dimension :math:`d`, which is half of `dim_feature`.
        """
        return self._weight.shape[0]

    @dim_weight.setter
    def dim_weight(self, val: int = 1):
        self.weight = torch.nn.init.normal_(torch.empty((val, self.dim_input),
                                                       dtype=utils.FTYPE,
                                                       device=self._weight.device,
                                                       requires_grad=self.weight_trainable))

    @property
    def weight(self) -> Union[torch.nn.Parameter, None]:
        """
            Tensor parameter containing the :math:`w_1, \ldots, w_d`. The first dimension is :math:`d` and the second
            `dim_input`.
        """
        if self._weight_exists:
            return self._weight
        return None

    @weight.setter
    def weight(self, val=None):
        if val is not None:
            assert val.shape[1] == self.dim_input, f"Sample dimension {self.dim_input} incompatible with the " \
                                                   f"supplied weight {val.shape[1]} (dim 1)."
            if isinstance(val, torch.nn.Parameter):
                self._weight = val
            else:
                val = utils.castf(val, tensor=True, dev=self._weight.device)
                self._weight.data = val

                # zeroing the gradients if relevant
                if self._weight_trainable and self._weight.grad is not None:
                    self._weight.grad.data.zero_()
            self._debug("The weight has been (re)initialized")
        else:
            self._weight = torch.nn.Parameter(torch.empty(0, dtype=utils.FTYPE),
                                             self.weight_trainable)
            self._log.info("The weight is unset.")

    @property
    def weight_trainable(self) -> bool:
        r"""
            True if the weight is considered for training during backpropagation, False if considered as a constant.
        """
        return self._weight.requires_grad

    @weight_trainable.setter
    def weight_trainable(self, val: bool = False):
        self._weight.requires_grad = val

    @utils.extend_docstring(explicit.dim_feature)
    @property
    def dim_feature(self) -> int:
        return 2 * self.dim_weight

    def __str__(self):
        return "RFF kernel"

    @property
    def params(self):
        return {"Dimension": self.dim_weight}

    @property
    def hparams(self):
        return {"Kernel": "Random Fourier Features", **super(rff, self).hparams}

    def _explicit(self, x=None):
        x = super(rff, self)._explicit(x)
        wx = x @ self.weight.T
        dim_inv_sqrt = 1 / sqrt(self.dim_weight)
        return dim_inv_sqrt * torch.cat((torch.cos(wx),torch.sin(wx)), dim=1)
