# coding=utf-8
"""
File containing the RFF kernel class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: May 2022
"""
from __future__ import annotations
from abc import ABCMeta, abstractmethod
import torch
from math import sqrt, inf
from typing import Union

from ...utils import extend_docstring, FTYPE, castf, BijectionError
from ...feature import Sample
from ..kernel import Kernel


@extend_docstring(Kernel)
class RandomFeatures(Kernel, metaclass=ABCMeta):
    r"""
    .. math::
        \phi(x) = \frac{1}{\sqrt{d}} \left(\begin{array}{c}
            \sigma(w_1^{\top}x / \sigma) \\
            \sigma(w_2^{\top}x / \sigma) \\
            \vdots \\
            \sigma(w_d^{\top}x / \sigma) \\
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
    :param sigma: Bandwidth :math:`\sigma` of the kernel. Defaults to 1.
    :param sigma_trainable: `True` if the gradient of the bandwidth is to be computed. If so, a graph is computed
        and the bandwidth can be updated. `False` just leads to a static computation., defaults to `False`
    :type sigma: float, optional
    :type sigma_trainable: bool, optional
    """

    def __init__(self, *args, **kwargs):
        super(RandomFeatures, self).__init__(*args, **kwargs)
        self._weights = torch.nn.Parameter(torch.empty(0, dtype=FTYPE),
                                           kwargs.pop('weights_trainable', False))

        weights = kwargs.pop('weights', None)
        if weights is None:
            self.num_weights = kwargs.pop('num_weights', 1)
        else:
            self.weights = weights

        # SIGMA
        self._sigma_trainable = kwargs.pop('sigma_trainable', False)
        sigma = torch.tensor(kwargs.pop('sigma', 1.), dtype=FTYPE)
        self._sigma = torch.nn.Parameter(sigma, requires_grad=self._sigma_trainable)

    @property
    def explicit(self) -> bool:
        try:
            return self._num_weights != inf
        except AttributeError:
            return True

    @property
    def sigma(self) -> float:
        r"""
        Bandwidth :math:`\sigma` of the kernel.
        """
        return self._sigma.data.cpu().numpy()

    @sigma.setter
    def sigma(self, val):
        self._reset_cache(reset_persisting=False, avoid_classes=[Sample, RandomFeatures])
        self._sigma.data.copy_(castf(val, tensor=False, dev=self._sigma.device))

    @property
    def sigma_trainable(self) -> bool:
        r"""
        Boolean indicating of the bandwidth is trainable.
        """
        return self._sigma_trainable

    @sigma_trainable.setter
    def sigma_trainable(self, val: bool):
        self._sigma_trainable = val
        self._sigma.requires_grad = self._sigma_trainable

    @property
    def _weights_exists(self) -> bool:
        r"""
        Returns if the weights of the random features has weights initialized or exist (if infinite dimensional).
        """
        return self._weights.nelement() != 0

    @property
    def num_weights(self) -> int | inf:
        r"""
        Weight dimension :math:`d`=`dim_feature`.
        """
        if not self.explicit:
            return inf
        if self._weights_exists:
            return self._num_weights
        return 0

    @num_weights.setter
    def num_weights(self, val: int | 'inf' = 1):
        if val == "inf" or val == inf:
            self._num_weights = inf
            self.weights = None
        elif isinstance(val, float) or isinstance(val, int):
            self._num_weights = int(val)
            self.weights = torch.nn.init.normal_(torch.empty((val, self.dim_input),
                                                             dtype=FTYPE,
                                                             device=self._weights.device,
                                                             requires_grad=self.weights_trainable))
        else:
            raise AttributeError("The number of weights attribute num_weights must be either a positive integer or "
                                 "'inf'.")

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
                                                   f"provided weights {val.shape[1]} (dim 1)."
            val = castf(val, dev=self._weights.device)
            self._weights.data = val

            # zeroing the gradients if relevant
            if self.weights_trainable and self._weights.grad is not None:
                self._weights.grad.data.zero_()
            self._logger.debug("The weights has been (re)initialized")
        else:
            self._weights = torch.nn.Parameter(torch.empty(0, dtype=FTYPE),
                                               self.weights_trainable)
            self._logger.info("The weights are unset.")

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
        return f" (num_weights: {self.num_weights})"

    @property
    def hparams_variable(self):
        return {"Random Features Weights": self.num_weights}

    def _explicit_preimage(self, phi) -> torch.Tensor:
        weights_pinv = self._get(key="Kernel_RF_weights_pinv", level_key="_rsf_piv",
                                 fun=lambda: torch.linalg.pinv(self.weights).T)
        return self.activation_fn_inv(phi * sqrt(self.num_weights)) @ weights_pinv * self.sigma

    def _explicit(self, x) -> torch.Tensor:
        wx = x @ self.weights.T
        fact_sigma = 1 / self.sigma
        fact_weights = 1 / sqrt(self.num_weights)
        return self.activation_fn(wx.mul_(fact_sigma)).mul_(fact_weights)

    @abstractmethod
    def activation_fn(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def activation_fn_inv(self, x: torch.Tensor) -> torch.Tensor:
        raise BijectionError(cls=self, message="The activation function is not invertible.")

    def closed_form_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise Exception("No closed form known for this random feature. Please provide a finite number of weights.")

    def _implicit(self, x, y) -> torch.Tensor:
        if self.explicit:
            phi_x = self._explicit(x)
            phi_y = self._explicit(y)
            return phi_x @ phi_y.T
        dim_inv_sqrt = 1 / self.sigma
        return self.closed_form_kernel(dim_inv_sqrt * x, dim_inv_sqrt * y)

    def after_step(self) -> None:
        self._remove_from_cache('Kernel_RF_weights_pinv')
