"""
Abstract RKM view class.

@author: HENRI DE PLAEN
@copyright: KU LEUVEN
@license: MIT
@date: March 2021
"""

import torch

from .. import utils
from .view import view


@utils.extend_docstring(view)
class primal_view(view):
    r"""
    :param weights: Weights variables
    :type weights: torch.nn.Parameter
    """

    def __init__(self, weights:torch.nn.Parameter, **kwargs):
        """
        A view is made of a kernel and primal or dual variables. This second part is handled by the daughter classes.
        """
        super(primal_view, self).__init__(**kwargs)
        self._weights = weights

        assert len(self._weights.shape) == 2, "The weights must be a matrix."
        self._feature_dim, self._dim_output = self._weights.shape

    @property
    def weights(self):
        r"""
        Primal weights
        """
        if self._weights.nelement() == 0:
            return None
        return self._weights.data

    @weights.setter
    def weights(self, val):
        val = utils.castf(val, tensor=False, dev=self._weights.device)
        if val is not None:
            if self._weights.nelement() == 0:
                self._weights = torch.nn.Parameter(val, requires_grad=self._weights_trainable)
            else:
                self._weights.data = val
                # zeroing the gradients if relevant
                if self._weights_trainable:
                    self._weights.grad.data.zero_()

    @property
    def weights_trainable(self) -> bool:
        return self._weights_trainable

    @weights_trainable.setter
    def weights_trainable(self, val: bool):
        self._weights_trainable = val
        self._weights.requires_grad = self._weights_trainable

    ## MATHS

    def h(self, x=None):
        if x is None:
            return self._weights.data
        raise NotImplementedError

    def w(self, x=None):
        return self._weights

    def wh(self, x=None):
        raise utils.DualError

    def phiw(self, x=None):
        return self.phi(x) @ self.W

    ## MISC

    @staticmethod
    def from_dual(cls, dual_view: view):
        r"""
        Returns a dual view based on a primal one. This is always possible.
        """
        raise NotImplementedError
